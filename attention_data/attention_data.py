import plotly.express as px
import torch as t
from typing import List
from typing_extensions import Literal
from IPython.display import display
from transformer_lens import HookedTransformer, ActivationCache
import openai
import random
from IPython.display import display, HTML


class HeadData:
    def __init__(self):
        self.descriptions = []
        self.samples = None
        self.ranked_multiples = None
        

class AttentionData:
    def __init__(self, 
                 model: HookedTransformer, 
                 openai_model: Literal["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-4", "gpt-4-32k"],
                 openai_api_key: str,
                 text_batch: List[List[str]],
                 min_length = 10,
                 max_length = 30
                 ) -> None:
        self.model = model
        self.openai_model = openai_model
        self.OPENAI_API_KEY = openai_api_key
        self.heads = {layer_idx: {head_idx: HeadData() for head_idx in range(self.model.cfg.n_heads)} for layer_idx in range(self.model.cfg.n_layers)}
        self.min_length = min_length
        self.max_length = max_length
        self.text_batch = text_batch
        self.str_tokens = None
        self.input_ids = None
        self.cache: ActivationCache = None
        self.pad_token_id = model.tokenizer.pad_token_id
        self.prompt_prefix = ("Here is a row by row break down of which tokens were attended to for each individual generation step for a transformer. "
                              "Each row is labeled with its index and is of the format for each row: `\{text\}\\n{{token} {multiple} for token, multiple in above_avg_tokens}`. "
                              "Notably, the tokens with above average attention are sorted by their score's multiple relative to the average attention score for that row, "
                              "and this multiple is included next to it, i.e. `tokenX 2, tokenY 1.5` etc.")
        self.prompt_suffix = ("Please generate a response that summarizes what this attention head pays attention to, "
                             "generalizing from the specific contexts for each example. Be as specific as possible, considering all possible "
                             "reasons certain tokens may have been attended to more than others, including the position of "
                             "the tokens, what the tokens themselves represent, the attention scores, and how these all interact. "
                             "Do NOT explain how attention heads or transformers work or include highly general information. "
                             "Limit your response to 1 paragraph.")
    
    def _create_cache(self):
        input_features = self.model.tokenizer(self.text_batch, padding=True, return_tensors="pt")
        input_ids = input_features["input_ids"]
        if input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
        decoded = self.model.tokenizer.batch_decode(input_ids)
        str_tokens = self.model.to_str_tokens(decoded, prepend_bos=False)
        _, cache = self.model.run_with_cache(input_ids)
        
        self.str_tokens = str_tokens
        self.input_ids = input_ids
        self.cache = cache
        
    def create_samples(self, head=0, layer=0):
        print(f"Creating new samples for layer {layer} head {head}")
        if self.cache is None:
            self._create_cache()
        # Calculate the average score multiples
        attention_patterns = self.cache["pattern", layer][:, head, ...]
        avg_scores = 1 / t.arange(1, attention_patterns.shape[1] + 1).float().to(attention_patterns.device)
        avg_scores = avg_scores.unsqueeze(-1)
        attention_multiples = attention_patterns / avg_scores
        
        # Create the samples of [((seq_idx, length_idx), score_multiples1), ...]
        samples = []
        for seq_idx in range(attention_patterns.shape[0]):
            seq_ids = self.input_ids[seq_idx].tolist()
            pad_token_idx = seq_ids.index(self.pad_token_id) if self.pad_token_id in seq_ids else float('inf')
            for length_idx in range(self.min_length, min(pad_token_idx, attention_patterns.shape[1])):
                if length_idx >= pad_token_idx: 
                    break
                str_tokens = self.str_tokens[seq_idx][:length_idx+1]
                score_multiples = attention_multiples[seq_idx, length_idx, :length_idx+1].tolist()
                score_multiples = [(str_token, round(multiple, 1)) for str_token, multiple in zip(str_tokens, score_multiples) if multiple > 1]
                score_multiples.sort(key=lambda x: x[1])
                samples.append(((seq_idx, length_idx), score_multiples))

        head_data = self.heads[layer][head]
        head_data.samples = samples
        head_data.ranked_multiples = None
    
    def get_head_samples(self, head=0, layer=0, num_samples=10):
        head_data = self.heads[layer][head]
        if head_data.samples is None:
            self.create_samples(head=head, layer=layer)
        samples = head_data.samples
        sample_indices = random.sample(range(len(samples)), num_samples)
        rows = []
        for i in sample_indices:
            (seq_idx, length_idx), score_multiples = samples[i]
            token_ids = self.input_ids[seq_idx][:length_idx+1]
            text = self.model.tokenizer.decode(token_ids)
            above_avg_tokens = ", ".join([f"{token} {multiple}" for token, multiple in score_multiples])
            row = f"{text}\n{above_avg_tokens}\n"
            rows.append(row)
        
        return "\n".join(rows)
    
    def describe_head(self, head=0, layer=0, num_samples=10, custom_prompt=None, print_description=True):
        # Retrieve samples
        head_data = self.heads[layer][head]
        if head_data.samples is None:
            self.create_samples(head=head, layer=layer)
        samples = self.get_head_samples(head=head, layer=layer, num_samples=num_samples)
        
        # Create prompt
        prompt = f"{self.prompt_prefix}\n{samples}"
        if custom_prompt:
            prompt += f"\n{custom_prompt}"
        prompt += f"\n{self.prompt_suffix}"
        
        # Send prompt and save result
        print(f"Making API call to {self.openai_model}...\n")
        description = self._prompt_gpt(prompt)
        if print_description:
            for line in description.split('\n'):
                print("\n".join(line[i:i+80] for i in range(0, len(line), 80)))
        head_data.descriptions.append(description)
        
        return prompt, description
        
    
    def _prompt_gpt(self, prompt: str) -> str:
        content = "You are a helpful mechanistic interpretability researcher who is an expert in analyzing attention patterns"
        response = openai.ChatCompletion.create(model=self.openai_model, messages=[{"role": "system", "content": content}, {"role": "user", "content": prompt}])
        response_content = response['choices'][0]['message']['content']
        return response_content
    
    def _create_ranked_multiples(self, layer=0, head=0):
        head_data = self.heads[layer][head]
        if head_data.samples is None:
            self.create_samples(head=head, layer=layer)
    
        ranked_multiples = []
        for sample in head_data.samples:
            (seq_idx, length_idx), multiples = sample
            for (token, multiple) in multiples:
                str_toks = self.str_tokens[seq_idx][:length_idx+1]
                attention_pattern = self.cache["pattern", layer][seq_idx, head, length_idx, :length_idx+1]
                ranking = [token, multiple, str_toks, attention_pattern]
                ranked_multiples.append(ranking)
                
        ranked_multiples.sort(key=lambda x: x[1], reverse=True)
        head_data.ranked_multiples = ranked_multiples
    
    def get_ranked_multiples(self, layer=0, head=0, str_token=None, num_multiples=10, reverse=False, display=False):
        head_data = self.heads[layer][head]
        if head_data.ranked_multiples is None:
            self._create_ranked_multiples(layer=layer, head=head)
        if reverse:
            multiples = head_data.ranked_multiples[-num_multiples:]
        else:
            multiples = head_data.ranked_multiples[:num_multiples]
        
        # Optionally only retrieve certain tokens
        if str_token is not None:
            count = 0
            multiples = []
            for multiple in head_data.ranked_multiples:
                ranked_str_token, *_ = multiple
                if str_token == ranked_str_token:
                    multiples.append(multiple)
                    count += 1
                if count >= num_multiples:
                    break
        
        if display:
            title = f"Layer {layer} Head {head}, Top {num_multiples} / {len(head_data.ranked_multiples)} Multiples"
            if str_token:
                title += f" for '{str_token}'"
            self._display_rows(multiples, title=title)
        return multiples
    
    def get_random_multiples(self, layer=0, head=0, num_multiples=10, display=False):
        head_data = self.heads[layer][head]
        if head_data.ranked_multiples is None:
            self._create_ranked_multiples(layer=layer, head=head)
        
        indices = random.sample(range(len(head_data.ranked_multiples)), num_multiples)
        random_multiples = [head_data.ranked_multiples[i] for i in indices]
        
        if display: 
            title = f"Layer {layer} Head {head}, {num_multiples} / {len(head_data.ranked_multiples)} Random Multiples"
            self._display_rows(random_multiples, title=title)
        return random_multiples
    
    def _display_rows(self, rows, title=""):
        html_str = "<table style='border:1px solid black;'>"
        # Add title row if title is not empty
        if title:
            html_str += f"<tr><th colspan='10' style='text-align:center'>{title}</th></tr>"
        # Add column headers
        html_str += "<tr><th>Token</th><th>Multiple of Avg. score</th><th>Pattern</th></tr>"
        for token, multiple, str_tokens, attention_pattern in rows:
            html_str += "<tr>"
            html_str += f"<td style='border:1px solid black;'>{token}</td>"
            html_str += f"<td style='border:1px solid black;'>{multiple}</td>"

            # Add pattern cells
            for i, p in enumerate(attention_pattern.tolist()):
                # Calculate color intensity based on pattern value (assuming pattern value is between 0 and 1)
                intensity = int(p * 255)
                html_str += f"<td style='border:1px solid black; background-color:rgb({intensity}, 0, 0);'>{str_tokens[i].strip()}</td>"  # Changed color to red
            html_str += "</tr>"
        html_str += "</table>"
        display(HTML(html_str))