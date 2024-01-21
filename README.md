<div align="center">

![Logo](https://github.com/connor-henderson/attention-data/assets/78612354/6db331f2-0591-4a3a-9276-fd6c970e9298)

</div>

# AttentionData

**Attention patterns** are somewhat unique in transformers in that they are mappable back to individual tokens [n_ctx n_ctx]. This makes them amenable to visual interpretation. However, these patterns are part of a larger computation, contributing to an internal representation that the model uses to generate the output and we can't take this too literally.

Nonetheless, visualizing and generating data on attention patterns can be beneficial for understanding and interpreting the model's behavior. Here, I've attempted to try if the "dumbest and most automatable" thing of passing this attention data to Language Models (LLMs) with proper prompting could potentially help them identify patterns. Since LLMs are trained on language, they might extract meaningful insights from the attention data, especially if it's presented in a language-like format. I've additionally added a few methods for visualizing attention paid in different contexts and ranking tokens by their attention score's multiple of the average score for their row.

Importantly, the core AttentionData class can be used with any arbitrary combination of dataset (provided it is List[List[str]]), `HookedTransformer` instance, and OpenAI GPT model. Please note this is a hackathon project and thus may contain bugs and is unlikely to be maintained or developed further.

## Usage

[Demo notebook](https://github.com/connor-henderson/attention-data/blob/main/demo.ipynb)

Instantiate an `AttentionData` class with your chosen passed parameters and call any of the following:

- `describe_head`
  - params: `head=0, layer=0, num_samples=10, custom_prompt=None, print_description=True`
  - Creates a prompt based of `num_samples` of examples and returns (prompt, description) where the description is the GPT's guess at the themes of the attention patterns
- `get_ranked_multiples`
  - params: `layer=0, head=0, str_token=None, num_multiples=10, reverse=False, display=False`
  - Returns and displays the top (or bottom if `reverse=True`) `num_multiples` number of tokens with the highest score multiples, optionally pass in a specific str_token to only return instances of that token
- `get_random_multiples`
  - params: `layer=0, head=0, num_multiples=10, display=False`
  - Returns and displays a random `num_multiples` number of tokens
