import IPython
from openai import OpenAI


# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id

context_size = 8192
dict_token_size = 32_000

prompts = [
    "What is the capital of the United States?",
    "What is 9 + 10?",
]
max_prompt_length = max(len(prompt) for prompt in prompts)

completion = client.completions.create(
    model=model,
    prompt=prompts,
    max_tokens=context_size - max_prompt_length,
    logprobs=dict_token_size,
)

choices = completion.choices
IPython.embed()
