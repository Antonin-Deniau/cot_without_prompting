from transformers import AutoTokenizer
import requests
import json
import numpy as np
from numpy.typing import NDArray

from workspace_types import (
    TokenLogits,
    PromptOutput,
    GenerateLogits,
    Decode,
)


def use_model(model_name: str) -> tuple[Decode, GenerateLogits]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    url = "http://localhost:8000/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }

    context_size = 8192
    vocab = tokenizer.get_vocab()

    def decode(logits: NDArray[np.int_]) -> str:
        return tokenizer.decode(logits)

    def encode(prompt: str) -> NDArray[np.int_]:
        return tokenizer.encode(prompt)

    def token_probs_to_logits(token_probs: dict[str, float]) -> TokenLogits:
        return [
            (
                vocab[token],
                prob,
            )
            for token, prob in token_probs.items()
        ]

    def get_choice_completion_logits(completion) -> PromptOutput:
        top_logprobs = completion["logprobs"]["top_logprobs"]
        all_logits = [token_probs_to_logits(logits) for logits in top_logprobs]
        return all_logits

    def generate_logits(prompt: str, limit: int | None, k: int) -> PromptOutput:
        prompt_token_len = len(encode(prompt)) + 1
        final_limit = (
            min([context_size - prompt_token_len, limit])
            if limit is not None
            else context_size - prompt_token_len
        )

        data = {
            "model": model_name,
            "prompt": prompt,
            "logprobs": 1 if k == 0 else k,
            "max_tokens": final_limit,
            "stop_token_ids": [tokenizer.eos_token_id],
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        return get_choice_completion_logits(response.json()["choices"][0])

    return decode, generate_logits
