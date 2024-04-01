import numpy as np
from numpy.typing import NDArray
from workspace_types import (
    Decode,
    GenerateLogits,
    TokenLogits,
    PromptOutput,
)


def use_cot_utils(
    decode: Decode,
    generate_logits: GenerateLogits,
):
    def top_k(logits: TokenLogits, k: int) -> NDArray[np.int_]:
        if len(logits) == 1:
            return np.array([logits[0][0], logits[0][0]])
        elif len(logits) == 2:
            return np.array([logits[0][0], logits[1][0]])

        all_tokens: NDArray[np.int_] = np.array(list(map(lambda e: e[0], logits)))
        all_probs: NDArray[np.float_] = np.array(list(map(lambda e: e[1], logits)))

        top_k_indices = np.argpartition(all_probs, -k)[-k:]
        top_k_tokens = all_tokens[top_k_indices]

        return top_k_tokens

    def get_top_two_tokens(token_probs: TokenLogits) -> tuple[int, int]:
        top_two_tokens = top_k(token_probs, 2)
        return top_two_tokens[0], top_two_tokens[1]

    def get_logits_probability(logits: PromptOutput) -> float:
        total_probability_diff = 0
        total_answer_tokens = len(logits)

        for token_probs in logits:
            # Get the top two tokens at each decoding step in the k-th decoding path
            tok1, tok2 = get_top_two_tokens(token_probs)

            probability_diff = tok1 - tok2

            # Add the difference to the total
            total_probability_diff += probability_diff

        # Calculate the average probability difference
        average_probability_diff = total_probability_diff / total_answer_tokens

        return average_probability_diff

    def greedy_decoding(prompt: str) -> str:
        # Generate the logits
        logits = generate_logits(prompt, None, 1)

        # Get the top token at each decoding step
        top_k_tokens: list[int] = [x[0][0] for x in logits]

        return decode(np.array(top_k_tokens))

    def cot_decoding(prompt: str, k: int) -> str:
        # Generate the topk start token
        first_token_logits = generate_logits(prompt, 1, k)[0]
        top_k_tokens = top_k(first_token_logits, k)

        probabilities: list[tuple[float, int, PromptOutput]] = []

        # Generate every continuation for each topk token
        for i in range(1, k):
            x = top_k_tokens[i]
            token = top_k_tokens[i]

            continue_logits = generate_logits(prompt + decode(x), None, i)
            topk_probability = get_logits_probability(continue_logits)

            probabilities.append((topk_probability, token, continue_logits))

        # Sort the results by the average probability difference
        probabilities.sort(key=lambda x: x[0], reverse=True)

        # Get the top decoding path
        top_logits = probabilities[0][2]
        first_token = probabilities[0][1]

        # Concat the first token to the prompt
        response_tokens = [first_token]

        for token_probs in top_logits:
            # Get the top token at each decoding step in the k-th decoding path
            top_token = top_k(token_probs, 1)[0]
            # Decode the token and add it to the decoded text
            response_tokens.append(top_token)

        return decode(np.array(response_tokens))

    return (greedy_decoding, cot_decoding)
