from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.2-bnb-4bit")


def total_tokens(tokenizer) -> int:
    return len(tokenizer.get_vocab())


print(total_tokens(tokenizer))
