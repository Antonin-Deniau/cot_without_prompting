from model_config import get_config
from vllm_model import use_model as use_vllm_model
from datasets import Dataset, load_from_disk
from cot_decoding import use_cot_utils


config = get_config()
decode, generate_logits = use_vllm_model(config.model)
greedy_decoding, cot_decoding = use_cot_utils(decode, generate_logits)


def q(data: str):
    return config.prompt.format(data=data)


dataset = load_from_disk("./datasets/intermediate_dataset")
if type(dataset) != Dataset:
    raise ValueError("Expected a dataset, not a dataset dictionary")

dataset = dataset.select(range(config.dataset_size))


def generate_answers(example):
    question = q(example["question"])
    greedy_answer = greedy_decoding(question)
    cot_answer = cot_decoding(question, config.top_k)

    example["greedy_answer"] = greedy_answer
    example["cot_answer"] = cot_answer

    return example


dataset = dataset.map(generate_answers)

dataset.save_to_disk("datasets/final_dataset")
