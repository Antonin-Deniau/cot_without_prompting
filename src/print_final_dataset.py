from datasets import Dataset, load_from_disk
from dotenv import load_dotenv

from model_config import get_config

load_dotenv()

config = get_config()

dataset = load_from_disk("./datasets/final_dataset")
if type(dataset) != Dataset:
    raise ValueError("Expected a dataset, not a dataset dictionary")

dataset = dataset.select(range(min([config.dataset_size, 10])))


for example in dataset:
    if type(example) != dict:
        raise ValueError("Expected a dictionary, not a dataset")

    print("---------------")
    print(f"QUESTION:\n{example['question']}")
    print("---------------")
    print(f"GREEDY:\n{example['greedy_answer']}")
    print("---------------")
    print(f"COT:\n{example['cot_answer']}")
    print()
