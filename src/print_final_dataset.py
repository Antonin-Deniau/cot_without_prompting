from datasets import Dataset, load_from_disk

NUM_EXAMPLES = 1  # 15_000

dataset = load_from_disk("./datasets/final_dataset")
if type(dataset) != Dataset:
    raise ValueError("Expected a dataset, not a dataset dictionary")

dataset = dataset.select(range(NUM_EXAMPLES))


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
