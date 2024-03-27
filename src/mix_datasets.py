from datasets import load_dataset, interleave_datasets

EXAMPLES_AMOUNT = 15_000

orca_math = load_dataset(
    "microsoft/orca-math-word-problems-200k",
    split="train[:{}]".format(EXAMPLES_AMOUNT),
)
open_orca = load_dataset(
    "Open-Orca/OpenOrca",
    split="train[:{}]".format(EXAMPLES_AMOUNT),
)
rag_dataset = load_dataset(
    "neural-bridge/rag-dataset-12000",
    split="train[:{}]".format(EXAMPLES_AMOUNT),
)

# Limit and only take the necessary columns
orca_math = orca_math.select_columns("question")
open_orca = open_orca.select_columns("question")
rag_dataset = rag_dataset.select_columns(["context", "question"])


# Format the rag dataset
QUESTION_TEMPLATE = """\
Context:
{context}

Question:
{question}
"""


def format_prompt(example):
    example["question"] = QUESTION_TEMPLATE.format(
        context=example["context"],
        question=example["question"],
    )
    return example


rag_dataset = rag_dataset.map(format_prompt)

rag_dataset = rag_dataset.select_columns("question")

# Load the datasets
seed = 666
probabilities = [0.3, 0.4, 0.3]
dataset = interleave_datasets(
    [orca_math, open_orca, rag_dataset],
    probabilities=probabilities,
    seed=seed,
)

# Save the intermediate dataset
dataset.save_to_disk("datasets/intermediate_dataset")
