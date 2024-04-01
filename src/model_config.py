from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Config:
    context_size: int
    top_k: int
    model: str
    prompt: str
    dataset_size: int


def get_config():
    context_size = os.getenv("CONTEXT_SIZE")
    if context_size is None:
        raise ValueError("CONTEXT_SIZE must be set in the environment")

    top_k = os.getenv("TOP_K")
    if top_k is None:
        raise ValueError("TOP_K must be set in the environment")

    model = os.getenv("MODEL")
    if model is None:
        raise ValueError("MODEL must be set in the environment")

    prompt = os.getenv("PROMPT")
    if prompt is None:
        raise ValueError("PROMPT must be set in the environment")

    dataset_size = os.getenv("DATASET_SIZE")
    if dataset_size is None:
        raise ValueError("DATASET_SIZE must be set in the environment")

    return Config(
        context_size=int(context_size),
        top_k=int(top_k),
        model=model,
        prompt=prompt,
        dataset_size=int(dataset_size),
    )
