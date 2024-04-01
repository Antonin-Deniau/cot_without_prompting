from dotenv import load_dotenv
import os

load_dotenv()

from vllm_model import use_model as use_vllm_model
from cot_decoding import use_cot_utils

context_size = int(os.getenv("CONTEXT_SIZE") or 1024)
top_k = int(os.getenv("TOP_K") or 0)
model = os.getenv("MODEL") or ""


decode, generate_logits = use_vllm_model(model)
greedy_decoding, cot_decoding = use_cot_utils(decode, generate_logits)

question = "<s>[INST] If Jimmy has 100$, give Andy 20$, then Andy double the money and give it back, minus 5$. After that, how much money does Jimmy have in total ? [/INST]"

print("Testing generation of logits")
max_tokens_len = max(len(x) for x in question)
logits = generate_logits(question, context_size - max_tokens_len, 4)
# print(f"Logits: {logits}")

print("Testing greedy decoding")
greedy_answer = greedy_decoding(question)
print(f"Greedy answer: {greedy_answer}")

print("Testing COT decoding")
cot_answer = cot_decoding(question, top_k)
print(f"COT answer: {cot_answer}")
