from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

# 1. Load dataset (CPU only)
ds = load_dataset("mlabonne/FineTome-100k", split="train")

# 2. Load tokenizer (CPU only)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    use_fast=True,
    padding_side="left"
)

# 3. Helpers to build prompt & extract response
def make_prompt(conversations):
    instr = conversations[0]["value"]
    inp = "\n".join(turn["value"] for turn in conversations[1:-1]) if len(conversations) > 2 else "N/A"
    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instr}\n\n"
        f"### Input:\n{inp}\n\n"
        "### Response:\n"
    )

def make_response(conversations):
    return conversations[-1]["value"].strip()

# 4. Compute lengths with a progress bar
prompt_lens = []
response_lens = []

for ex in tqdm(ds, desc="Tokenizing examples"):
    conv = ex["conversations"]
    prompt_ids = tokenizer(make_prompt(conv), truncation=True)["input_ids"]
    resp_ids   = tokenizer(make_response(conv), truncation=True)["input_ids"]
    prompt_lens.append(len(prompt_ids))
    response_lens.append(len(resp_ids))

# 5. Print statistics
print(
    f"Prompt lengths  →  Avg: {np.mean(prompt_lens):.1f}, "
    f"Min: {np.min(prompt_lens)}, Max: {np.max(prompt_lens)}"
)
print(
    f"Response lengths  →  Avg: {np.mean(response_lens):.1f}, "
    f"Min: {np.min(response_lens)}, Max: {np.max(response_lens)}"
)

###
# Prompt lengths  →  Avg: 287.5, Min: 43, Max: 11529
# Response lengths  →  Avg: 316.1, Min: 2, Max: 5221