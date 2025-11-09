import os
import csv
from random import randrange
import argparse
from functools import partial
import torch
import sys
import time
import re
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextStreamer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer
import torch.nn.utils as nn_utils
from datasets import load_dataset
import evaluate
from datasets import load_dataset
from huggingface_hub import login
login(token='HF-TOKEN') 
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(description="Fine-tune a language model")
parser.add_argument("--model_name", type=str, required=True, help="The name of the model to fine-tune")
parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use")
args = parser.parse_args()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'CAUSAL_LM'
)
def load_model(model_name,bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    n_gpus = torch.cuda.device_count()
    print('number of gpus',n_gpus)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
    

    # Set padding token as EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

model_name = args.model_name 
model, tokenizer = load_model(model_name, bnb_config)


alpaca_prompt_imdb = """Instruction: Read the review below and classify its sentiment as either 'positive' or 'negative'. Only output one of those two words. No explanation.

Review: {}
Answer:"""

alpaca_prompt_news = """Below is an instruction and input pair. Classify the political leaning of the article as 'left', 'center', or 'right', no reasoning required. Give only the political leaning.
### Instruction:
{}
### Article:
{}
### Political Leaning:
{}"""

# Evaluation metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")



if args.dataset_name == 'imdb':
    #test_dataset = load_dataset("stanfordnlp/imdb", split="test")
    test_dataset = load_dataset("stanfordnlp/imdb", split="test") # Select 3000 samples for testing
    label2id = {"negative": 0, "positive": 1}
    print(test_dataset)
    labels_list = list(label2id.keys())

    def make_prompt(example):
        return alpaca_prompt_imdb.format(example["text"], "")

    def get_reference(example):
        return example["label"]

elif args.dataset_name == 'newsarticles':
    dataset = load_dataset('nlpatunt/NewsArticles-Baly-et-al', split='train[:30000]')
    test_dataset = dataset.train_test_split(test_size=0.2, seed=42)["test"]  # Select 3000 samples for testing
    label2id = {"left": 0, "center": 1, "right": 2}
    labels_list = list(label2id.keys())

    def make_prompt(example):
        return alpaca_prompt_news.format(example["instruction"], example["input"], "")

    def get_reference(example):
        return label2id[example["output"].strip().lower()]

else:
    raise ValueError("Invalid dataset name. Use 'imdb' or 'newsarticles'.")

model.eval()
predictions, references = [], []

def extract_label(decoded_output, labels_list):
    # Grab only the part after the word 'sentiment:'
    if args.dataset_name == 'imdb':
        match = re.search(r"answer:\s*(positive|negative)", decoded_output)
        if match:
            decoded_output = match.group(1).strip()
            print(f"Decoded Output:\n{decoded_output}")
        # if "answer:" in decoded_output.lower():
            
        #     decoded_output = decoded_output.lower().split("answer:")[-1].strip()
        #     print(f"Decoded Output:\n{decoded_output}")
        else:
            decoded_output = decoded_output.lower().strip()
            #print(f"Decoded Output:\n{decoded_output}")
    else:
        if "political leaning:" in decoded_output.lower():
            match = re.search(r"political leaning:\s*(\*\*?\s*)?(left|center|right)(\s*\*\*)?", decoded_output)
            if match:
                decoded_output = match.group(2).strip() 
        else:
            decoded_output = decoded_output.lower().strip()
            #print(f"Decoded Output:\n{decoded_output}")
    for label in labels_list:
        if re.fullmatch(label, decoded_output):  # exact match only
            return label
    return "unknown"

with torch.no_grad():
    ### This will generate the output for the test dataset and take some time
    print('Generating output')
    for example in test_dataset:
        prompt = make_prompt(example)
        reference = get_reference(example)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1048).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
        #print('decoded', decoded)  

        pred_label = extract_label(decoded, labels_list)
        #print(f"Predicted Label: {pred_label}, Reference: {reference}")
        predictions.append(pred_label)
        references.append(reference)


# Filter unknown predictions
filtered_preds, filtered_refs, filtered_samples,indices = [], [],[],[]
for idx ,( example, p, r) in enumerate(zip(test_dataset, predictions, references)):
    if p != "unknown":
        filtered_preds.append(label2id[p])
        filtered_refs.append(r)
        filtered_samples.append(example)
        indices.append(idx)
model_id = args.model_name.split("/")[-1]
dataset_name = args.dataset_name
filename = "evaluation_results.csv"

result_df = pd.DataFrame(filtered_samples)
result_df['id'] = indices
result_df['predicted_label'] = filtered_preds
result_df['reference_label'] = filtered_refs

output_filename = f"./Eval-predictions/{model_id}_{args.dataset_name}_predictions.csv"
result_df.to_csv(output_filename, index=False)

print(f"Filtered Predictions: {len(filtered_preds)}, Filtered References: {len(filtered_refs)}")
# Compute metrics

# Saving results to file

acc = accuracy.compute(predictions=filtered_preds, references=filtered_refs)
f1_score = f1.compute(predictions=filtered_preds, references=filtered_refs, average="weighted")

# Save results

# Prepare the row
row = {
    "model_id": model_id,
    "dataset_name": dataset_name,
    "accuracy": acc,
    "f1_score": f1_score
}

# Check if file exists
file_exists = os.path.isfile(filename)

