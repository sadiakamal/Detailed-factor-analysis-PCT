import os
from random import randrange
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from functools import partial
import torch
import sys
import time
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
from transformers import LlamaForSequenceClassification, LlamaTokenizer,LlamaModel
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
login(token='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') # Sadia
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
def load_model(model_name, bnb_config):
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
    # device_map = "balanced",
)

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = True)
    

    # Set padding token as EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    #tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

model_name = args.model_name 
model, tokenizer = load_model(model_name, bnb_config)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token






if args.dataset_name == 'finetome':
    num_train_epochs = 2
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:5000]")
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    predictions = []
    references = []
    for example in test_dataset:
        instruction = example["conversations"][0]["value"]
        input_text = "\n".join([turn["value"] for turn in example["conversations"][1:-1]]) if len(example["conversations"]) > 2 else "N/A"
        reference = example["conversations"][-1]["value"]
        # print('reference',reference)

        prompt = alpaca_prompt.format(instruction, input_text, "")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=500)
        # end_time = time.time()    # Record end time
        # generation_time = end_time - start_time
        #print(f"Generation time: {generation_time:.2f} seconds")
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the prediction to get just the response part (after "### Response:")
        response_start = prediction.find("### Response:")
        if response_start != -1:
            prediction = prediction[response_start + len("### Response:"):].strip()
        #print('prediction',prediction)

        predictions.append(prediction)
        references.append([reference]) 
    print(len(train_dataset))
elif args.dataset_name == 'pol-convo':
    dataset = load_dataset('nlpatunt/Political-conversation',split = 'train')
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    predictions = []
    references = []
    for example in test_dataset:
        instruction = example["conversations"][0]["content"]
        input_text = "\n".join([turn["content"] for turn in example["conversations"][1:-1]]) if len(example["conversations"]) > 2 else "N/A"
        reference = example["conversations"][-1]["content"]
        # print('reference',reference)

        prompt = alpaca_prompt.format(instruction, input_text, "")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=500)
        # end_time = time.time()    # Record end time
        # generation_time = end_time - start_time
        #print(f"Generation time: {generation_time:.2f} seconds")
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the prediction to get just the response part (after "### Response:")
        response_start = prediction.find("### Response:")
        if response_start != -1:
            prediction = prediction[response_start + len("### Response:"):].strip()
        #print('prediction',prediction)

        predictions.append(prediction)
        references.append([reference]) 
else:
    print("Please provide a valid dataset name")  

model_id = args.model_name.split('/')[-1]  # Extract the part after the last '/'
filename = f"{model_id.capitalize()}-Eval"


# Load the evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")



# Define the prompt format (same as your fine-tuning format)
finetune_prompt_eval = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token

# Generate predictions on the evaluation set
# for example in test_dataset:
#     instruction = example["conversations"][0]["value"]
#     input_text = "\n".join([turn["value"] for turn in example["conversations"][1:-1]]) if len(example["conversations"]) > 2 else "N/A"
#     reference = example["conversations"][-1]["value"]
#     # print('reference',reference)

#     prompt = finetune_prompt_eval.format(instruction, input_text, "")
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     # start_time = time.time()
#     outputs = model.generate(**inputs, max_new_tokens=500)
#     # end_time = time.time()    # Record end time
#     # generation_time = end_time - start_time
#     #print(f"Generation time: {generation_time:.2f} seconds")
#     prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Clean up the prediction to get just the response part (after "### Response:")
#     response_start = prediction.find("### Response:")
#     if response_start != -1:
#         prediction = prediction[response_start + len("### Response:"):].strip()
#     #print('prediction',prediction)

#     predictions.append(prediction)
#     references.append([reference]) # References should be a list of lists

# Compute BLEU score

test_dataset = test_dataset.add_column('ref', references)
test_dataset = test_dataset.add_column('pred', predictions)

test_dataset.to_csv(f"{filename}.csv", index=False)

bleu_results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score on evaluation set: {bleu_results}")
rouge_results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE score on evaluation set: {rouge_results}")
with open(f"{filename}.txt", 'w') as f:
    f.write(f"BLEU score on evaluation set: {bleu_results}\n")
    f.write(f"ROUGE score on evaluation set: {rouge_results}\n")