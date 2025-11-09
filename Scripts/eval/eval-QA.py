import os
import csv
import re
from random import randrange
import argparse
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
from transformers import LlamaForSequenceClassification, LlamaTokenizer, LlamaModel
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

def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """

    # Get number of GPU device and set maximum memory
    n_gpus = torch.cuda.device_count()
    print('number of gpus', n_gpus)
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="auto",
    # device_map = "balanced",
    )

    # Load model tokenizer with the user authentication token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    
    # Set padding token as EOS token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    return model, tokenizer

model_name = args.model_name 
model, tokenizer = load_model(model_name, bnb_config)


alpaca_prompt = """Below is an QA pair that describes a question, paired with an answer that provides that is the response for the question. Write a response that appropriately answers the question no explanation.


### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    inputs       = examples["Question"]
    outputs      = examples["Answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }


alpaca_prompt1 = """Below is an QA pair that describes a mathematical problem, paired with solution and answer that provides that is the response for the question. Write a response that appropriately answers the question only.

# ### Problem:
# {}

# ### Input:
# {}

# ### Response:
# {}"""


def formatting_prompts_func_R1(examples):
    problems = examples["problem"]
    inputs       = examples["solution"]
    outputs      = examples["answer"]
    texts = []
    for problem, input, output in zip(problems, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt1.format(problem, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

def clean_prediction(prediction: str) -> str:
    """
    Extract clean answer from model prediction.
    Prioritize LaTeX or numeric answers.
    """
    # Remove leading label text like "Answer:", "### Response:", etc.
    prediction = prediction.strip()

    # Try extracting LaTeX or boxed answers
    match = re.search(r"\$\s*([^$]+)\s*\$", prediction)  # anything inside single dollar signs
    if match:
        return f"${match.group(1).strip()}$"

    # Try extracting boxed answer
    match = re.search(r"\\boxed\{([^}]+)\}", prediction)
    if match:
        return match.group(1).strip()

    # Try extracting number after common patterns
    match = re.search(r"Answer:.*?(-?\d+(?:\.\d+)?)", prediction)
    if match:
        return match.group(1)

    # Fallback to the last number or expression
    match = re.findall(r"-?\d+(?:\.\d+)?|\$[^\$]+\$", prediction)
    if match:
        return match[-1].strip()

    return prediction.strip()


if args.dataset_name == 'openR1':
    # Load the dataset
    dataset = load_dataset('open-r1/OpenR1-Math-220k', split='train[:10000]')
    # Split the dataset into train and test sets
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    test_dataset = test_dataset.select(range(1000))
    # print(test_dataset[0])
    predictions = []
    references = []

    for example in test_dataset:
        problem = example["problem"]
        input_text = example["solution"]
        reference = example["answer"]
        prompt = alpaca_prompt1.format(problem, input_text, "")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50,pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
       

        # Clean up the prediction to get just the response part (after "### Response:")
        response_start = prediction.find("### Response:")
        if response_start != -1:
            prediction = prediction[response_start + len("### Response:"):].strip()
            # match = re.search(r"#\s*\*\*Answer:\*\*\s*#?\s*(.*?)$", prediction, re.DOTALL)
            # if match:
            #     prediction = match.group(1).strip()
            # else:
            #     prediction = prediction.strip()
        cleaned_pred = clean_prediction(prediction)
        predictions.append(cleaned_pred)

        #predictions.append(prediction)
        references.append([reference])
        print('prediction\n',cleaned_pred) 
        print('reference\n', reference)
    # train_dataset = train_dataset.map(formatting_prompts_func_R1, batched = True,)
    
elif args.dataset_name == 'canadianQA':
    dataset = load_dataset('nlpatunt/canadian-parliamentary-qa',split = 'train[:10000]')
    print(len(dataset))
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    test_dataset = test_dataset.select(range(1000))
    print(len(test_dataset))
    # print(test_dataset[0])
    predictions = []
    references = []
    for example in test_dataset:
        input_text = example["Question"]
        reference = example["Answer"]
        prompt = alpaca_prompt.format(input_text, "")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200,pad_token_id=tokenizer.eos_token_id)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the prediction to get just the response part (after "### Response:")
        response_start = prediction.find("### Response:")
        if response_start != -1:
            prediction = prediction[response_start + len("### Response:"):].strip()
        
        print('prediction\n', prediction)
        print('reference\n', reference)

        predictions.append(prediction)
        references.append([reference])
    # train_dataset = train_dataset.map(formatting_prompts_func, batched = True,
else:
    print("Please provide a valid dataset name")   


# model_id = args.model_name.split('/')[-1]  # Extract the part after the last '/'
# filename = f"{model_id.capitalize()}-Eval"
model_id = args.model_name.split('/')[-1]  # Extract the part after the last '/'
filename = "evaluation_results_QA.csv"

# Load the evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
exact_match_metric = evaluate.load("exact_match")


# Compute exact match score
flattened_references = [ref[0] for ref in references]  # Extracting the first element from each list

# Compute the exact match score
em_results = exact_match_metric.compute(predictions=predictions, references=flattened_references)
print(f"Exact Match Score on evaluation set: {em_results['exact_match']:.4f}")

# Save results to file
test_dataset = test_dataset.add_column('ref', references)
test_dataset = test_dataset.add_column('pred', predictions)
output_filename = f"./Eval-predictions/{model_id}_{args.dataset_name}_predictions.csv"
test_dataset.to_csv(f"{output_filename}", index=False)



bleu_results = bleu.compute(predictions=predictions, references=references)
print(f"BLEU score on evaluation set: {bleu_results}")
rouge_results = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE score on evaluation set: {rouge_results}")
bertscore_results = bertscore.compute(predictions=predictions, references=references, lang="en")
bertscore_precision = float(np.mean(bertscore_results["precision"]))
bertscore_recall = float(np.mean(bertscore_results["recall"]))
bertscore_f1 = float(np.mean(bertscore_results["f1"]))
print(f"BERTScore on evaluation set: {bertscore_precision}, {bertscore_recall}, {bertscore_f1}")
row = {
    "model_id": model_id,
    "dataset_name": args.dataset_name,
    "BLEU": bleu_results["bleu"],
    "ROUGE-1": rouge_results["rouge1"],
    "ROUGE-2": rouge_results["rouge2"],
    "ROUGE-L": rouge_results["rougeL"],
    "BERTScore_P": bertscore_precision,
    "BERTScore_R": bertscore_recall,
    "BERTScore_F1": bertscore_f1
}


