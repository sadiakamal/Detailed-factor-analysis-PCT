import os
import argparse
from random import randrange
from functools import partial
import torch
import sys

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

parser = argparse.ArgumentParser(description="Fine-tune a language model")
parser.add_argument("--model_n", type=str, required=True, help="The name of the model to fine-tune")
parser.add_argument("--dataset_n", type=str, required=True, help="The name of the dataset to use")
args = parser.parse_args()


from huggingface_hub import login
login(token='Give your HF token here') # Sadia

output_dir = f"{args.model_n.capitalize()}-FT-{args.dataset_n.capitalize()}"



 # BitsAndBytesConfig int-4 config
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

# %%
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

    return model, tokenizer

if args.model_n == 'mistral':
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model_n == 'llama3':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model_n == 'falcon':
    model_name = "tiiuae/Falcon3-7B-Instruct"
elif args.model_n == 'phi4':
    model_name = "microsoft/phi-4"
elif args.model_n == 'gemma':
    model_name = "google/gemma-3-4b-it"
else:
    print("Please provide a valid model name")
    sys.exit(1)

# Load model and tokenizer
model, tokenizer = load_model(model_name, bnb_config)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Article:
{}

### Political Leaning:
{}"""

EOS_TOKEN = tokenizer.eos_token

#### For News Articles dataset
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

alpaca_prompt2 = """Below are movie review and sentiment pair. Sentiment can be positive or negative. Write a response that appropriately completes the request.
### Review:
{}
### Sentiment:
{}"""

def formatting_prompts_func_imdb(examples):
    inputs = examples["text"]
    labels = examples["label"]
    texts = []
    
    for input, label in zip(inputs, labels):
        # Convert numerical label to text
        sentiment = "positive" if label == 1 else "negative"
        
        # Format the prompt with review and sentiment
        text = alpaca_prompt2.format(input, sentiment) + EOS_TOKEN
        texts.append(text)
        
    return {"text": texts}


if args.dataset_n == 'newsarticles':
    dataset = load_dataset('nlpatunt/NewsArticles-Baly-et-al',split = 'train[:30000]')
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
elif args.dataset_n == 'imdb':
    train_dataset = load_dataset("stanfordnlp/imdb", split="train")
    test_dataset = load_dataset("stanfordnlp/imdb", split="test")
    train_dataset = train_dataset.map(formatting_prompts_func_imdb, batched=True)
else:
    print("Please provide a valid dataset name")   



from trl import SFTTrainer

class SafeSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # FIX: convert tensor to Python scalar (device agnostic)
        if "num_items_in_batch" in kwargs and isinstance(kwargs["num_items_in_batch"], torch.Tensor):
            kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"].item()

        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)




from transformers import DataCollatorForLanguageModeling
def safe_data_collator(features):
    # Collate the batch normally first
    collated = DataCollatorForLanguageModeling(tokenizer, mlm=False)(features)
    return collated  # Keep tensors on CPU, let Trainer handle transfer



trainer = SafeSFTTrainer(
    model=model,
    processing_class=tokenizer,
    max_seq_length= 2048,
    train_dataset=train_dataset,  # Adjust to correct dataset split
    dataset_text_field = 'text',
    data_collator=safe_data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        push_to_hub=True,
        seed=3407,
        output_dir=output_dir,
    ),
)


trainer_stats = trainer.train()


