import os
import argparse
from random import randrange
## If specific GPU is needed, uncomment the line below and set the desired GPU ID, this has to be done before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
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
from huggingface_hub import login

parser = argparse.ArgumentParser(description="Fine-tune a language model")
parser.add_argument("--model_n", type=str, required=True, help="The name of the model to fine-tune")
parser.add_argument("--dataset_n", type=str, required=True, help="The name of the dataset to use")
args = parser.parse_args()



login(token='Give your HF token here') 


output_dir = f"{args.model_n.capitalize()}-FT-{args.dataset_n.capitalize()}"

 # BitsAndBytesConfig 
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


# %%
model, tokenizer = load_model(model_name, bnb_config)
# print(tokenizer.model_max_length)
# # %%
model = prepare_model_for_kbit_training(model)

# %%
model = get_peft_model(model, lora_config)


alpaca_prompt = """Below is an QA pair that describes a question, paired with an answer that provides that is the response for the question. Write a response that appropriately answers tthe question.


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


alpaca_prompt1 = """Below is an QA pair that describes a mathematical problem, paired with solution and answer that provides that is the response for the question. Write a response that appropriately answers the question.

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

if args.dataset_n == 'openR1':
    # Load the dataset
    dataset = load_dataset('open-r1/OpenR1-Math-220k', split='train[:10000]')
    # Split the dataset into train and test sets
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    train_dataset = train_dataset.map(formatting_prompts_func_R1, batched = True,)
    num_train_epochs = 2
elif args.dataset_n == 'canadianQA':
    dataset = load_dataset('nlpatunt/canadian-parliamentary-qa',split = 'train[:10000]')
    print(len(dataset))
    train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = train_subset["train"]
    test_dataset = train_subset["test"]
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,) 
    num_train_epochs = 4
else:
    print("Please provide a valid dataset name")   


# dataset = load_dataset('open-r1/OpenR1-Math-220k',split = 'train[:10000]')
# train_subset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
# train_dataset = train_subset["train"]
# test_dataset = train_subset["test"]


print(train_dataset.column_names)

print(train_dataset[0])

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
    train_dataset=train_dataset,
    dataset_text_field = 'text',  # Adjust to correct dataset split
    data_collator=safe_data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=num_train_epochs,
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



