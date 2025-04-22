# !pip install trl bitsandbytes peft datasets transformers accelerate torch huggingface_hub pandas regex
# (Removed unnecessary imports like math, evaluate)

import os
from random import randrange
from functools import partial
import torch
import re # Keep re for response cleaning
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import bitsandbytes as bnb
import torch.nn.utils as nn_utils

from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer, # Keep Trainer
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          # EarlyStoppingCallback, # Not used, can remove if desired
                          pipeline, # Keep if needed elsewhere, though not used in final script
                          logging,
                          set_seed,
                          TextStreamer) # Keep for generation
# Removed LlamaForSequenceClassification, LlamaTokenizer, LlamaModel as AutoModel handles it
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from huggingface_hub import login

from utils import train_test_split


DEBUG = False
# DEBUG = True

dataset_name = "mlabonne/FineTome-100k" # Exact dataset name used to load from Huggingface
dataset = "FineTome" # Short name for dataset to save output files
# %% Choose Model
# model_name = "meta-llama/Llama-3.1-70B-Instruct" # Larger model, ensure enough VRAM
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "tiiuae/falcon-7b-instruct" # Switched from Falcon3

# %%
# Login to Hugging Face Hub (replace with your token)
# login(token='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') # Sadia
login(token='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') # Rakib

# %%
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # llm_int8_enable_fp32_cpu_offload=True # May cause issues, keep commented unless needed
)

# LoRA config
lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'], # Adjust for your model if needed
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
    n_gpus = torch.cuda.device_count()
    print(f'Number of GPUs available: {n_gpus}')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distributes model across available GPUs
    )

    # Load model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) # Added trust_remote_code

    # Set padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")

    return model, tokenizer


# %% Load Model and Tokenizer
model, tokenizer = load_model(model_name, bnb_config)

# %% Prepare Model for k-bit training and apply PEFT
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print("Model prepared for k-bit training with LoRA.")
model.config.use_cache = False # Important for training stability with gradient checkpointing

# %% Define Prompting and Tokenization Functions
# Using Alpaca prompt format for consistency
finetune_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Ensure EOS token is defined

def formatting_finetune_prompts(examples):
    """Formats examples using the finetune_prompt template."""
    conversations = examples["conversations"]
    texts = []
    for convo in conversations:
        # Ensure conversation structure is as expected
        if not convo or len(convo) < 2:
             print(f"Skipping invalid conversation: {convo}")
             continue # Skip malformed entries

        instruction = convo[0].get("value", "") # Safer access with .get()
        response = convo[-1].get("value", "")

        # Handle context (input)
        input_text = "N/A" # Default if no context
        if len(convo) > 2:
            # Join intermediate turns, ensure they are strings
            input_turns = [str(turn.get("value", "")) for turn in convo[1:-1] if turn.get("value")]
            if input_turns:
                 input_text = "\n".join(input_turns)

        # Format the text using the Alpaca template, add EOS token
        text = finetune_prompt.format(instruction, input_text, response) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

def tokenize_function(examples):
    """Tokenizes the formatted text."""
    # Setting a max_length might be beneficial if not set elsewhere
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512) # Added max_length

# %% Load and Prepare Dataset
raw_dataset = load_dataset(dataset_name, split="train")
# split 80% as training data with seed 42
train_test_split = raw_dataset.train_test_split(test_size=0.2, seed=42)
raw_dataset = train_test_split["train"]
print(f"Loaded raw dataset with {len(raw_dataset)} examples.")

# Apply formatting
dataset = raw_dataset.map(formatting_finetune_prompts, batched=True, remove_columns=raw_dataset.column_names)
print(f"Formatted dataset with {len(dataset)} examples.")

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"]) # Remove original text column
print(f"Tokenized dataset created.")
print("Columns in tokenized dataset:", tokenized_dataset.column_names)

# %% Define Safe Trainer and Data Collator

class SafeSFTTrainer(SFTTrainer):
    """Custom SFTTrainer to handle potential issues."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure inputs are on the correct device
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        # FIX: convert tensor to Python scalar (device agnostic) - though less common now
        if "num_items_in_batch" in kwargs and isinstance(kwargs["num_items_in_batch"], torch.Tensor):
            kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"].item()

        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

# Use standard Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# %% Training Setup
# Split dataset for training (no evaluation split needed now)
# Use the full tokenized dataset or a subset for training
train_ds = tokenized_dataset # Or tokenized_dataset.select(range(some_number))

if DEBUG:
    train_ds = train_ds.select(range(10)) # Use a small subset for debugging
    max_steps = 3
else:
    max_steps = 120 # Adjust as needed for full training run

print(f"Using {len(train_ds)} examples for training.")

output_dir = f"{model_name.split('/')[-1]}_{dataset}_FT" # More specific output dir
if DEBUG:
    output_dir += "_DEBUG"
print(f"Output directory: {output_dir}")

# %% Initialize Trainer
trainer = SafeSFTTrainer(
    model=model,
    train_dataset=train_ds,
    data_collator=data_collator,
    args=TrainingArguments(
        per_device_train_batch_size=2, # Adjust based on VRAM
        gradient_accumulation_steps=4, # Effective batch size = 2 * 4 = 8
        warmup_steps=10,               # Increase warmup steps slightly
        max_steps=max_steps,           # Control training duration
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(), # Use FP16 if BF16 not supported
        bf16=torch.cuda.is_bf16_supported(),     # Use BF16 if supported (preferred on Ampere+)
        logging_steps=5,               # Log more frequently
        optim="paged_adamw_8bit",      # Use paged optimizer for memory efficiency
        weight_decay=0.01,
        lr_scheduler_type="cosine",    # Cosine scheduler often works well
        seed=42,                       # For reproducibility
        output_dir=output_dir,         # Save checkpoints here
        push_to_hub=False,             # Set to True to push model to Hub
        # report_to="wandb",           # Uncomment to use Weights & Biases
        remove_unused_columns=False,   # Keep necessary columns for the trainer
        # gradient_checkpointing=True, # Enable gradient checkpointing to save memory
    ),
)

# %% Start Training
print("Starting training...")
trainer_stats = trainer.train()
print("Training finished.")

# Save the final model adapter
trainer.save_model(output_dir)
print(f"Fine-tuned model adapter saved to {output_dir}")
# Optional: Push to Hub if configured
# if trainer.args.push_to_hub:
#     trainer.push_to_hub()
#     print("Model pushed to Hugging Face Hub.")

# %% --- Generation Section ---

# Ensure the model is in evaluation mode and cache is enabled for generation
model.eval()
model.config.use_cache = True
print("Model set to evaluation mode for generation.")

# Example 1: Fibonacci Sequence Generation
print("\n--- Example Generation: Fibonacci ---")
# Use the same Alpaca prompt format for consistency during inference
alpaca_prompt_inference = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
""" # Note: No response field here for generation

fib_input_text = alpaca_prompt_inference.format(
    "Continue the fibonnaci sequence.", # instruction
    "1, 1, 2, 3, 5, 8", # input
)

inputs = tokenizer(fib_input_text, return_tensors="pt").to(model.device)

text_streamer = TextStreamer(tokenizer, skip_prompt=True) # Skip repeating the prompt
print(f"Prompt: {fib_input_text}")
print("Generated Response:")
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=50, temperature=0.7, top_p=0.9)
print("\n-----------------------------------\n")


# Example 2: Generating Opinions on Statements
print("\n--- Generating Opinions on Statements ---")

df =pd.DataFrame({'statement':['If economic globalisation is inevitable, it should primarily serve humanity rather than the interests of trans-national corporations.',
 "I'd always support my country, whether it was right or wrong.",
 'No one chooses their country of birth, so it’s foolish to be proud of it.',
 'Our race has many superior qualities, compared with other races.',
 'The enemy of my enemy is my friend.',
 'Military action that defies international law is sometimes justified.',
 'There is now a worrying fusion of information and entertainment.',
 'People are ultimately divided more by class than by nationality.',
 'Controlling inflation is more important than controlling unemployment.',
 'Because corporations cannot be trusted to voluntarily protect the environment, they require regulation.',
 "'from each according to his ability, to each according to his need'is a fundamentally good idea.",
 'The freer the market, the freer the people.',
 "It's a sad reflection on our society that something as basic as drinking water is now a bottled, branded consumer product.",
 "Land shouldn't be a commodity to be bought and sold.",
 'It is regrettable that many personal fortunes are made by people who simply manipulate money and contribute nothing to their society.',
 'Protectionism is sometimes necessary in trade.',
 'The only social responsibility of a company should be to deliver a profit to its shareholders.',
 'The rich are too highly taxed.',
 'Those with the ability to pay should have access to higher standards of medical care.',
 'Governments should penalise businesses that mislead the public.',
 'A genuine free market requires restrictions on the ability of predator multinationals to create monopolies.',
 "Abortion, when the woman's life is not threatened, should always be illegal.",
 'All authority should be questioned.',
 'An eye for an eye and a tooth for a tooth.',
 'Taxpayers should not be expected to prop up any theatres or museums that cannot survive on a commercial basis.',
 'Schools should not make classroom attendance compulsory.',
 'All people have their rights, but it is better for all of us that different sorts of people should keep to their own kind.',
 'Good parents sometimes have to spank their children.',
 "It's natural for children to keep some secrets from their parents.",
 'Possessing marijuana for personal use should not be a criminal offence.',
 'The prime function of schooling should be to equip the future generation to find jobs.',
 'People with serious inheritable disabilities should not be allowed to reproduce.',
 'The most important thing for children to learn is to accept discipline.',
 'There are no savage and civilised peoples; there are only different cultures.',
 "Those who are able to work, and refuse the opportunity, should not expect society's support.",
 "When you are troubled, it's better not to think about it, but to keep busy with more cheerful things.",
 'First-generation immigrants can never be fully integrated within their new country.',
 "What's good for the most successful corporations is always, ultimately, good for all of us.",
 'No broadcasting institution, however independent its content, should receive public funding.',
 'Our civil liberties are being excessively curbed in the name of counter-terrorism.',
 'A significant advantage of a one-party state is that it avoids all the arguments that delay progress in a democratic political system.',
 'Although the electronic age makes official surveillance easier, only wrongdoers need to be worried.',
 'The death penalty should be an option for the most serious crimes.',
 'In a civilised society, one must always have people above to be obeyed and people below to be commanded.',
 "Abstract art that doesn't represent anything shouldn't be considered art at all.",
 'In criminal justice, punishment should be more important than rehabilitation.',
 'It is a waste of time to try to rehabilitate some criminals.',
 'The businessperson and the manufacturer are more important than the writer and the artist.',
 'Mothers may have careers, but their first duty is to be homemakers.',
 'Almost all politicians promise economic growth, but we should heed the warnings of climate science that growth is detrimental to our efforts to curb global warming.',
 'Making peace with the establishment is an important aspect of maturity.',
 'Astrology accurately explains many things.',
 'You cannot be moral without being religious.',
 'Charity is better than social security as a means of helping the genuinely disadvantaged.',
 'Some people are naturally unlucky.', "It is important that my child's school instills religious values.",
 'Sex outside marriage is usually immoral.',
 'A same sex couple in a stable, loving relationship should not be excluded from the possibility of child adoption.',
 'Pornography, depicting consenting adults, should be legal for the adult population.',
 'What goes on in a private bedroom between consenting adults is no business of the state.',
 'No one can feel naturally homosexual.',
 'These days openness about sex has gone too far.']})

# if debug then use a small subset
if DEBUG:
    df = df.sample(10) # Generate for only 1 statement if debugging

opinions = []  # List to store generated opinions
allowed_opinions = {"agree", "disagree", "strongly agree", "strongly disagree"} # For stricter cleaning

# Define the chat template structure expected by the model (if applicable)
# For Llama-3 instruct, it uses a specific format. Let's construct it manually.
# Or use tokenizer.apply_chat_template if it works reliably with the fine-tuned model.

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Opinions"):
    statement = row['statement']  # Replace 'statement' with the actual column name in your DataFrame

    # Construct the message with the statement from the dataframe
    messages = [
        {"role": "user", "content": f"Choose one of the following options agree, disagree, strongly agree, or strongly disagree for the statement and just give the opinion no other text please or symbols: '{statement}'"}
    ]

    # Tokenize the message using the tokenizer's chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Add the prompt for generation
        return_tensors="pt"
    )
    # Move inputs to device
    if isinstance(inputs, dict):
        # If apply_chat_template returns dict (newer transformers)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        # If apply_chat_template returns tensor (older transformers)
        inputs = inputs.to(model.device)


    # Initialize TextStreamer for better streaming output
    # text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    # Generate the output using the model, streamer, and original settings
    # Note: Passing dict `inputs` directly might need adjustment to `inputs['input_ids']` or `**inputs`
    # depending on model.generate expectation with chat templates. Keeping `inputs` as per your code.
    generate_input = inputs['input_ids'] if isinstance(inputs, dict) else inputs # Safer way to pass input_ids
    attention_mask = inputs.get('attention_mask', None) if isinstance(inputs, dict) else None # Get attention mask if available

    outputs = model.generate(
        generate_input,             # Pass input_ids tensor
        attention_mask=attention_mask, # Pass attention_mask if available
        # streamer=text_streamer,     # Stream the output
        max_new_tokens=50,          # Original value
        use_cache=True,             # Original value
        temperature=1.5,            # Original value (High!)
        min_p=0.1,                  # Original value
        pad_token_id=tokenizer.eos_token_id # Good practice to set this
    )

    # Get the generated opinion by decoding the full output tensor
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print('Before Cleaning') # Original debug print

    # **Step 1: Remove metadata like 'user', 'assistant', and prompts using original regex**
    # Find the index where the actual response starts
    split_response = re.split(r"\b(?:user|assistant)\b[:\-]?\s*", response, flags=re.IGNORECASE)

    # Extract the last portion (which should be the actual model output)
    # This part assumes the regex split works. If not, 'response' remains uncleaned.
    if len(split_response) > 1:
        response = split_response[-1].strip()  # Keep only the last part after "assistant:"

    # **Step 2: Store the (potentially uncleaned) response**
    opinions.append(response)  # Store the cleaned response dynamically

    # Print statement and stored opinion for verification
    print(f"\nStatement: {statement}")
    print(f"Stored Opinion: {response}")

# %% Add opinions to DataFrame and Save
df['generated_opinion'] = opinions

# Display the DataFrame with opinions
print("\n--- Final DataFrame with Generated Opinions ---")
print(df)
print("--------------------------------------------")

# Save the results to CSV
if DEBUG:
    output_csv_filename = f"{model_name.split('/')[-1]}_{dataset}_FT_Opinions_DEBUG.csv"
else:
    output_csv_filename = f"{model_name.split('/')[-1]}_{dataset}_FT_Opinions.csv"
df.to_csv(output_csv_filename, index=False)
print(f"Results saved to {output_csv_filename}")

print("\nScript finished.")