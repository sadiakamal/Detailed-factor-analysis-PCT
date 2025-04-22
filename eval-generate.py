# !pip install torch transformers datasets evaluate peft bitsandbytes accelerate sentencepiece trl rouge_score bert_score pandas

import os
import gc
import torch
import pandas as pd # Added pandas for CSV output
import evaluate  # Hugging Face Evaluate library

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer, # Use the base Trainer for evaluation
    TrainingArguments, # Need minimal args for Trainer
    DataCollatorForLanguageModeling,
    TextStreamer,
    logging,
    set_seed
)
from peft import PeftModel, PeftConfig # Import PeftModel to load the adapter
from tqdm import tqdm


DEBUG = False # Set to True to run on a small subset for testing
# DEBUG = True

# Set seed for reproducibility if needed
set_seed(42)
logging.set_verbosity_info()

# --- Configuration ---
base_model_name = "meta-llama/Llama-3.1-8B-Instruct" # Or "tiiuae/Falcon3-7B-Instruct" or your base model
adapter_path = "Llama-3.1-8B-Instruct_FineTome_FT"  # Directory where the fine-tuned adapter was saved by trainer.save_model()
dataset_name = "mlabonne/FineTome-100k" # Dataset used for fine-tuning
eval_batch_size = 32 # Adjust based on GPU memory for evaluation
max_new_tokens_generation = 512 # Max tokens for generation during evaluation

# --- Hugging Face Login (Using the token from your original code) ---
from huggingface_hub import login
try:
    # Explicitly use the token provided in the original script
    # login_token='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' # Sadia
    login_token='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' # Rakib
    print(f"Attempting Hugging Face login with provided token...")
    login(token=login_token)
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    # Depending on the model visibility, this might cause issues later

# --- Quantization Configuration (Should match training) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # llm_int8_enable_fp32_cpu_offload=True # Optional, might need adjustment
)

# --- Load Base Model and Tokenizer ---
print(f"Loading base model: {base_model_name}")
# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute across available GPUs
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True # Important for some models like Falcon
    # use_auth_token=True might be implicitly handled by login(), but can add if needed
)

print(f"Loading tokenizer for: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    padding_side='left',
    trust_remote_code=True
    # use_auth_token=True might be implicitly handled by login(), but can add if needed
)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad_token to eos_token")
else:
     # If pad token exists, still ensure padding side is left
     if tokenizer.padding_side != 'left':
         print(f"Tokenizer has pad_token, explicitly setting padding_side='left' (was '{tokenizer.padding_side}').")
         tokenizer.padding_side = 'left'

# --- Load PEFT Adapter ---
print(f"Loading PEFT adapter from: {adapter_path}")
# Load the fine-tuned PEFT model (adapter) on top of the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
print("Successfully loaded PEFT adapter.")


model.eval() # Set the model to evaluation mode

# --- Dataset Loading and Preparation ---

# Define the prompt formatting function (must match the one used in training)
alpaca_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
EOS_TOKEN = tokenizer.eos_token

def formatting_finetune_prompts(examples):
    conversations = examples["conversations"]
    texts = []
    for convo in conversations:
        instruction = convo[0]["value"]
        response = convo[-1]["value"]
        input_text = "\n".join([turn["value"] for turn in convo[1:-1]]) if len(convo) > 2 else "N/A"
        # Format the text using the Alpaca template, leaving response blank for the prompt part
        text = alpaca_prompt_template.format(instruction, input_text, response) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def tokenize_function(examples):
    # Tokenize the full text including the response for perplexity calculation
    return tokenizer(examples["text"], truncation=True, padding=False) # Don't pad here, collator will handle it

def split_prompt_and_ref(example):
    # This function separates the prompt from the reference answer for generation evaluation
    conversations = example["conversations"]
    instruction = conversations[0]["value"]
    response = conversations[-1]["value"] # The actual ground truth response
    input_text = "\n".join([turn["value"] for turn in conversations[1:-1]]) if len(conversations) > 2 else "N/A"
    # Create the prompt *without* the response
    prefix = alpaca_prompt_template.format(instruction, input_text, "")
    ref = response.strip()
    return {"prompt": prefix, "reference": ref}

# Load the raw dataset
print(f"Loading raw dataset: {dataset_name}")
raw_dataset = load_dataset(dataset_name, split="train") # Load the split used for training

# Split the *raw* dataset to get the same evaluation set used during training
print("Splitting dataset (using seed 42)...")
split_raw = raw_dataset.train_test_split(test_size=0.1, seed=42)
eval_raw_ds = split_raw["test"]

# Prepare the dataset for perplexity calculation (needs full text tokenized)
print("Formatting dataset for perplexity calculation...")
eval_formatted_ds = eval_raw_ds.map(formatting_finetune_prompts, batched=True)
eval_tokenized_ds = eval_formatted_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_formatted_ds.column_names # Remove old columns
)
print("Columns in tokenized dataset for perplexity:", eval_tokenized_ds.column_names)


# Prepare the dataset for generation evaluation (needs prompt and reference separated)
print("Preparing dataset for generation evaluation (prompt/reference split)...")
eval_paired_ds = eval_raw_ds.map(
    split_prompt_and_ref,
    remove_columns=eval_raw_ds.column_names # Remove old columns
)
prompts = eval_paired_ds["prompt"]
references = eval_paired_ds["reference"]
print(f"Prepared {len(prompts)} prompts and references for generation.")

# Shorten evaluation if debugging
if DEBUG:
    print("--- DEBUG MODE ENABLED: Using small subset for evaluation ---")
    debug_size = 10
    eval_tokenized_ds = eval_tokenized_ds.select(range(debug_size))
    eval_paired_ds = eval_paired_ds.select(range(debug_size))
    prompts = prompts[:debug_size]
    references = references[:debug_size]


# --- Perplexity Calculation ---
print("Calculating Perplexity...")
# We need a Trainer instance to use its evaluate method
# Define minimal TrainingArguments needed for evaluation
eval_args = TrainingArguments(
    output_dir="./eval_output", # Temporary directory
    per_device_eval_batch_size=1,
    logging_steps=10,
    report_to="none", # Don't report to wandb/tensorboard
    fp16=not torch.cuda.is_bf16_supported(), # Match training settings if possible
    bf16=torch.cuda.is_bf16_supported(),
)

# Data collator for language modeling (handles padding)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Instantiate Trainer
trainer = Trainer(
    model=model, # Use the loaded PEFT model
    args=eval_args,
    eval_dataset=eval_tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Run evaluation
# eval_results = trainer.evaluate()
# eval_loss = eval_results["eval_loss"]
# perplexity = math.exp(eval_loss)
# print(f"Evaluation Loss: {eval_loss:.4f}")
# print(f"Perplexity: {perplexity:.4f}") # Perplexity = 2.2654 & Eval Loss = 0.8177 for Llama-3.1-8B-Instruct on 10% (seed 42) test split (10k) of FineTome-100k dataset
torch.cuda.empty_cache() # Clear cache after evaluation
eval_loss = 0.0
perplexity = 0.0

# --- Generation ---
print(f"Generating predictions for {len(prompts)} prompts (Batch Size: {eval_batch_size})...")
preds = []


for i in tqdm(range(0, len(prompts), eval_batch_size), desc="Generating predictions"):
    batch_prompts = prompts[i : i + eval_batch_size]

    # Tokenize prompts, ensuring padding is done correctly for the batch
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True, # Pad batch to longest sequence
        truncation=True,
        # tokenizer.model_max_length = 131072k context tokens for Llama-3.1-8B-Instruct
        max_length=1024 # To avoid OOM errors, truncate any prompt to at most 1024 tokens (so 1024 + 256 (generation) ≤ 131072)
    ).to(model.device)

    # Generate outputs
    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens_generation, # 256 tokens for generation
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, # Important for generation
            do_sample=False, # Use greedy decoding for consistent evaluation
            temperature=None,
            top_p=None,
            cache_implementation="offloaded", # OffloadedCache conserves GPU memory by keeping only the current layer’s KV cache on GPU and spilling all others to CPU
            # You might want to adjust generation parameters (temperature, top_k, etc.)
            # depending on how you want to evaluate (e.g., sample vs greedy)
        )

    # Decode predictions
    batch_preds_full = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # Strip the prompt part from the generated text
    for idx, (prompt, generated_text) in enumerate(zip(batch_prompts, batch_preds_full)):
         # Find the start of the generated response (after the prompt)
        if generated_text.startswith(prompt):
             pred = generated_text[len(prompt):].strip()
        else:
             # Fallback or warning if prompt isn't exactly at the start
             # This might happen due to tokenization nuances or if generation starts unexpectedly
             print(f"Warning: Generated text for sample {i+idx} does not start with prompt. Using heuristic slicing based on input length.")
             # Use the length of the input_ids for slicing the output tokens before decoding
             input_len = inputs['input_ids'][idx].shape[0]
             generated_ids = outputs[idx][input_len:]
             pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
             # As a last resort, if the above is empty or problematic, try a simple string find
             if not pred:
                 response_marker = "### Response:"
                 marker_pos = generated_text.find(response_marker)
                 if marker_pos != -1:
                     pred = generated_text[marker_pos + len(response_marker):].strip()
                 else:
                     pred = "[WARNING: Could not reliably separate prediction from prompt]"


        preds.append(pred)

    # ---- free GPU memory after each batch ----
    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    # Optional: Print progress
    if (i + eval_batch_size) % (eval_batch_size * 10) == 0:
         print(f"  Generated {i + eval_batch_size}/{len(prompts)}")

print("Generation complete.")
torch.cuda.empty_cache() # Clear cache after generation


# --- Metric Calculation ---
print("Calculating evaluation metrics (BLEU, ROUGE, BERTScore, Exact Match)...")

# Load metrics
try:
    bleu    = evaluate.load("bleu")
    rouge   = evaluate.load("rouge")
    berts   = evaluate.load("bertscore")
except Exception as e:
    print(f"Error loading metrics: {e}")
    print("Make sure you have run: pip install evaluate bleu rouge_score bert_score sentencepiece")
    exit()

# Ensure references for BLEU are in the correct format (list of lists)
bleu_references = [[r] for r in references]

# Compute metrics
# Initialize results in case of errors
bleu_res = {'bleu': 0.0}
rouge_res = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
bert_res = {'f1': [0.0], 'precision': [0.0], 'recall': [0.0]}

try:
    if preds and references: # Only compute if lists are not empty
        bleu_res  = bleu.compute(predictions=preds, references=bleu_references)
        rouge_res = rouge.compute(predictions=preds, references=references)
        # Run BERTScore on GPU if possible and if model has a device attribute
        compute_device = model.device if hasattr(model, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        bert_res  = berts.compute(predictions=preds, references=references, lang="en", device=compute_device)
    else:
        print("Warning: Empty predictions or references list. Skipping metric computation.")
except Exception as e:
    print(f"Error computing metrics: {e}")
    # Often due to empty predictions or references
    print("Preds sample:", preds[:2])
    print("Refs sample:", references[:2])


# Compute Exact Match
exact_match = sum(p == r for p, r in zip(preds, references)) / len(preds) if preds else 0.0

# Calculate average BERTScore values (ensure they are tensors/lists of numbers)
# Adding .get with default value [0.0] and checking for emptiness before calculating mean
bert_f1_scores = bert_res.get('f1', [0.0])
bert_precision_scores = bert_res.get('precision', [0.0])
bert_recall_scores = bert_res.get('recall', [0.0])

# Ensure scores are not empty before converting to tensor and calculating mean
avg_bert_f1 = torch.tensor(bert_f1_scores).mean().item() if bert_f1_scores else 0.0
avg_bert_precision = torch.tensor(bert_precision_scores).mean().item() if bert_precision_scores else 0.0
avg_bert_recall = torch.tensor(bert_recall_scores).mean().item() if bert_recall_scores else 0.0


# --- Reporting ---
print("\n--- Preparing Evaluation Outputs ---")
model_id_safe = os.path.basename(adapter_path)

# 1. Prepare the Summary Report String
report_string = f"""
Model: {model_id_safe} (Evaluated Adapter: {adapter_path} on Base: {base_model_name})
Dataset: {dataset_name} (Test Split Seed 42)
Number of Eval Samples: {len(prompts)}

--- Metrics ---
Perplexity: {perplexity:.4f} (Lower is better)
Exact Match (EM): {exact_match:.4f} (Higher is better)
BLEU: {bleu_res.get('bleu', 0.0):.4f} (Higher is better)
ROUGE-1: {rouge_res.get('rouge1', 0.0):.4f} (Higher is better)
ROUGE-2: {rouge_res.get('rouge2', 0.0):.4f} (Higher is better)
ROUGE-L: {rouge_res.get('rougeL', 0.0):.4f} (Higher is better)
BERTScore F1 (Avg): {avg_bert_f1:.4f} (Higher is better)
BERTScore Precision (Avg): {avg_bert_precision:.4f}
BERTScore Recall (Avg): {avg_bert_recall:.4f}
"""

# 2. Prepare the Detailed Predictions DataFrame
results_df = pd.DataFrame({
    'Prompt': prompts,
    'Reference': references,
    'Prediction': preds
})

# 3. Define Output Filenames
report_filename = f"{model_id_safe}_evaluation_report.txt"
predictions_filename = f"{model_id_safe}_predictions_vs_references.csv"

if DEBUG:
    report_filename = f"{model_id_safe}_evaluation_report_DEBUG.txt"
    predictions_filename = f"{model_id_safe}_predictions_vs_references_DEBUG.csv"

# 4. Save the Summary Report (.txt)
try:
    with open(report_filename, "w", encoding='utf-8') as f: # Added encoding
        f.write(report_string)
    print(f"Evaluation summary report saved to: {report_filename}")
except Exception as e:
    print(f"Error saving summary report to file: {e}")

# 5. Save the Detailed Predictions (.csv)
try:
    results_df.to_csv(predictions_filename, index=False, encoding='utf-8') # Added encoding
    print(f"Detailed predictions vs references saved to: {predictions_filename}")
except Exception as e:
    print(f"Error saving detailed predictions to CSV: {e}")


print("\nEvaluation script finished.")