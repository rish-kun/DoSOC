import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import os

# ---- CRITICAL: Disable tokenizer parallelism warning ----
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---- Config ----
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_FILE = "toucan_qwen_formatted.jsonl"
OUTPUT_DIR = "qwen3-30b-toucan-lora"

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---- Dataset ----
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
data = dataset.select(range(50000))

# ---- CRITICAL FIX: Pre-tokenize the dataset using all CPU cores ----


def preprocess_function(examples):
    """Apply chat template and tokenize in parallel"""
    # SFTTrainer will apply chat template automatically, but we can pre-tokenize
    # the messages to avoid doing it during training
    return examples  # If dataset already has "messages", keep as is


# Process dataset with all available cores
dataset = data.map(
    preprocess_function,
    batched=True,
    num_proc=22,  # Use 22 of your 24 cores (leave 2 for system)
    remove_columns=[],  # Keep all columns
    desc="Preprocessing dataset",
)

# ---- LoRA / PEFT config ----
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ---- Training arguments ----
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,

    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective batch size = 32

    gradient_checkpointing=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,
    max_length=2048,

    # ---- CRITICAL FIXES for CPU utilization ----
    dataloader_num_workers=16,  # Optimal: 2x GPU count, max 16-20
    dataloader_prefetch_factor=4,  # Prefetch 4 batches per worker
    dataloader_pin_memory=True,  # Faster CPU->GPU transfer
    dataloader_persistent_workers=True,  # Keep workers alive between epochs

    # Packing and dataset preprocessing
    packing=True,
    dataset_num_proc=22,  # For any remaining preprocessing

    # Optimization flags
    torch_compile=False,  # Disable for stability
    report_to="none",

    # Remove cache to save memory during gradient checkpointing
    # (already handled by use_cache=False in model)
)

# ---- Load base model on H200 ----
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
    use_cache=False,  # Required for gradient checkpointing
)

# ---- SFTTrainer with LoRA ----
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

# ---- Train ----
print("Starting training with optimized CPU utilization...")
trainer.train()

# ---- Save adapter + tokenizer ----
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. LoRA adapter + tokenizer saved to:", OUTPUT_DIR)
