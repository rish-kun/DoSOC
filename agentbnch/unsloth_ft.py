import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import os

# Set environment variables for max performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "24"
torch.set_num_threads(24)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_FILE = "toucan_qwen_formatted.jsonl"
OUTPUT_DIR = "qwen3-30b-toucan-lora"

# Load with Unsloth for speed
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)

# Unsloth PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0 for max speed
    bias="none",
    use_gradient_checkpointing=False,  # Disabled for speed
    random_state=3407,
)

# Load dataset
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

# Maximum speed configuration
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,

    # Batch size - increase until OOM
    per_device_train_batch_size=8,  # Start with 8, try 12, 16
    gradient_accumulation_steps=1,   # No accumulation for speed

    # Learning rate
    learning_rate=2e-4,
    warmup_steps=100,

    # Optimization settings
    bf16=True,
    tf32=True,  # Enable TF32 for speed on H200
    gradient_checkpointing=False,  # DISABLED for speed

    # Sequence length
    max_length=2048,

    # Dataloader optimization
    dataloader_num_workers=20,  # Use most CPUs
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    dataloader_persistent_workers=True,

    # Packing
    packing=True,  # Keep enabled, Unsloth handles it well

    # Logging
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    # Speed optimizations
    optim="adamw_torch_fused",  # Fused optimizer for H200
    dataset_num_proc=24,  # Use all CPUs for preprocessing
    report_to="none",
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_num_proc=24,
)

# Train with Unsloth's optimized training
trainer.train()

# Save
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
