import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---- Configuration ----
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_FILE = "toucan_qwen_formatted.jsonl"
OUTPUT_DIR = "qwen3-30b-toucan-unsloth"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None = Auto-detects bfloat16 for H200
# Set to True if you want to save massive memory (tiny accuracy drop)
LOAD_IN_4BIT = False

# ---- 1. Load Model (The Unsloth Way) ----
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
    # Unsloth handles the "use_cache=False" internally during training
)

# ---- 2. Add LoRA Adapters ----
# Unsloth automatically targets all the right linear layers for Qwen
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # Unsloth suggests 0 for faster optimization
    bias="none",
    use_gradient_checkpointing="unsloth",  # Uses Unsloth's optimized checkpointing
    random_state=3407,
)

# ---- 3. Data Loading ----
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
# Optional: Slice to test speed
# dataset = dataset.select(range(1000))

# ---- 4. Training Config (H200 Optimized) ----
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,  # Unsloth saves RAM, so we can pump this up
    gradient_accumulation_steps=4,  # 16 * 4 = 64 Effective Batch
    warmup_steps=10,
    max_steps=0,  # Set to >0 if you want to limit steps, else uses epochs
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,

    # ---- OPTIMIZATIONS FROM PREVIOUS TURN ----
    dataloader_num_workers=16,      # Keep CPU busy
    dataloader_pin_memory=True,     # Fast GPU transfer
    dataloader_persistent_workers=True,

    optim="adamw_8bit",             # Saves more memory
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",

    # Unsloth handles packing implicitly but setting this ensures compatibility
    packing=True,
)

# ---- 5. Trainer ----
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    # Unsloth usually expects 'text' or 'messages' handled via formatter
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=24,       # Use all cores for startup processing
    args=training_args,
)

# ---- 6. Train & Save ----
trainer_stats = trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

# ---- 7. (Optional) Save to GGUF directly ----
# model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")
