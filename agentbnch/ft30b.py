import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# ---- Config ----
# change if you use a different Qwen3-30B variant
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TRAIN_FILE = "toucan_qwen_formatted.jsonl"
OUTPUT_DIR = "qwen3-30b-toucan-lora"

# ---- Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure padding token exists and is on the right (better for causal LM)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---- Dataset ----
# Each row should look like: {"messages": [{"role": "...", "content": "..."}, ...]}
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
data = dataset.select(range(100000))
# Optionally shuffle or subsample if you’re just testing
# dataset = dataset.shuffle(seed=42).select(range(10000))

# ---- LoRA / PEFT config ----
# Typical LoRA settings for Qwen3; adjust r / target_modules if you want to experiment
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
    num_train_epochs=1,                  # increase after everything works

    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,       # effective batch size = 64

    gradient_checkpointing=True,         # reduces memory usage
    learning_rate=2e-4,                  # highuse_cache=Falseer LR is typical for LoRA
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    bf16=True,                           # H100 supports bfloat16 very well
    max_length=2048,                     # increase later if you can afford it


    dataloader_num_workers=16,  # Use your 24 vCPUs to pre-process data fast
    dataloader_pin_memory=True,  # Faster transfer to VRAM
    dataloader_pin_memory=True,      # Faster transfer to GPU

    # pack multiple examples per sequence for throughput
    packing=True,
    report_to="none",
    dataset_num_proc=22
    # or "wandb" / "tensorboard"

    # Disable compile for now as it can cause the "startup lag" you saw earlier
    torch_compile=False
)

# ---- Load base model on H200 ----
# On a single H200, bf16 + LoRA with small batch size is feasible for 30B.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
    use_cache=False,
)

# ---- SFTTrainer with LoRA ----
# SFTTrainer understands conversational datasets with a "messages" field and
# will apply the Qwen chat template automatically.
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,             # this wraps the model with PEFT LoRA
    processing_class=tokenizer,
    # ---- ADD THIS LINE ----
    # No formatting_func needed because we already have `messages`
)

# ---- Train ----
trainer.train()

# ---- Save adapter + tokenizer ----
# For a PEFT model, `save_model` will save the LoRA adapter weights.
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. LoRA adapter + tokenizer saved to:", OUTPUT_DIR)
