import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline, logging
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig


model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "Llama FineTuned Testing"


# QLoRA params
rank = 64
alpha = 16
dropout = 0.1
use_4bit = True
quantization_type = "nf4"
nested_quantisation = False
output_dir = "./results"
epochs = 1
fp16 = False
bf16 = False


batch_per_gpu = 4
per_device_eval_batch = 4

gradient_accumulation_steps = 1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (Adamw optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 0
logging_steps = 25


# SFT (Self fine tuning)  params
# Maximum sequence length to use
max_seg_length = None
# Pack multiple short examples in the same in
packing = False
# Load the entire model on the GPU 0
device_map = {"": 0}


compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=quantization_type,
    bnb_4bit_compute_dtype=compute_dtype,
    nested_quantisation=nested_quantisation,
)

dataset = load_dataset(dataset_name, split="train")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)

model.config.use_cache = False
model.config.pretrained_tp = 1


# Load Llama tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=alpha,
    lora_dropout=dropout,
    r=rank,
    bias="none",
    task_type="CAUSAL_LM"
)

# Set training parameters
training_arguments = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_per_gpu,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    max_length=512 if max_seg_length is None else max_seg_length,
    packing=packing,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
)
trainer.train()


# Save trained model
trainer.save_model(new_model)
