import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen3-30B-Instruct"     # same base as during training
# where SFTTrainer saved the adapter
ADAPTER_DIR = "qwen3-30b-toucan-lora"
MERGED_DIR = "qwen3-30b-toucan-merged"          # final merged model directory

# 1. Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Attach LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# 3. Merge LoRA weights into the base model and unload PEFT wrappers
merged_model = peft_model.merge_and_unload()

# 4. Load / save tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(MERGED_DIR)

# 5. Save final merged model
merged_model.save_pretrained(MERGED_DIR)

print("Merged model saved to:", MERGED_DIR)
