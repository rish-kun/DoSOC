import json
import re
from datasets import load_dataset


def format_toucan_for_qwen(example):
    """
    Processes a single row from the Toucan dataset for Qwen fine-tuning.

    Steps:
    1. Parses the 'messages' string into a Python list.
    2. Cleans the System Prompt: Extracts the tool definitions from the custom 
       <|im_system|> tags and reformats them into a clean natural language instruction.
    3. Maps Roles: Converts 'tool'/'function' roles to 'user' (standard practice 
       for ChatML to represent environment feedback) and ensures content is stringified.
    """
    # 1. Parse the stringified JSON from the dataset
    try:
        messages = json.loads(example['messages'])
    except (json.JSONDecodeError, TypeError):
        # Return empty if parsing fails to avoid breaking the pipeline
        return {"messages": []}

    cleaned_messages = []

    for msg in messages:
        role = msg.get("role", "")
        # Ensure content is a string (some datasets have dicts/lists in content)
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)

        # 2. Clean System Content
        if role == "system":
            # The dataset wraps tools in specific tokens:
            # <|im_system|>tool_declare<|im_middle|>[TOOLS_JSON]<|im_end|>
            # We regex to find the content between middle and end tags.
            tool_pattern = r"tool_declare<\|im_middle\|>(.*?)<\|im_end\|>"
            match = re.search(tool_pattern, content, re.DOTALL)

            if match:
                # Extract the raw JSON tool definition
                tool_json = match.group(1).strip()
                # Reformat into a clean system instruction for Qwen
                content = f"You are an intelligent agent with access to the following tools. Use them when necessary:\n\n{tool_json}"
            else:
                # Fallback: just strip the tags if the pattern doesn't match perfectly
                content = content.replace("<|im_system|>", "").replace(
                    "<|im_middle|>", "").replace("<|im_end|>", "")

        # 3. Map Roles for ChatML Compatibility
        # Qwen models typically treat tool outputs as User messages (inputs from the environment).
        if role in ["tool", "function"]:
            role = "user"
            content = f"Tool Output:\n{content}"

        cleaned_messages.append({
            "role": role,
            "content": content
        })

    return {"messages": cleaned_messages}


def main():
    # Configuration
    DATASET_NAME = "Agent-Ark/Toucan-1.5M"
    OUTPUT_FILE = "toucan_qwen_formatted.jsonl"

    print(f"Loading {DATASET_NAME}...")
    # Use streaming=True if you don't want to download the whole dataset at once
    dataset = load_dataset(DATASET_NAME, 'Kimi-K2',
                           split="train")

    print("Formatting dataset (this may take a while)...")
    # Apply the formatting function
    formatted_dataset = dataset.map(
        format_toucan_for_qwen,
        num_proc=8,  # Adjust based on your CPU cores
        remove_columns=dataset.column_names,  # Remove old raw columns to save space
        desc="Formatting messages"
    )

    # Filter out empty rows (failed parses)
    formatted_dataset = formatted_dataset.filter(
        lambda x: len(x["messages"]) > 0)

    # Verification: Print the first formatted example
    print("\n--- Sample Formatted Entry ---")
    sample_msgs = formatted_dataset[0]["messages"]
    for m in sample_msgs:
        print(f"[{m['role'].upper()}]: {m['content'][:100]}...")
    print("------------------------------")

    # Saving for Fine-Tuning (JSONL format is standard for Unsloth/Axolotl/TRL)
    print(f"Saving to {OUTPUT_FILE}...")
    formatted_dataset.to_json(OUTPUT_FILE)
    print("Done!")


if __name__ == "__main__":
    main()
