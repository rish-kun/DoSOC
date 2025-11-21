from datasets import load_dataset
import json

# Load the emotion dataset
dataset = load_dataset("dair-ai/emotion")

# Define emotion labels mapping
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


def format_to_chatml(example):
    """
    Convert emotion dataset examples to ChatML format for Qwen2.5
    ChatML format uses:
    <|im_start|>role
    content<|im_end|>
    """
    text = example['text']
    emotion = emotion_labels[example['label']]

    # Create conversation in ChatML format
    conversation = {
        "type": "chatml",
        "messages": [
            {
                "role": "system",
                "content": "You are an emotion classification assistant. Analyze the given text and identify the primary emotion expressed."
            },
            {
                "role": "user",
                "content": f"Classify the emotion in this text: {text}"
            },
            {
                "role": "assistant",
                "content": f"The emotion expressed in this text is: {emotion}"
            }
        ]
    }

    return conversation


def format_to_chatml_tokens(example):
    """
    Alternative format: Direct ChatML token format as string
    """
    text = example['text']
    emotion = emotion_labels[example['label']]

    chatml_string = f"""<|im_start|>system
You are an emotion classification assistant. Analyze the given text and identify the primary emotion expressed.<|im_end|>
<|im_start|>user
Classify the emotion in this text: {text}<|im_end|>
<|im_start|>assistant
The emotion expressed in this text is: {emotion}<|im_end|>"""

    return {"text": chatml_string}


# Format the entire dataset
formatted_train = [format_to_chatml(example) for example in dataset['train']]
formatted_val = [format_to_chatml(example)
                 for example in dataset['validation']]
formatted_test = [format_to_chatml(example) for example in dataset['test']]

# Save to JSONL files (recommended format)


def save_to_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


save_to_jsonl(formatted_train, 'emotion_train.jsonl')
save_to_jsonl(formatted_val, 'emotion_val.jsonl')
save_to_jsonl(formatted_test, 'emotion_test.jsonl')

print(f"Training samples: {len(formatted_train)}")
print(f"Validation samples: {len(formatted_val)}")
print(f"Test samples: {len(formatted_test)}")

# Display sample formatted output
print("\nSample formatted conversation:")
print(json.dumps(formatted_train[0], indent=2, ensure_ascii=False))
