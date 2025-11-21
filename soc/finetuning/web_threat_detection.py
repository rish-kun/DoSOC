import os
import glob
import json

import pandas as pd

# Path where all the CSVs from the Kaggle dataset live
DATA_DIR = "data/cybersecurity-threat-detection-logs"

# Output JSONL file for Qwen2.5 fine-tuning
OUTPUT_JSONL = "qwen_cyber_train.jsonl"

# Adjust these based on the actual CSV schema
LABEL_COLUMN = "label"        # e.g. "label", "is_malicious", "Threat_Detected"
FEATURE_COLUMNS = None        # None -> use all columns except LABEL_COLUMN

SYSTEM_PROMPT = (
    "You are a senior SOC analyst. "
    "Given a single log line, explain what it indicates and "
    "classify it as Benign or Malicious."
)


def row_to_example(row):
    # Decide which feature columns to use
    if FEATURE_COLUMNS is None:
        feature_cols = [c for c in row.index if c != LABEL_COLUMN]
    else:
        feature_cols = FEATURE_COLUMNS

    # Build a compact representation of the log line
    log_dict = {col: row[col] for col in feature_cols}

    # Map numeric / categorical label to text; adjust as needed
    raw_label = row[LABEL_COLUMN]

    try:
        # Example mapping: 1 -> Malicious, 0 -> Benign
        label_int = int(raw_label)
        label_text = "Malicious" if label_int == 1 else "Benign"
    except Exception:
        # If label is already text, just use it
        label_text = str(raw_label)

    user_content = (
        "Analyze the following log entry and decide if it is benign or malicious.\n\n"
        f"Log entry:\n{json.dumps(log_dict, ensure_ascii=False)}\n\n"
        "Respond with a short explanation and end with a final label in the form "
        "`label: Benign` or `label: Malicious`."
    )

    assistant_content = f"label: {label_text}"

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ]
    }


def iter_csv_files(data_dir):
    pattern = os.path.join(data_dir, "*.csv")
    for path in glob.glob(pattern):
        yield path


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSONL) or ".", exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for csv_path in iter_csv_files(DATA_DIR):
            print(f"Processing {csv_path}...")
            # Use chunks so you can handle millions of rows
            for chunk in pd.read_csv(csv_path, chunksize=50_000):
                for _, row in chunk.iterrows():
                    # Skip rows without a label
                    if pd.isna(row.get(LABEL_COLUMN, None)):
                        continue

                    example = row_to_example(row)
                    out_f.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
