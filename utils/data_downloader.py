"""
Utility: Data Downloader for DELA
Downloads and prepares prompts for GFlowNet training.
"""

import os
import json
from datasets import load_dataset
from typing import Optional

def download_prompts(
    dataset_name: str = "tatsu-lab/alpaca", 
    save_path: str = "data/train_prompts.jsonl",
    split: str = "train",
    column_name: str = "instruction"
):
    """
    Downloads a dataset from Hugging Face and saves specific prompt column locally.
    """
    if os.path.exists(save_path):
        print(f"[Data] Local dataset already exists at {save_path}. Skipping download.")
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"[Data] Downloading {dataset_name} from Hugging Face...")
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        print(f"[Data] Saving to {save_path}...")
        with open(save_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                if column_name in item:
                    # Save as JSONL: one JSON object per line
                    entry = {"instruction": item[column_name]}
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"[Data] Successfully saved {len(dataset)} prompts.")
        
    except Exception as e:
        print(f"[Data] Error during download: {e}")

if __name__ == "__main__":
    # Allows running as a standalone script
    download_prompts()