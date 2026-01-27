"""
Train and Test DELA on GSM8K

This script demonstrates how to train the DELA policy on the GSM8K dataset and test its performance.
"""

import sys
import os
import json
import torch
import datetime
from datasets import load_dataset
# Add parent directory to sys.path to access the dela package
# Using __file__ allows running from any directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from core.dela import DELA
from configs.config_loader import Config
from utils.data_downloader import download_prompts

# --- Logging Setup ---
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Create log directory
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# Generate log filename with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"{timestamp}.log")

# Open log file and redirect stdout/stderr
log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, log_file)
# sys.stderr = Tee(sys.stderr, log_file) # Optional: redirect stderr as well

print(f"Logging to: {os.path.abspath(log_path)}")
# Configuration
data_path = "./data/gsm8k_train_prompts.jsonl"
print(f"Data path: {os.path.abspath(data_path)}")
checkpoint_path = "./checkpoints/dela_gsm8k_policy.pt"
print(f"Checkpoint will be saved to: {os.path.abspath(checkpoint_path)}")
# Download GSM8K data if not present
if not os.path.exists(data_path):
    print("Downloading GSM8K dataset...")
    try:
        # GSM8K requires 'main' config
        dataset = load_dataset("gsm8k", "main", split="train")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                entry = {"instruction": item["question"]}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
else:
    print(f"Dataset found at {data_path}")

# Load Training Data
training_contexts = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'instruction' in data:
            training_contexts.append(data['instruction'])

print(f"Loaded {len(training_contexts)} training examples")

# Use a smaller subset for demonstration purposes if needed
# training_contexts = training_contexts[:50] 
print(f"First example: {training_contexts[0]}")

# Initialize DELA Agent
# Use the HuggingFace ID since local model might not be present
config = Config(model_path="GSAI-ML/LLaDA-8B-Instruct")
# Adjust configuration if needed
# config.num_epochs = 1 
config.batch_size = 4 # Example adjustment

print("Initializing DELA...")
dela = DELA(config)
print("DELA initialized.")

# Train Policy
print("Starting training...")
dela.train(
    contexts=training_contexts,
    num_epochs=1, # 1 Epoch for now
    save_path=checkpoint_path
)
print(f"Training finished. Model saved to {os.path.abspath(checkpoint_path)}")

# Initialize DELA Agent for Testing
# We initialize a fresh agent to make sure we are loading the saved weights correctly
config = Config(model_path="GSAI-ML/LLaDA-8B-Instruct")
config.batch_size = 4 

print("Initializing DELA for Testing...")
dela = DELA(config)

# Load the trained policy
if os.path.exists(checkpoint_path):
    print(f"Loading policy from {os.path.abspath(checkpoint_path)}...")
    dela.load_policy(checkpoint_path)
else:
    print(f"Warning: Checkpoint not found at {os.path.abspath(checkpoint_path)}. Using random initialization.")
print("DELA ready for testing.")

# Try test
test_questions = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
]

print("\n--- Testing Generation ---")
for q in test_questions:
    print(f"\nQuestion: {q}")
    # Generate response
    response = dela.generate(
        condition=q,
        temperature=0.7,
        greedy=False
    )
    print(f"Answer: {response}")

# Test Accuracy on GSM8K Test Set
import re
from tqdm import tqdm

# Configuration for Test Data
test_data_path = "./data/gsm8k_test_prompts.jsonl"

# Download GSM8K Test data if not present
if not os.path.exists(test_data_path):
    print("Downloading GSM8K test dataset...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test")
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        with open(test_data_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # We need both question and answer for evaluation
                entry = {"instruction": item["question"], "answer": item["answer"]}
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("Test data download complete.")
    except Exception as e:
        print(f"Test data download failed: {e}")
else:
    print(f"Test dataset found at {test_data_path}")

# Load Test Data from file
test_dataset = []
with open(test_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'instruction' in data and 'answer' in data:
            test_dataset.append(data)

print(f"Loaded {len(test_dataset)} test examples.")

def extract_answer(text):
    """Extract the numerical answer from the text."""
    # If the text contains the gold separator, use it
    if "####" in text:
        return text.split("####")[-1].strip().replace(',', '')
    # Otherwise look for the last number
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None

def is_correct(pred, gold):
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except:
        return False

correct = 0
total = 0

print("Starting evaluation on GSM8K Test Set...")
# You can remove the slicing [:20] to test on the whole dataset (1319 examples)
# It might take a significant amount of time.
for i, item in enumerate(tqdm(test_dataset)): # Iterate through the whole dataset
    question = item["instruction"]
    gold_answer_raw = item["answer"]
    
    # Generate response
    response = dela.generate(
        condition=question,
        temperature=0.0, # Use greedy decoding for evaluation
        greedy=True
    )
    
    # Extraction and Comparison
    pred_val = extract_answer(response)
    gold_val = extract_answer(gold_answer_raw)
    
    if pred_val and gold_val and is_correct(pred_val, gold_val):
        correct += 1
        
    total += 1
    
    # Optional: Print progress every 10 examples
    if (i + 1) % 10 == 0:
        print(f"Step {i+1}: Acc = {correct/total:.4%} ({correct}/{total})")

print(f"Final Accuracy: {correct/total:.4%} ({correct}/{total})")


