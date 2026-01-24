from datasets import load_dataset
import random
from typing import List, Optional

def load_prompts(dataset_names: List[str], max_samples: Optional[int] = None) -> List[dict]:
    """
    Loads and normalizes multiple datasets.
    Returns a list of dicts: {"prompt": str, "answer": str, "dataset": str}
    """
    all_prompts = []
    
    for d_name in dataset_names:
        dataset_data = []
        try:
            if d_name == "gsm8k":
                ds = load_dataset("gsm8k", "main", split="train")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["question"], 
                        "answer": item["answer"], 
                        "dataset": "gsm8k"
                    })
            
            elif d_name == "math":
                ds = load_dataset("hendrycks/competition_math", split="train")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["problem"], 
                        "answer": item["solution"], 
                        "dataset": "math"
                    })
            
            elif d_name == "humaneval":
                ds = load_dataset("openai_humaneval", split="test")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["prompt"], 
                        "answer": item["canonical_solution"], 
                        "dataset": "humaneval"
                    })

            elif d_name == "mbpp":
                # MBPP often needs 'sanitized' split for cleaner code
                ds = load_dataset("mbpp", "sanitized", split="train")
                for item in ds:
                    # Construct a prompt that looks like code comments
                    prompt_text = f'"""\n{item["text"]}\n"""\n'
                    dataset_data.append({
                        "prompt": prompt_text, 
                        "answer": item["code"], 
                        "dataset": "mbpp"
                    })
            
            print(f"Loaded {len(dataset_data)} samples from {d_name}.")
            all_prompts.extend(dataset_data)

        except Exception as e:
            print(f"Error loading {d_name}: {e}")

    # Shuffle mixed datasets to prevent catastrophic forgetting
    random.shuffle(all_prompts)

    if not all_prompts:
        all_prompts = [{"prompt": "Explain the theory of relativity.", "answer": "N/A", "dataset": "k"}] # Fallback

    if max_samples is not None:
        all_prompts = all_prompts[:max_samples]
        
    return all_prompts
