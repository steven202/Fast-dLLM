from datasets import load_dataset
import os
import random
from typing import List, Optional


_DEFAULT_CACHE_DIR = os.path.expanduser(
    os.environ.get("FAST_DLLM_DATASETS_DIR", "~/fast_dllm_datasets")
)
os.environ.setdefault("HF_DATASETS_CACHE", _DEFAULT_CACHE_DIR)
def _dataset_kwargs():
    download_mode = "reuse_dataset_if_exists"
    local_only = os.environ.get("FAST_DLLM_DATASETS_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
    if local_only:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return {
        "cache_dir": _DEFAULT_CACHE_DIR,
        "download_mode": download_mode,
    }


def _load_dataset_with_cache(path: str, name: Optional[str] = None, split: Optional[str] = None):
    kwargs = _dataset_kwargs()
    if name is not None:
        return load_dataset(path, name, split=split, **kwargs)
    return load_dataset(path, split=split, **kwargs)


def _format_fewshot_example(dataset: str, prompt: str, answer: str) -> str:
    if dataset == "gsm8k":
        return f"Question: {prompt}\nAnswer: {answer}\n"
    if dataset == "math":
        return f"Problem: {prompt}\nSolution: {answer}\n"
    if dataset == "humaneval":
        return f"### Prompt:\n{prompt}\n### Solution:\n{answer}\n"
    if dataset == "mbpp":
        return f"Task:\n{prompt}\nSolution:\n{answer}\n"
    return f"Prompt: {prompt}\nAnswer: {answer}\n"


def _build_fewshot_prefix(dataset_data: List[dict], dataset: str, k: int, exclude_index: Optional[int] = None) -> str:
    if k <= 0:
        return ""
    examples = []
    for i, item in enumerate(dataset_data):
        if exclude_index is not None and i == exclude_index:
            continue
        examples.append(_format_fewshot_example(dataset, item["prompt"], item["answer"]))
        if len(examples) >= k:
            break
    if not examples:
        return ""
    return "\n".join(examples).strip() + "\n\n"


def load_prompts(
    dataset_names: List[str],
    max_samples: Optional[int] = None,
    fewshot_map: Optional[dict] = None,
) -> List[dict]:
    """
    Loads and normalizes multiple datasets.
    Returns a list of dicts: {"prompt": str, "answer": str, "dataset": str}
    """
    all_prompts = []
    
    for d_name in dataset_names:
        dataset_data = []
        try:
            if d_name == "gsm8k":
                try:
                    ds = _load_dataset_with_cache("gsm8k", "main", split="train")
                except Exception:
                    ds = _load_dataset_with_cache("openai/gsm8k", "main", split="train")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["question"],
                        "answer": item["answer"],
                        "dataset": "gsm8k",
                    })
            
            elif d_name == "math":
                ds = _load_dataset_with_cache("qwedsacf/competition_math", split="train")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["problem"],
                        "answer": item["solution"],
                        "dataset": "math",
                    })
            
            elif d_name == "humaneval":
                ds = _load_dataset_with_cache("openai/openai_humaneval", split="test")
                for item in ds:
                    dataset_data.append({
                        "prompt": item["prompt"],
                        "answer": item["canonical_solution"],
                        "dataset": "humaneval",
                    })

            elif d_name == "mbpp":
                ds = _load_dataset_with_cache("google-research-datasets/mbpp", "full", split="train")
                for item in ds:
                    # Construct a prompt that looks like code comments
                    prompt_text = f'"""\n{item["text"]}\n"""\n'
                    dataset_data.append({
                        "prompt": prompt_text,
                        "answer": item["code"],
                        "dataset": "mbpp",
                    })
            
            print(f"Loaded {len(dataset_data)} samples from {d_name}.")

            if fewshot_map is not None and d_name in fewshot_map:
                k = int(fewshot_map.get(d_name, 0))
                if k > 0:
                    for idx, item in enumerate(dataset_data):
                        prefix = _build_fewshot_prefix(dataset_data, d_name, k, exclude_index=idx)
                        item = dict(item)
                        item["prompt"] = prefix + item["prompt"]
                        all_prompts.append(item)
                else:
                    all_prompts.extend(dataset_data)
            else:
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
