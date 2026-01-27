import os
import sys
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.data_utils import load_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "math", "humaneval", "mbpp"])
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--local_only", action="store_true")
    parser.add_argument("--show_text", action="store_true")
    args = parser.parse_args()

    if args.local_only:
        # os.environ["FAST_DLLM_DATASETS_LOCAL_ONLY"] = "1"
        pass

    total = 0
    for dataset in args.datasets:
        try:
            prompts = load_prompts([dataset], max_samples=args.max_samples)
        except Exception as exc:
            print(f"Error loading {dataset}: {exc}")
            continue

        print(f"Loaded {len(prompts)} prompts from {dataset}")
        # Basic validation
        for i, item in enumerate(prompts[: min(len(prompts), args.max_samples)]):
            assert isinstance(item, dict), f"item {i} is not dict"
            assert isinstance(item.get("prompt"), str), f"item {i} prompt not str"
            assert isinstance(item.get("answer"), str), f"item {i} answer not str"
            assert isinstance(item.get("dataset"), str), f"item {i} dataset not str"
            assert item["prompt"] != "", f"item {i} prompt empty"
            print(f"[{dataset} #{i}] prompt_len={len(item['prompt'])} answer_len={len(item['answer'])}")
            if args.show_text:
                print("PROMPT:\n" + item["prompt"])
                print("ANSWER:\n" + item["answer"])
                print("-" * 40)

        total += len(prompts)

    print(f"OK (total_loaded={total})")


if __name__ == "__main__":
    main()
