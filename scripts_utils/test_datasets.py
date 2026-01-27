from datasets import load_dataset


def print_samples(title, rows, prompt_key, answer_key, limit=5):
    print(f"\n=== {title} ===")
    for i, item in enumerate(rows):
        if i >= limit:
            break
        prompt = item.get(prompt_key, "")
        answer = item.get(answer_key, "")
        print(f"\n[{i+1}] Q:\n{prompt}\nA:\n{answer}")


def main():
    # GSM8K
    gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    print_samples("gsm8k (train)", gsm8k, "question", "answer")

    # Hendrycks Math (all configs)
    math_configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    for cfg in math_configs:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
            print_samples(f"hendrycks_math/{cfg} (train)", ds, "problem", "solution")
        except Exception as e:
            print(f"Error loading {cfg}: {e}")

    # HumanEval
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    print_samples("humaneval (test)", humaneval, "prompt", "canonical_solution")

    # MBPP
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    print_samples("mbpp (train)", mbpp, "text", "code")


if __name__ == "__main__":
    main()
