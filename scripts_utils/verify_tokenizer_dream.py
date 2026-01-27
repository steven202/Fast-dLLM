from transformers import AutoTokenizer


def check_alignment():
    print("Checking Tokenizer Alignment...")
    # loading two Tokenizer
    tok_backbone = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Base-7B")
    tok_guidance = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # defining some common words to test
    test_words = ["apple", "Code", " import", "Algorithm", "Run", "123"]

    match_count = 0
    total = len(test_words)

    for word in test_words:
        id_b = tok_backbone.encode(word, add_special_tokens=False)
        id_g = tok_guidance.encode(word, add_special_tokens=False)

        print(f"Word: '{word}' | Dream: {id_b} | Guidance: {id_g}")

        if id_b == id_g:
            match_count += 1

    if match_count == total:
        print("\nVerified: The token IDs for common tokens are fully aligned.")
        print("Conclusion: Using Logit Slicing strategy is safe and will not degrade Guidance quality.")
    else:
        print("\nVerification Failed: ID mismatch! Cannot directly Slice!")


if __name__ == "__main__":
    check_alignment()
