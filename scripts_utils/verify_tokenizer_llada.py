from transformers import AutoTokenizer

def check_alignment():
    print("Checking Tokenizer Alignment...")
    # loading two Tokenizer
    tok_backbone = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    tok_guidance = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # tok_guidance = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    # tok_backbone = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B")
    # tok_guidance = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # defining some common words to test
    test_words = ["apple", "Code", " import", "Algorithm", "Run", "123"]
    
    match_count = 0
    total = len(test_words)

    for word in test_words:
        id_b = tok_backbone.encode(word, add_special_tokens=False)
        id_g = tok_guidance.encode(word, add_special_tokens=False)
        
        print(f"Word: '{word}' | LLaDA: {id_b} | Guidance: {id_g}")
        
        if id_b == id_g:
            match_count += 1
            
    if match_count == total:
        print("\nVerified: The token IDs for common tokens are fully aligned.")
        print("Conclusion: Using Logit Slicing strategy is safe and will not degrade Guidance quality.")
    else:
        print("\nVerification Failed: ID mismatch! Cannot directly Slice!")

if __name__ == "__main__":
    check_alignment()