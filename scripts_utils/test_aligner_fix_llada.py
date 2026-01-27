
import torch
import sys
import os

from utils.aligner import StaticTokenAligner

# Add current dir to path
sys.path.append(os.getcwd())

from transformers import AutoTokenizer

def test_aligner():
    print("Loading Tokenizers...")
    try:
        # TGT = Backbone
        tokenizer_tgt = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        # SRC = Guidance
        tokenizer_src = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)
        # tokenizer_src = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
        # tokenizer_src = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        # Using TinyLlama as it is smaller/faster and was used in my previous command, although Llama-3.2 is preferred if available.
        # Let's use the one from the fail log if possible? No, verify_tokenizer used Llama-3.2. 
        # verify_tokenizer.py uses:
        # tok_backbone = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
        # tok_guidance = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on {device}...")

    print("Initializing StaticTokenAligner...")
    aligner = StaticTokenAligner(tokenizer_src, tokenizer_tgt, device=device)

    # Test Case 1: Word "apple"
    # verify_tokenizer says: Word: 'apple' | LLaDA: [32164] | Guidance: [23182]
    # NOTE: TinyLlama might have different IDs than Llama-3.2. 
    # Let's rely on string consistency.
    for word in ["apple", "banana", "orange", "grape", "watermelon"]:
        print(f"\nTesting word: '{word}'")
    
        # 1. Test Input Translation (Backbone/Tgt -> Guidance/Src)
        tgt_id = tokenizer_tgt.encode(word, add_special_tokens=False)[0]
        print(f"Target (Backbone) ID for '{word}': {tgt_id}")
        
        tgt_tensor = torch.tensor([[tgt_id]], device=device)
        translated_src_ids = aligner.translate_input(tgt_tensor)
        src_id_pred = translated_src_ids[0, 0].item()
        print(f"Translated Source (Guidance) ID: {src_id_pred}")
        
        decoded_src = tokenizer_src.decode([src_id_pred])
        print(f"Decoded translated ID: '{decoded_src}'")
        
        if word in decoded_src: # 'apple' vs ' apple' etc
            print("[PASS] Input Translation correct.")
        else:
            print(f"[FAIL] Input Translation mismatch. Expected something like '{word}', got '{decoded_src}'")

        # 2. Test Logic Alignment (Guidance/Src -> Backbone/Tgt)
        # Simulate logits from Guidance model focusing on 'apple'
        vocab_src = tokenizer_src.vocab_size
        vocab_tgt = tokenizer_tgt.vocab_size
        
        sim_logits = torch.full((1, vocab_src), -100.0, device=device)
        real_src_id_arr = tokenizer_src.encode(word, add_special_tokens=False)
        if len(real_src_id_arr) > 0:
            real_src_id = real_src_id_arr[0]
            sim_logits[0, real_src_id] = 100.0 # High prob for apple
            
            print(f"Simulating high probability for Source ID {real_src_id} ('{word}')")
            
            aligned_logits = aligner.align(sim_logits, vocab_tgt, topk=10)
            
            # Check if the highest value in aligned_logits corresponds to target ID of apple
            top_val, top_idx = torch.topk(aligned_logits, k=1)
            pred_tgt_id = top_idx.item()
            
            decoded_aligned = tokenizer_tgt.decode([pred_tgt_id])
            print(f"Aligned Target ID: {pred_tgt_id} -> '{decoded_aligned}'")
            
            if word in decoded_aligned:
                print("[PASS] Logit Alignment correct.")
            else:
                print("[FAIL] Logit Alignment mismatch.")
                
        # 3. Test Boundary Safety (The original crash cause)
        print("Testing Boundary Safety...")
        try:
            # Create logits that might pick valid Src IDs that map to -1 or something
            # Also ensure we don't crash if src_logits has shape issues or whatever
            bad_logits = torch.randn((1, vocab_src), device=device)
            _ = aligner.align(bad_logits, vocab_tgt)
            print("[PASS] Random logits handled safely.")
            
            # Test Input Translation with out of bounds
            huge_id = torch.tensor([[vocab_tgt + 100]], device=device)
            safe_trans = aligner.translate_input(huge_id)
            if safe_trans.item() == 0:
                print("[PASS] Out of bounds input handled safely.")
            else:
                print(f"[FAIL] Out of bounds input returned {safe_trans.item()}")
                
        except Exception as e:
            print(f"[FAIL] Safety Check Crashed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_aligner()
