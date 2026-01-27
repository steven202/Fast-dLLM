
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

def get_causal_attention_bias(seq_len, device):
    return torch.zeros((1, 1, seq_len, seq_len), device=device)

def test_generation():
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"Tokenizer mask token: {tokenizer.mask_token}")
    print(f"Tokenizer mask token id: {tokenizer.mask_token_id}")
    
    # Check special tokens map
    print(f"Special tokens map: {tokenizer.special_tokens_map}")
    
    # Determine correct mask ID
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        # Fallback to config check
        from llada.model.configuration_llada import LLaDAConfig
        config = LLaDAConfig.from_pretrained(model_path)
        print(f"Config mask token id: {config.mask_token_id}")
        mask_id = config.mask_token_id
        
    if mask_id is None:
        print("Mask ID still None. Using hardcoded 126336 for test.")
        mask_id = 126336

    print(f"Using Mask ID: {mask_id}")

    # Load Model
    from llada.model.modeling_llada import LLaDAModelLM
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()

    # Test generation consistency
    prompt = "Natalia sold clips to 48 of her friends in April,"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    gen_len = 10
    x = torch.full((1, input_ids.shape[1] + gen_len), mask_id, dtype=torch.long, device="cuda")
    x[:, :input_ids.shape[1]] = input_ids
    
    print("Testing Causal Attention (Default)...")
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        # Decode the generated part
        generated_ids = preds[:, input_ids.shape[1]:]
        print(f"Generated (Causal): {tokenizer.decode(generated_ids[0])}")

    print("Testing Bidirectional Attention (Zero Bias)...")
    seq_len = x.shape[1]
    # Create all-zeros attention bias (no masking)
    # Shape: (batch_size, 1, seq_len, seq_len)
    # 0.0 means visible. -inf means masked.
    attention_bias = torch.zeros((1, 1, seq_len, seq_len), device="cuda", dtype=torch.float)
    
    with torch.no_grad():
        # Pass attention_bias to override causal default
        outputs = model(x, attention_bias=attention_bias)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        generated_ids = preds[:, input_ids.shape[1]:]
        print(f"Generated (Bidirectional): {tokenizer.decode(generated_ids[0])}")

if __name__ == "__main__":
    test_generation()
