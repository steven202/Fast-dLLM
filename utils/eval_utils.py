import re

def extract_answer_math(text):
    """Specific extractor for MATH/GSM8K"""
    # 1. Handle GSM8K style (####)
    if "####" in text:
        return text.split("####")[-1].strip().replace(',', '')
    
    # 2. Handle MATH style (\boxed{})
    # Find the last occurrence of \boxed{...}
    boxed_matches = re.findall(r'\\boxed\{(.*?)\}', text)
    if boxed_matches:
        return boxed_matches[-1]
        
    # 3. Fallback: Find last number
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None

def normalize_code(code):
    """Strip whitespace, comments, and imports for comparison"""
    # Simple normalization: remove all whitespace to compare logic structure
    return re.sub(r'\s+', '', code).strip()

def is_correct_smart(pred, gold, dataset_type):
    """Polymorphic evaluator"""
    try:
        if dataset_type in ["gsm8k", "math"]:
            pred_val = extract_answer_math(pred)
            gold_val = extract_answer_math(gold)
            if pred_val is None or gold_val is None:
                return False
            # Check for float equality
            return abs(float(pred_val) - float(gold_val)) < 1e-6
            
        elif dataset_type in ["humaneval", "mbpp"]:
            # For code, Exact Match is very harsh. 
            # We usually rely on NLL reward for code training.
            # But for logging accuracy, we try normalized string match.
            return normalize_code(pred) == normalize_code(gold)
            
    except:
        # If conversion to float fails (e.g. comparing "5" to "x=5"), return False
        return False
    return False
