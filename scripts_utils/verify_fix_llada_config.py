
import sys
import os
import torch
# Allow importing local modules
sys.path.append(os.getcwd())

from llada.model.modeling_llada import LLaDAModelLM
from transformers import AutoConfig

model_path='GSAI-ML/LLaDA-8B-Instruct'
print(f"Loading config from {model_path}")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print("Attempting to instantiate LLaDAModelLM...")
try:
    # We don't load weights to save time/memory, just check config init logic
    # But from_pretrained usually loads weights.
    # We can try to manually invoke the init logic that failed.
    
    from llada.model.modeling_llada import create_model_config_from_pretrained_config
    
    model_config = create_model_config_from_pretrained_config(config)
    print("create_model_config_from_pretrained_config successful!")
    print(f"ModelConfig: {model_config}")
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
