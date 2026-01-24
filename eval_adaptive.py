
import argparse
import os
import sys
import torch
import logging
from tqdm import tqdm
from typing import Optional, Union, List

# Set environment variables for evaluation safety and remote code
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
os.environ.setdefault("LMEVAL_LOG_LEVEL", "INFO")

print("[eval_adaptive] Initializing lm-eval. Task indexing and dataset cache setup can take a few minutes on first run.")

# Add current directory to sys.path to import from train_adaptive_policy
sys.path.append(os.getcwd())

try:
    from train_adaptive_policy import rollout_llada, rollout_dream
    # Import base classes to inherit from if possible, or just use LM
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.__main__ import cli_evaluate
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    # Helper to load LLaDA/Dream models using their specific classes
    from llada.model.modeling_llada import LLaDAModelLM
    from dream.model.modeling_dream import DreamModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

eval_logger = logging.getLogger(__name__)

def load_policy_weights(model, policy_path, model_type):
    eval_logger.info(f"Loading policy from {policy_path}...")
    state_dict = torch.load(policy_path, map_location=model.device)
    
    if model_type == "llada":
        if hasattr(model, "model") and hasattr(model.model, "block_policy"):
            policy_net = model.model.block_policy
        elif hasattr(model, "block_policy"):
             policy_net = model.block_policy
        else:
             # Fallback attempt to find where policy is attached
             policy_net = model.model.block_policy
    else: # dream
        if hasattr(model, "block_policy"):
            policy_net = model.block_policy
        else:
            raise ValueError("Dream model does not have block_policy submodule")
        
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    return policy_net

class AdaptiveBase(LM):
    def __init__(
        self,
        policy_path: str,
        pretrained: str,
        model_type: str = "llada",
        guidance_model_name: Optional[str] = None,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        # Generation/Policy args
        gen_length: int = 256,
        steps: int = 256,
        block_len_min: int = 8,
        block_len_max: int = 64,
        threshold: float = 0.9,
        guidance_gamma: float = 0.5,
        guidance_temperature: float = 0.5,
        # CLI compatibility args (may not be used but passed by scripts)
        use_cache: bool = False,
        dual_cache: bool = False,
        factor: float = 1.0,
        show_speed: bool = False,
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        self.device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.block_len_min = int(block_len_min)
        self.block_len_max = int(block_len_max)
        self.threshold = float(threshold)
        self.guidance_gamma = float(guidance_gamma)
        self.guidance_temperature = float(guidance_temperature)
        
        # Load Model
        eval_logger.info(f"Loading {model_type} model: {pretrained}")
        if model_type == "llada":
            config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
            config.flash_attention = True
            self.model = LLaDAModelLM.from_pretrained(
                pretrained, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                config=config
            ).to(self.device)
        else:
            self.model = DreamModel.from_pretrained(
                pretrained,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            ).to(self.device)
            
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        
        # Load Policy
        self.policy_net = load_policy_weights(self.model, policy_path, model_type)
        
        # Load Guidance Model if needed
        self.ar_guidance_model = None
        if guidance_model_name:
            eval_logger.info(f"Loading guidance model: {guidance_model_name}")
            self.ar_guidance_model = AutoModelForCausalLM.from_pretrained(
                guidance_model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(self.device)
            self.ar_guidance_model.eval()

        if use_cache or dual_cache:
            eval_logger.warning("Prefix/Dual Cache requested but Adaptive Policy rollout does not strictly implement KV-caching yet. Running standard adaptive rollout.")

    def generate_until(self, requests):
        if not requests:
            return []
            
        results = []
        # Process one by one since rollout is single-instance
        for req in tqdm(requests, desc=f"Evaluating {self.model_type.upper()}"):
            prompt = req.args[0]
            gen_kwargs = req.args[1]
            until = gen_kwargs.get("until", [])
            max_new_tokens = gen_kwargs.get("max_new_tokens", self.gen_length)
            
            # Override instance defaults if provided in gen_kwargs
            current_steps = self.steps if "steps" not in gen_kwargs else gen_kwargs["steps"]
            
            if isinstance(until, str): until = [until]
            
            with torch.no_grad():
                if self.model_type == "llada":
                    res = rollout_llada(
                        model=self.model,
                        policy_net=self.policy_net,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        ar_guidance_model=self.ar_guidance_model,
                        guidance_gamma=self.guidance_gamma,
                        guidance_temperature=self.guidance_temperature,
                        gen_length=max_new_tokens,
                        steps=current_steps, 
                        block_len_min=self.block_len_min,
                        block_len_max=self.block_len_max,
                        threshold=self.threshold
                    )
                else:
                    res = rollout_dream(
                        model=self.model,
                        policy_net=self.policy_net,
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        ar_guidance_model=self.ar_guidance_model,
                        guidance_gamma=self.guidance_gamma,
                        guidance_temperature=self.guidance_temperature,
                        gen_length=max_new_tokens,
                        steps=current_steps, 
                        block_len_min=self.block_len_min,
                        block_len_max=self.block_len_max,
                        threshold=self.threshold
                    )

            # Decode just the new tokens
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            
            if res.generated_ids.shape[1] > prompt_len:
                new_ids = res.generated_ids[0, prompt_len:]
                gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            else:
                gen_text = ""

            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text.split(stop)[0]
            
            results.append(gen_text)
            
        return results

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not implemented for Adaptive Policy")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Loglikelihood rolling not implemented for Adaptive Policy")


@register_model("adaptive_llada")
class AdaptiveLLaDA(AdaptiveBase):
    def __init__(self, **kwargs):
        super().__init__(model_type="llada", **kwargs)

@register_model("adaptive_dream")
class AdaptiveDream(AdaptiveBase):
    def __init__(self, **kwargs):
        super().__init__(model_type="dream", **kwargs)

if __name__ == "__main__":
    cli_evaluate()
