
import argparse
import os
import sys
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm

# Set environment variables for evaluation safety and remote code
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

# Add current directory to sys.path to import from train_adaptive_policy
sys.path.append(os.getcwd())

try:
    from train_adaptive_policy import rollout_llada, rollout_dream
    from dream.eval import Dream
    from llada.eval_llada import LLaDAEvalHarness
    from lm_eval import evaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def load_policy_weights(model, policy_path, model_type):
    print(f"Loading policy from {policy_path}...")
    state_dict = torch.load(policy_path, map_location=model.device)
    
    if model_type == "llada":
        policy_net = model.model.block_policy
    else:
        policy_net = model.block_policy
        
    policy_net.load_state_dict(state_dict)
    policy_net.eval()
    return policy_net

class AdaptiveDream(Dream):
    def __init__(self, policy_path, guidance_model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = load_policy_weights(self.model, policy_path, "dream")
        
        self.ar_guidance_model = None
        if guidance_model_name:
            from transformers import AutoModelForCausalLM
            print(f"Loading guidance model: {guidance_model_name}")
            self.ar_guidance_model = AutoModelForCausalLM.from_pretrained(
                guidance_model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(self.model.device)
            self.ar_guidance_model.eval()

    def generate_until(self, requests):
        results = []
        for req in tqdm(requests, desc="Evaluating Dream"):
            prompt = req.args[0]
            gen_kwargs = req.args[1]
            max_gen_len = gen_kwargs.get("max_new_tokens", 256)
            until = gen_kwargs.get("until", [])
            if isinstance(until, str): until = [until]

            with torch.no_grad():
                res = rollout_dream(
                    model=self.model,
                    policy_net=self.policy_net,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    ar_guidance_model=self.ar_guidance_model,
                    guidance_gamma=0.5, # Default from train script
                    guidance_temperature=0.5,
                    gen_length=max_gen_len,
                    steps=max_gen_len, 
                    block_len_min=8,
                    block_len_max=64,
                    threshold=0.9
                )
            
            # Decode just the new tokens
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            
            new_ids = res.generated_ids[0, prompt_len:]
            gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text.split(stop)[0]
            
            results.append(gen_text)
            
        return results

class AdaptiveLLaDA(LLaDAEvalHarness):
    def __init__(self, policy_path, guidance_model_name=None, **kwargs):
        super().__init__(**kwargs)
        self.policy_net = load_policy_weights(self.model, policy_path, "llada")
        
        self.ar_guidance_model = None
        if guidance_model_name:
            from transformers import AutoModelForCausalLM
            print(f"Loading guidance model: {guidance_model_name}")
            self.ar_guidance_model = AutoModelForCausalLM.from_pretrained(
                guidance_model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(self.device)
            self.ar_guidance_model.eval()

    def _generate_batch(self, requests):
        # LLaDAEvalHarness structure is different. It doesn't have _generate_batch taking prompts list in the same way.
        # It has _forward_process in eval_llada.py.
        # But wait, LLaDAEvalHarness inherits from LM.
        # Check eval_llada.py generate_until.
        # It calls generate_with_prefix_cache or generate.
        # I should override generate_until.
        pass
    
    def generate_until(self, requests):
        # Simplified implementation of generate_until using rollout_llada
        results = []
        for req in tqdm(requests, desc="Evaluating LLaDA"):
            prompt = req.args[0]
            gen_kwargs = req.args[1]
            max_gen_len = gen_kwargs.get("max_new_tokens", 256)
            
            until = gen_kwargs.get("until", [])
            if isinstance(until, str): until = [until]

            with torch.no_grad():
                res = rollout_llada(
                    model=self.model,
                    policy_net=self.policy_net,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    ar_guidance_model=self.ar_guidance_model,
                    guidance_gamma=0.5,
                    guidance_temperature=0.5,
                    gen_length=max_gen_len,
                    steps=max_gen_len, # Assuming 1 step per token roughly for safety, or tuned
                    block_len_min=8,
                    block_len_max=64,
                    threshold=0.9
                )
            
            # Decode
            full_text = self.tokenizer.decode(res.generated_ids[0], skip_special_tokens=True)
            
            # Strip prompt to get generation
            # Note: prompt decoding might differ slightly due to tokenization
            # Ideally we mask input ids
            input_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
            # generated_ids include prompt
            new_ids = res.generated_ids[0][input_len:] # Approx
            # Better:
            # rollout_llada returns x which has input_ids prompt at start.
            # We can just decode the new part.
            prompt_len_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            new_ids = res.generated_ids[0, prompt_len_ids:]
            gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

            # Handle stop sequences (until)
            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text.split(stop)[0]
            
            results.append(gen_text)
            
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["llada", "dream"], required=True)
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--guidance_model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Define tasks and shots
    tasks_mapping = {
        "gsm8k": ("gsm8k", 5),
        "math": ("minerva_math", 4), # Use minerva_math as fallback for proper formatting
        "humaneval": ("humaneval", 0),
        "mbpp": ("mbpp", 3)
    }

    # Defaults for guidance if not provided
    if args.guidance_model is None:
        if args.model == "llada":
            args.guidance_model = "meta-llama/Llama-3.2-1B-Instruct"
        elif args.model == "dream":
            args.guidance_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            raise ValueError(f"Unknown model type: {args.model}")

    # Initialize model wrapper
    if args.model == "llada":
        lm = AdaptiveLLaDA(
            policy_path=args.policy_path,
            guidance_model_name=args.guidance_model,
            model_path='GSAI-ML/LLaDA-8B-Instruct',
            device=args.device,
            batch_size=1
        )
    else:
        lm = AdaptiveDream(
            policy_path=args.policy_path,
            guidance_model_name=args.guidance_model,
            pretrained='Dream-org/Dream-v0-Base-7B',
            device=args.device,
            batch_size=1
        )

    # Run evaluation
    results = {}
    for user_task_name, (lm_eval_task_name, num_fewshot) in tasks_mapping.items():
        print(f"\nEvaluating {user_task_name} (mapped to {lm_eval_task_name}) ({num_fewshot}-shot)...")
        try:
            # lm_eval 0.4.x style
            res = evaluator.simple_evaluate(
                model=lm,
                tasks=[lm_eval_task_name],
                num_fewshot=num_fewshot,
                batch_size=1
            )
            # Extract score
            if "results" in res:
                print(f"Result for {user_task_name}: {res['results']}")
                results[user_task_name] = res['results']
            else:
                 print(f"Result for {user_task_name}: {res}")
                 results[user_task_name] = str(res)
        except Exception as e:
            print(f"Failed to evaluate {user_task_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nFinal Summmary:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
