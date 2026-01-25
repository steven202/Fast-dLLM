
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
    from utils.eval_utils import is_correct_smart
    from utils.logging_utils import Tee
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

eval_logger = logging.getLogger(__name__)

_LOG_FILE = None
_LOG_INITIALIZED = False
_WANDB_INITIALIZED = False


def _sanitize_name_segment(value: str) -> str:
    value = (value or "").strip()
    for ch in ["/", "\\", ":", " "]:
        value = value.replace(ch, "_")
    return value


def build_eval_run_name(
    model_type: str,
    pretrained: str,
    policy_path: str,
    guidance_model_name: Optional[str],
    gen_length: int,
    steps: int,
    block_len_max: int,
) -> str:
    policy_part = _sanitize_name_segment(os.path.splitext(os.path.basename(policy_path))[0])
    pretrained_part = _sanitize_name_segment(pretrained)
    guidance_part = _sanitize_name_segment(guidance_model_name or "none")
    return f"{model_type}_{policy_part}_{pretrained_part}_{guidance_part}_L{gen_length}_S{steps}_B{block_len_max}"


def setup_eval_logging(log_dir: str = "./eval_log", run_name: Optional[str] = None):
    global _LOG_INITIALIZED, _LOG_FILE
    if _LOG_INITIALIZED:
        return _LOG_FILE
    os.makedirs(log_dir, exist_ok=True)
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = _sanitize_name_segment(run_name or "run")
    log_path = os.path.join(log_dir, f"eval_{safe_run_name}_{timestamp}.log")
    f = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, f)
    print(f"Logging initialized. Output saved to: {os.path.abspath(log_path)}")
    _LOG_FILE = f
    _LOG_INITIALIZED = True
    return f


def init_wandb(project: str, run_name: Optional[str], wandb_dir: str):
    global _WANDB_INITIALIZED
    if _WANDB_INITIALIZED:
        return None
    try:
        import wandb as _wandb
        _wandb.init(project=project, dir=wandb_dir, name=run_name)
        _WANDB_INITIALIZED = True
        return _wandb
    except Exception as e:
        eval_logger.warning(f"wandb init failed: {e}")
        return None


def extract_gold_answer(doc: dict) -> str:
    if not isinstance(doc, dict):
        return ""
    for key in ("answer", "gold", "target", "targets", "solution", "canonical_solution", "output"):
        if key in doc:
            val = doc[key]
            if isinstance(val, list):
                return str(val[0]) if val else ""
            return str(val)
    return ""


def infer_dataset_type(doc: dict) -> str:
    if isinstance(doc, dict):
        task_id = str(doc.get("task_id", "")).lower()
        if task_id.startswith("humaneval"):
            return "humaneval"
        if "mbpp" in task_id:
            return "mbpp"
        dataset = str(doc.get("dataset", "")).lower()
        if dataset in ("gsm8k", "math", "humaneval", "mbpp"):
            return dataset
        if "math" in dataset:
            return "math"
    return "gsm8k"

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
        log_dir: str = "./eval_log",
        enable_tee: bool = True,
        wandb: bool = False,
        wandb_project: str = "adaptive-dllm-eval",
        wandb_run_name: Optional[str] = None,
        wandb_dir: str = "./wandb_log",
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
        self._wandb = None
        run_name = build_eval_run_name(
            model_type=model_type,
            pretrained=pretrained,
            policy_path=policy_path,
            guidance_model_name=guidance_model_name,
            gen_length=self.gen_length,
            steps=self.steps,
            block_len_max=self.block_len_max,
        )
        if enable_tee:
            setup_eval_logging(log_dir=log_dir, run_name=run_name)
        if wandb:
            self._wandb = init_wandb(project=wandb_project, run_name=wandb_run_name or run_name, wandb_dir=wandb_dir)
        
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
        if guidance_model_name is None:
            if model_type == "llada":
                # guidance_model_name = "meta-llama/Llama-3.2-1B-Instruct"
                guidance_model_name = "Qwen/Qwen3-0.6B"
                # guidance_model_name = "pytorch/Phi-4-mini-instruct-INT4"
                # guidance_model_name = "HuggingFaceTB/SmolLM2-1.7B"
            elif model_type == "dream":
                guidance_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        if guidance_model_name:
            eval_logger.info(f"Loading guidance model: {guidance_model_name}")
            self.ar_guidance_model = AutoModelForCausalLM.from_pretrained(
                guidance_model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            ).to(self.device)
            self.ar_guidance_model.eval()
        else:
            eval_logger.info(f"Warning: guidance model not exists!")

        if use_cache or dual_cache:
            eval_logger.warning("Prefix/Dual Cache requested but Adaptive Policy rollout does not strictly implement KV-caching yet. Running standard adaptive rollout.")

    def generate_until(self, requests):
        if not requests:
            return []
            
        results = []
        correct_count = 0
        total_count = 0
        total_nfe = 0
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

            gold = extract_gold_answer(req.doc) if hasattr(req, "doc") else ""
            dataset_type = infer_dataset_type(req.doc) if hasattr(req, "doc") else "gsm8k"
            is_correct = is_correct_smart(gen_text, gold, dataset_type) if gold else False
            total_count += 1
            if is_correct:
                correct_count += 1
            acc = correct_count / max(1, total_count)
            if hasattr(res, "nfe"):
                total_nfe += int(res.nfe)
            avg_nfe = total_nfe / max(1, total_count)

            print("=" * 60)
            print(f"Accuracy so far: {acc:.2%} ({correct_count}/{total_count})")
            print("Question:")
            print(prompt)
            print("Answer:")
            print(gen_text)
            print("Gold Answer:")
            print(gold)
            print(f"Correct: {is_correct}")
            print(f"NFE (this): {getattr(res, 'nfe', 0)} | Avg NFE: {avg_nfe:.2f}")
            print("=" * 60)

            if self._wandb is not None:
                self._wandb.log({
                    "eval/accuracy": acc,
                    "eval/correct": int(is_correct),
                    "eval/step": total_count,
                })
            
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
