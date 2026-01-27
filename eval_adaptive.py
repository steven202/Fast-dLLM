
import argparse
import re
import os
import sys
import torch
import logging
from tqdm import tqdm
from typing import Optional, Union, List
from collections import Counter

# Set environment variables for evaluation safety and remote code
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"
_fast_dllm_datasets_dir = os.environ.get("FAST_DLLM_DATASETS_DIR")
if _fast_dllm_datasets_dir:
    os.environ.setdefault(
        "HF_DATASETS_CACHE",
        os.path.expanduser(_fast_dllm_datasets_dir),
    )
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
    from utils.eval_utils import is_correct_smart, extract_answer_math
    from utils.logging_utils import Tee
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

eval_logger = logging.getLogger(__name__)

_LOG_FILE = None
_LOG_INITIALIZED = False
_WANDB_INITIALIZED = False


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y", "on"):
            return True
        if v in ("0", "false", "f", "no", "n", "off", ""):
            return False
    return bool(value)


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return int(default)
        try:
            return int(v)
        except Exception:
            return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _sanitize_name_segment(value: str) -> str:
    value = (value or "").strip()
    for ch in ["/", "\\", ":", " "]:
        value = value.replace(ch, "_")
    return value


def infer_aligner_type_from_policy(policy_path: str) -> Optional[str]:
    if not policy_path:
        return None
    name = os.path.basename(policy_path).lower()
    if "_cached_" in name:
        return "cached"
    if "_robust_" in name:
        return "robust"
    if "_static_" in name:
        return "static"
    return None


def build_eval_run_name(
    model_type: str,
    pretrained: str,
    policy_path: str,
    guidance_model_name: Optional[str],
    aligner_type: str,
    threshold_impl: Optional[str],
    gen_length: int,
    steps: int,
    block_len_max: int,
) -> str:
    policy_part = _sanitize_name_segment(os.path.splitext(os.path.basename(policy_path))[0])
    pretrained_part = _sanitize_name_segment(pretrained)
    guidance_part = _sanitize_name_segment(guidance_model_name or "none")
    aligner_part = _sanitize_name_segment(aligner_type or "static")
    if threshold_impl is None:
        threshold_part = "llada" if model_type == "llada" else "dream"
    else:
        threshold_part = threshold_impl
    threshold_part = _sanitize_name_segment(threshold_part)
    return f"{model_type}_{policy_part}_{pretrained_part}_{guidance_part}_{aligner_part}_{threshold_part}_L{gen_length}_S{steps}_B{block_len_max}"


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


def init_wandb(project: str, run_name: Optional[str], wandb_dir: str, tee_log_path: Optional[str] = None):
    global _WANDB_INITIALIZED
    if _WANDB_INITIALIZED:
        return None
    try:
        import wandb as _wandb
        wandb_config = {"tee_log_path": tee_log_path} if tee_log_path else None
        _wandb.init(project=project, dir=wandb_dir, name=run_name, config=wandb_config)
        if _wandb.run is not None:
            print(f"WandB Log Dir (tee path): {_wandb.run.dir}")
            if tee_log_path:
                print(f"Tee Log Path: {tee_log_path}")
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
    checkpoint = torch.load(policy_path, map_location=model.device)
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        state_dict = checkpoint["policy_state_dict"]
    else:
        state_dict = checkpoint
    
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
        wandb_hist_every: int = 10,
        # Generation/Policy args
        gen_length: int = 256,
        steps: int = 256,
        block_len_min: int = 8,
        block_len_max: int = 64,
        threshold: Optional[float] = None,
        threshold_impl: Optional[str] = None,
        guidance_gamma: float = 0.5,
        guidance_temperature: float = 0.5,
        aligner_type: str = "static",
        # CLI compatibility args (may not be used but passed by scripts)
        use_cache: Optional[bool] = None,
        dual_cache: Optional[bool] = None,
        factor: float = 1.0,
        gumbel_temperature: Optional[float] = None,
        show_speed: bool = False,
        show_actions: bool = False,
        **kwargs
    ):
        super().__init__()
        self.model_type = model_type
        if threshold is None:
            threshold = 0.9
        if threshold_impl is None:
            if model_type == "llada":
                threshold_impl = "llada"
                if use_cache is None:
                    use_cache = True
                if dual_cache is None:
                    dual_cache = True
                if gumbel_temperature is None:
                    gumbel_temperature = 0.0
            else:
                threshold_impl = "dream"
                if use_cache is None:
                    use_cache = True
                if dual_cache is None:
                    dual_cache = True
                if gumbel_temperature is None:
                    gumbel_temperature = 0.0
        else:
            if use_cache is None:
                use_cache = False
            if dual_cache is None:
                dual_cache = False
            if gumbel_temperature is None:
                gumbel_temperature = 0.0
        if threshold_impl != "llada" and factor is not None:
            eval_logger.warning(
                f"factor is only supported for LLaDA. Ignoring factor for threshold_impl='{threshold_impl}'."
            )
            factor = None
        self.device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.block_len_min = int(block_len_min)
        self.block_len_max = int(block_len_max)
        self.threshold = float(threshold)
        self.threshold_impl = str(threshold_impl)
        self.guidance_gamma = float(guidance_gamma)
        self.guidance_temperature = float(guidance_temperature)
        self.factor = None if factor is None else float(factor)
        self.gumbel_temperature = float(gumbel_temperature)
        self.show_speed = _to_bool(show_speed)
        self.show_actions = _to_bool(show_actions)
        self.use_cache = _to_bool(use_cache)
        self.dual_cache = _to_bool(dual_cache)
        self.wandb_hist_every = _to_int(wandb_hist_every, default=10)
        self._action_counter = Counter()
        self._block_len_counter = Counter()
        self._hist_buf_action_blocklens: List[int] = []
        self._hist_buf_blocklens: List[int] = []
        self._wandb = None
        inferred_aligner = infer_aligner_type_from_policy(policy_path)
        if aligner_type == "static" and inferred_aligner and inferred_aligner != aligner_type:
            eval_logger.warning(
                f"aligner_type='{aligner_type}' but policy filename suggests '{inferred_aligner}'. Using '{inferred_aligner}'."
            )
            aligner_type = inferred_aligner

        run_name = build_eval_run_name(
            model_type=model_type,
            pretrained=pretrained,
            policy_path=policy_path,
            guidance_model_name=guidance_model_name,
            aligner_type=aligner_type,
            threshold_impl=threshold_impl,
            gen_length=self.gen_length,
            steps=self.steps,
            block_len_max=self.block_len_max,
        )
        tee_log_file = None
        if _to_bool(enable_tee):
            tee_log_file = setup_eval_logging(log_dir=log_dir, run_name=run_name)
        tee_log_path = os.path.abspath(tee_log_file.name) if tee_log_file is not None else None
        if _to_bool(wandb):
            self._wandb = init_wandb(project=wandb_project, run_name=wandb_run_name or run_name, wandb_dir=wandb_dir, tee_log_path=tee_log_path)
        
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

        # Attach aligner if guidance model exists
        if self.ar_guidance_model is not None:
            try:
                from utils.aligner import StaticTokenAligner, RobustTokenAligner, CachedTokenAligner
                ar_tokenizer = AutoTokenizer.from_pretrained(guidance_model_name, trust_remote_code=True)
                if aligner_type == "static":
                    aligner = StaticTokenAligner(ar_tokenizer, self.tokenizer, device=self.device)
                elif aligner_type == "robust":
                    aligner = RobustTokenAligner(ar_tokenizer, self.tokenizer, device=self.device)
                else:
                    aligner = CachedTokenAligner(ar_tokenizer, self.tokenizer, device=self.device)
                self.ar_guidance_model.logit_aligner = aligner
                eval_logger.info(f"Attached {aligner_type} aligner to guidance model.")
            except Exception as e:
                eval_logger.warning(f"Could not attach aligner: {e}")

        if self.use_cache and not self.dual_cache:
            eval_logger.warning("Prefix cache requested without dual_cache. Continuing without forcing dual_cache.")
        if self.use_cache or self.dual_cache:
            eval_logger.info("Cache enabled for eval_adaptive. Using cached adaptive rollout path.")

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
                        threshold=self.threshold,
                        threshold_impl=self.threshold_impl,
                        factor=self.factor,
                        gumbel_temperature=self.gumbel_temperature,
                        use_cache=self.use_cache,
                        dual_cache=self.dual_cache,
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
                        threshold=self.threshold,
                        threshold_impl=self.threshold_impl,
                        factor=self.factor,
                        gumbel_temperature=self.gumbel_temperature,
                        use_cache=self.use_cache,
                        dual_cache=self.dual_cache,
                    )

            # Decode just the new tokens
            prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            prompt_len = prompt_ids.shape[1]
            
            if res.generated_ids.shape[1] > prompt_len:
                new_ids = res.generated_ids[0, prompt_len:]
                gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            else:
                gen_text = ""


            gold = extract_gold_answer(req.doc) if hasattr(req, "doc") else ""
            dataset_type = infer_dataset_type(req.doc) if hasattr(req, "doc") else "gsm8k"

            if self.model_type == "llada":
                for stop in list(until):
                    if stop in gen_text:
                        gen_text = gen_text.split(stop)[0]
            else:  # dream
                eos_token = self.tokenizer.eos_token
                if eos_token and eos_token in gen_text:
                    gen_text = gen_text.split(eos_token)[0]
                extra_stops = [
                    "\nQuestion:",
                    "\n\nQuestion:",
                    "\nQ:",
                    "\n\nQ:",
                    "\nIn the context",
                    "\n\nIn the context",
                    "user:",
                    "assistant:",
                    "User:",
                    "Assistant:",
                ]
                for stop in list(until) + extra_stops:
                    if stop in gen_text:
                        gen_text = gen_text.split(stop)[0]

                if "\n\n" in gen_text:
                    gen_text = gen_text.split("\n\n", 1)[0]

                if gen_text.lstrip().startswith("Answer:"):
                    gen_text = gen_text.lstrip().split("Answer:", 1)[1].lstrip()
                raw_gen_text = gen_text
                eval_text = raw_gen_text
                if dataset_type == "gsm8k" and "####" in raw_gen_text:
                    after = raw_gen_text.split("####", 1)[1]
                    match = re.search(r"-?\d+\.?\d*", after.replace(",", ""))
                    if match:
                        eval_text = f"#### {match.group(0)}"
                elif dataset_type == "math" and "####" in raw_gen_text:
                    pred_val = extract_answer_math(raw_gen_text)
                    if pred_val is not None:
                        eval_text = f"#### {pred_val}"
            if self.model_type == "llada":
                is_correct = is_correct_smart(gen_text, gold, dataset_type) if gold else False
            else:
                is_correct = is_correct_smart(eval_text, gold, dataset_type) if gold else False

            # ---- Action / BlockLen analysis (per-rollout + running distribution) ----
            policy_actions = list(getattr(res, "policy_actions", []) or [])
            block_lens = list(getattr(res, "block_lens", []) or [])
            if block_lens:
                self._block_len_counter.update(block_lens)
                self._hist_buf_blocklens.extend(block_lens)
            if policy_actions:
                # policy_actions are (block_len - 1)
                self._action_counter.update(policy_actions)
                self._hist_buf_action_blocklens.extend([a + 1 for a in policy_actions])

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
            if self.model_type == "llada":
                print(gen_text)
            else:
                print(raw_gen_text)
            print("Gold Answer:")
            print(gold)
            print(f"Correct: {is_correct}")
            print(f"NFE (this): {getattr(res, 'nfe', 0)} | Avg NFE: {avg_nfe:.2f}")

            if self.show_actions or self.show_speed:
                if policy_actions:
                    # Convert actions -> block_len for easier interpretation
                    action_block_lens = [a + 1 for a in policy_actions]
                    action_counter = Counter(action_block_lens)
                    top_actions = action_counter.most_common(8)
                    print(f"Policy Actions (block_len-1): {policy_actions}")
                    print(f"Action BlockLens (a+1): {action_block_lens}")
                    print(f"Action Dist Top8: {top_actions}")
                else:
                    print("Policy Actions: (none)")

                if block_lens:
                    bl_counter = Counter(block_lens)
                    top_bl = bl_counter.most_common(8)
                    print(f"Block Lengths: {block_lens}")
                    print(f"BlockLen Dist Top8: {top_bl}")
                else:
                    print("Block Lengths: (none)")

                # Running (cumulative) view across requests
                if self._action_counter:
                    running_top_actions = self._action_counter.most_common(8)
                    # Display as block_len not action index
                    running_top_actions = [(a + 1, c) for (a, c) in running_top_actions]
                    print(f"Running Action BlockLen Top8: {running_top_actions}")
                if self._block_len_counter:
                    running_top_bl = self._block_len_counter.most_common(8)
                    print(f"Running BlockLen Top8: {running_top_bl}")

            print("=" * 60)

            if self._wandb is not None:
                wandb_payload = {
                    "eval/accuracy": acc,
                    "eval/correct": int(is_correct),
                    "eval/step": total_count,
                }

                # Light-weight per-sample scalars
                if self.show_actions or self.show_speed:
                    if policy_actions:
                        action_block_lens = [a + 1 for a in policy_actions]
                        wandb_payload.update({
                            "eval/action_blocklen_mean": float(sum(action_block_lens)) / max(1.0, float(len(action_block_lens))),
                            "eval/action_blocklen_max": float(max(action_block_lens)),
                            "eval/action_blocklen_min": float(min(action_block_lens)),
                        })

                # Heavy histograms: log every N eval steps (0 disables)
                hist_logged = False
                if self.wandb_hist_every and (total_count == 1 or (total_count % self.wandb_hist_every == 0)):
                    if self._hist_buf_action_blocklens:
                        wandb_payload["eval/dist_action_blocklen"] = self._wandb.Histogram(self._hist_buf_action_blocklens)
                    if self._hist_buf_blocklens:
                        wandb_payload["eval/dist_blocklen"] = self._wandb.Histogram(self._hist_buf_blocklens)
                    # reset buffers to keep memory bounded
                    self._hist_buf_action_blocklens.clear()
                    self._hist_buf_blocklens.clear()
                    hist_logged = True

                wandb_payload["eval/WandbHistLogged"] = int(hist_logged)
                self._wandb.log(wandb_payload)
            
            if self.model_type == "llada":
                results.append(gen_text)
            else:
                results.append(raw_gen_text)
            
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
