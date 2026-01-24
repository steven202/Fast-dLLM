"""
Adaptive-dLLM GDPO Training (Robust, Merged, Efficiency Optimized)

Supports both LLaDA and Dream models.
Uses GSM8K + HumanEval prompts for training.
Fixed: 
1. Replaced LogitAligner with StaticTokenAligner for 10x-100x speedup.
   - Moves string processing to pre-computation phase (CPU).
   - Uses pure GPU operations (indexing/scattering) during training loop.
2. Preserved all numerical stability fixes (Entropy, Input Sanitization).
"""

import argparse
import os
import math
import time
import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import set_seed, str2bool
from utils.aligner import StaticTokenAligner
from utils.eval_utils import is_correct_smart
from utils.logging_utils import setup_logging
from utils.data_utils import load_prompts
from utils.model_utils import get_mask_id

from llada.model.modeling_llada import LLaDAModelLM
from dream.model.modeling_dream import DreamModel
from tqdm import tqdm


@dataclass
class RolloutResult:
    generated_ids: torch.LongTensor
    block_lens: List[int]
    logprob: torch.Tensor
    entropy_ccd: torch.Tensor
    policy_features: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    policy_actions: List[int]
    nfe: int = 0


def policy_logits(policy_net: torch.nn.Module, hidden: torch.Tensor, entropy: torch.Tensor, guidance_entropy: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :]
    # [FIX] Force entropy to match hidden dtype (BFloat16) to avoid matmul error
    entropy = entropy.to(dtype=hidden.dtype).view(entropy.size(0), 1)
    if guidance_entropy.dim() == 0:
        guidance_entropy = guidance_entropy.unsqueeze(0)
    guidance_entropy = guidance_entropy.to(dtype=hidden.dtype).view(guidance_entropy.size(0), 1)
    x = torch.cat([hidden, entropy, guidance_entropy], dim=-1)
    x = policy_net.act(policy_net.fc1(x))
    return policy_net.fc2(x)


def sample_block_len(policy_net, hidden: torch.Tensor, entropy: torch.Tensor, guidance_entropy: torch.Tensor, min_len: int, max_len: int) -> Tuple[int, torch.Tensor]:
    # [FIX] Sanitize inputs: replace NaN/Inf with 0 to prevent network pollution
    # hidden = torch.nan_to_num(hidden, nan=0.0, posinf=0.0, neginf=0.0)
    # entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure dtype match (in case entropy is float32 and hidden is bfloat16)
    # entropy = entropy.to(hidden.dtype)

    logits = policy_logits(policy_net, hidden, entropy, guidance_entropy)
    # [FIX] Sanitize logits: replace NaN in logits to prevent categorical sampling crash
    # logits = torch.nan_to_num(logits, nan=-100.0, posinf=100.0, neginf=-100.0)

    # Mask logits to ensure valid block_len range
    if max_len < logits.size(-1):
        logits[:, max_len:] = float('-inf')
    if min_len > 1:
        logits[:, :min_len-1] = float('-inf')

    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    block_len = action.item() + 1
    
    return block_len, logprob


def apply_guidance_llada(model: LLaDAModelLM, hidden: torch.Tensor, guidance_target: Optional[torch.Tensor], guidance_gamma: float) -> torch.Tensor:
    if guidance_target is None or guidance_gamma == 0.0:
        return hidden
    if guidance_target.dim() == 2:
        guidance_target = guidance_target.unsqueeze(1)
    return hidden - guidance_gamma * (guidance_target - hidden)


def apply_guidance_dream(model: DreamModel, hidden: torch.Tensor, guidance_target: Optional[torch.Tensor], guidance_gamma: float) -> torch.Tensor:
    if guidance_target is None or guidance_gamma == 0.0:
        return hidden
    if guidance_target.dim() == 2:
        guidance_target = guidance_target.unsqueeze(1)
    return hidden - guidance_gamma * (guidance_target - hidden)


def get_transfer_index(
    logits: torch.Tensor,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    threshold: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_with_noise = logits
    x0 = torch.argmax(logits_with_noise, dim=-1)

    p = F.softmax(logits.to(torch.float32), dim=-1)
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = transfer_index | force_mask
        transfer_index = transfer_index & mask_index
        return x0, transfer_index

    raise ValueError("threshold must be set for training rollouts")


def rollout_llada(
    model: LLaDAModelLM,
    policy_net: torch.nn.Module,
    tokenizer,
    prompt: str,
    ar_guidance_model,
    guidance_gamma: float,
    guidance_temperature: float,
    gen_length: int,
    steps: int,
    block_len_min: int,
    block_len_max: int,
    threshold: float,
) -> RolloutResult:
    device = model.device
    
    # --- [MODIFIED] Safe Vocab Access ---
    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 126464 # Fallback
        
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # --- [CRITICAL FIX] Sanitize Input IDs ---
    # The tokenizer might produce tokens outside LLaDA's embedding range.
    if input_ids.max() >= backbone_vocab_size:
        input_ids = torch.where(input_ids >= backbone_vocab_size, torch.tensor(0, device=device), input_ids)
    # -----------------------------------------
    
    mask_id = get_mask_id("llada", model)

    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    nfe = 0
    current_len = 0
    steps_remaining = steps
    entropy_accum = []
    block_lens = []
    logprobs = []
    policy_features = []
    policy_actions = []

    # [FIX] Use bidirectional attention bias for MDM/Diffusion generation
    def get_attention_bias(length, device):
        return torch.zeros((1, 1, length, length), device=device, dtype=torch.float)

    while current_len < gen_length:
        context_len = input_ids.shape[1] + current_len
        context_ids = x[:, :context_len]
        
        att_bias = get_attention_bias(context_len, device)
        outputs = model(context_ids, attention_bias=att_bias, output_hidden_states=True)
        
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_logits = outputs.logits[:, -1, :]
        
        # [FIX] Stable Entropy
        # entropy = torch.distributions.Categorical(logits=last_logits).entropy()
        entropy = compute_sparse_entropy(last_logits, topk=50)

        # Guidance Logic moved up
        guidance_entropy = torch.tensor(0.0, device=device)
        ar_logits = None
        guidance_target = None

        if ar_guidance_model is not None:
            # Use Static Aligner if attached
            if hasattr(ar_guidance_model, "logit_aligner") and ar_guidance_model.logit_aligner is not None:
                # 1. Translate Context (Backbone -> Guidance)
                ar_input_ids = ar_guidance_model.logit_aligner.translate_input(context_ids)
                
                # 2. Get Raw Logits from Guidance
                ar_logits_src = ar_guidance_model(ar_input_ids).logits[:, -1, :]
                guidance_entropy = compute_sparse_entropy(ar_logits_src, topk=50)

                # 3. Align to Backbone Vocab (GPU Operation)
                ar_logits = ar_guidance_model.logit_aligner.align(ar_logits_src, backbone_vocab_size)
            else:
                # Fallback: Raw logits + Slicing
                ar_logits_full = ar_guidance_model(context_ids).logits[:, -1, :]
                guidance_entropy = compute_sparse_entropy(ar_logits_full, topk=50)

                if ar_logits_full.shape[-1] > backbone_vocab_size:
                    ar_logits = ar_logits_full[:, :backbone_vocab_size]
                else:
                    ar_logits = ar_logits_full

            guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)
        
        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, guidance_entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)

        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len

        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach(), guidance_entropy.detach()))
        policy_actions.append(block_len - 1)

        i = 0
        while True:
            if i >= steps_per_block:
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0

            att_bias = get_attention_bias(x.shape[1], device)
            outputs = model(x, attention_bias=att_bias, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            hidden_states = apply_guidance_llada(model, hidden_states, guidance_target, guidance_gamma)

            if model.model.config.weight_tying:
                logits = F.linear(hidden_states, model.model.transformer.wte.weight, None)
            else:
                logits = model.model.transformer.ff_out(hidden_states)

            if hasattr(model.model.config, "scale_logits") and model.model.config.scale_logits:
                 logits = logits * (1 / math.sqrt(model.model.config.d_model))
            
            mask_logits = logits[mask_index]
            step_entropy = torch.distributions.Categorical(logits=mask_logits).entropy().mean()
            entropy_accum.append(step_entropy)

            x0, transfer_index = get_transfer_index(logits, mask_index, x, threshold)
            x[transfer_index] = x0[transfer_index]

            i += 1
            nfe += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

        current_len += block_len

    entropy_ccd = torch.stack(entropy_accum).mean() if entropy_accum else torch.tensor(0.0, device=device)
    logprob_sum = torch.stack(logprobs).sum()

    return RolloutResult(
        generated_ids=x,
        block_lens=block_lens,
        logprob=logprob_sum,
        entropy_ccd=entropy_ccd,
        policy_features=policy_features,
        policy_actions=policy_actions,
        nfe=nfe,
    )


def rollout_dream(
    model: DreamModel,
    policy_net: torch.nn.Module,
    tokenizer,
    prompt: str,
    ar_guidance_model,
    guidance_gamma: float,
    guidance_temperature: float,
    gen_length: int,
    steps: int,
    block_len_min: int,
    block_len_max: int,
    threshold: float,
) -> RolloutResult:
    device = model.device
    
    # --- [MODIFIED] Safe Vocab Access for Dream ---
    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 32000 
    # ----------------------------------------------
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # --- [CRITICAL FIX] Sanitize Input IDs ---
    if input_ids.max() >= backbone_vocab_size:
        input_ids = torch.where(input_ids >= backbone_vocab_size, torch.tensor(0, device=device), input_ids)
    # -----------------------------------------
    
    mask_id = get_mask_id("dream", model)

    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    current_len = 0
    nfe = 0
    steps_remaining = steps
    entropy_accum = []
    block_lens = []
    logprobs = []
    policy_features = []
    policy_actions = []

    while current_len < gen_length:
        context_len = input_ids.shape[1] + current_len
        context_ids = x[:, :context_len]
        position_ids = torch.arange(context_len, device=device).unsqueeze(0)
        outputs = model(context_ids, position_ids=position_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_logits = outputs.logits[:, -1, :]
        
        # [FIX] Stable Entropy
        # entropy = torch.distributions.Categorical(logits=last_logits).entropy()
        entropy = compute_sparse_entropy(last_logits, topk=50)

        # Guidance Logic moved up
        guidance_entropy = torch.tensor(0.0, device=device)
        ar_logits = None
        guidance_target = None
        
        if ar_guidance_model is not None:
             if hasattr(ar_guidance_model, "logit_aligner") and ar_guidance_model.logit_aligner is not None:
                ar_input_ids = ar_guidance_model.logit_aligner.translate_input(context_ids)
                ar_logits_src = ar_guidance_model(ar_input_ids).logits[:, -1, :]
                guidance_entropy = compute_sparse_entropy(ar_logits_src, topk=50)
                ar_logits = ar_guidance_model.logit_aligner.align(ar_logits_src, backbone_vocab_size)
             else:
                ar_logits_full = ar_guidance_model(context_ids).logits[:, -1, :]
                guidance_entropy = compute_sparse_entropy(ar_logits_full, topk=50)
                if ar_logits_full.shape[-1] > backbone_vocab_size:
                    ar_logits = ar_logits_full[:, :backbone_vocab_size]
                else:
                    ar_logits = ar_logits_full
             
             guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)

        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, guidance_entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)

        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len

        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach(), guidance_entropy.detach()))
        policy_actions.append(block_len - 1)

        i = 0
        while True:
            if i >= steps_per_block:
                break

            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0
            position_ids = torch.arange(x.shape[1], device=device).unsqueeze(0)
            outputs = model(x, position_ids=position_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            hidden_states = apply_guidance_dream(model, hidden_states, guidance_target, guidance_gamma)
            logits = model.lm_head(hidden_states)
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            mask_logits = logits[mask_index]
            step_entropy = torch.distributions.Categorical(logits=mask_logits).entropy().mean()
            entropy_accum.append(step_entropy)

            x0, transfer_index = get_transfer_index(logits, mask_index, x, threshold)
            x[transfer_index] = x0[transfer_index]

            i += 1
            nfe += 1
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

        current_len += block_len

    entropy_ccd = torch.stack(entropy_accum).mean() if entropy_accum else torch.tensor(0.0, device=device)
    logprob_sum = torch.stack(logprobs).sum()

    return RolloutResult(
        generated_ids=x,
        block_lens=block_lens,
        logprob=logprob_sum,
        entropy_ccd=entropy_ccd,
        policy_features=policy_features,
        policy_actions=policy_actions,
        nfe=nfe,
    )


def prepare_reward_cache(reward_model, reward_tokenizer, prompt, device):
    """
    Runs the reward model on the prompt ONCE to cache the KV states.
    Returns:
        prompt_last_logit: The prediction for the first generated token.
        past_key_values: The cached KV states for the prompt.
    """
    # Tokenize prompt
    prompt_inputs = reward_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Run forward pass with use_cache=True
        outputs = reward_model(**prompt_inputs, use_cache=True)
    
    # We need the logit for the *next* token (the first token of generation)
    # Shape: [1, vocab_size]
    prompt_last_logit = outputs.logits[:, -1, :]
    
    return prompt_last_logit, outputs.past_key_values

def compute_sparse_entropy(logits: torch.Tensor, topk: int = 50) -> torch.Tensor:
    """
    Approximates entropy by only considering the top-k most likely tokens.
    
    Why this helps:
    - Standard Entropy: Softmax(128,000) -> Log -> Sum. Very slow on large vocabs.
    - Sparse Entropy:   TopK(50) -> Softmax(50) -> Log -> Sum. Extremely fast.
    
    The Categorical distribution automatically renormalizes the Top-K logits
    so they sum to 1, providing a stable "local uncertainty" metric.
    """
    # Safety: ensure we don't request more tokens than exist
    k = min(topk, logits.size(-1))
    
    # 1. Get Top-K values (We don't need the indices)
    top_logits, _ = logits.topk(k, dim=-1)
    
    # 2. Compute entropy on this smaller subset
    return torch.distributions.Categorical(logits=top_logits).entropy()

def compute_reward_nll(
    reward_model, 
    reward_tokenizer, 
    prompt_cache, 
    generated_text, 
    device
) -> torch.Tensor:
    """
    Computes NLL using the pre-computed prompt cache.
    Only processes the new generated_text tokens.
    """
    prompt_last_logit, past_key_values = prompt_cache
    
    # Tokenize ONLY the generated text
    # add_special_tokens=False is critical because the prompt handles the start
    gen_inputs = reward_tokenizer(generated_text, add_special_tokens=False, return_tensors="pt").to(device)
    gen_ids = gen_inputs.input_ids
    
    # Edge case: Empty generation
    if gen_ids.size(1) == 0:
        return torch.tensor(0.0, device=device)
        
    with torch.no_grad():
        # Feed gen_ids + past_key_values
        # The model will only process the new tokens
        outputs = reward_model(input_ids=gen_ids, past_key_values=past_key_values, use_cache=True)
        gen_logits = outputs.logits
        
        # --- Construct Prediction Logits ---
        # 1. Prediction for gen_ids[0] comes from prompt_last_logit
        # 2. Prediction for gen_ids[1:] comes from gen_logits[:, :-1, :]
        
        if gen_ids.shape[1] > 1:
            # Concat [Prompt Last Logit] + [Gen Logits excluding the very last step]
            prediction_logits = torch.cat([prompt_last_logit.unsqueeze(1), gen_logits[:, :-1, :]], dim=1)
        else:
            # Only one token generated; prediction comes solely from prompt
            prediction_logits = prompt_last_logit.unsqueeze(1)
            
        # --- Calculate NLL ---
        loss = F.cross_entropy(
            prediction_logits.reshape(-1, prediction_logits.size(-1)),
            gen_ids.reshape(-1),
            reduction="none"
        )
        
        return loss.mean()


def compute_r_speed(block_lens: List[int], min_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    L_avg = sum(block_lens) / max(1, len(block_lens))
    return torch.tensor((L_avg - min_len) / max(1e-6, (max_len - min_len)), device=device)


def ppo_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    ratio = torch.exp(new_logprob - old_logprob)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    return -torch.min(ratio * advantage, clipped * advantage).mean()

def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["llada", "dream"], default="llada")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--resume", type=str2bool, default=False)
    # [MODIFIED] Default is None so we can set it dynamically based on model_type
    parser.add_argument("--ar_guidance_model", type=str, default=None)
    
    parser.add_argument("--ar_reward_model", type=str, choices=["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-8B"], default="Qwen/Qwen3-8B")
    parser.add_argument("--reward_offload_dir", type=str, default="./offload_reward")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--datasets", nargs="+", default=["gsm8k"], 
                   choices=["gsm8k", "math", "humaneval", "mbpp"],
                   help="List of datasets to use")
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_len_min", type=int, default=8)
    parser.add_argument("--block_len_max", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)

    parser.add_argument("--guidance_gamma", type=float, default=0.5)
    parser.add_argument("--guidance_temperature", type=float, default=0.5)

    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=0.0)
    parser.add_argument("--w_acc", type=float, default=1.0)
    parser.add_argument("--lambda_ccd", type=float, default=0.1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=True)
    args = parser.parse_args()
    set_seed(args.seed)
    
    # --- [MODIFIED] Automatic Guidance Model Selection ---
    if args.ar_guidance_model is None:
        if args.model_type == "llada":
            args.ar_guidance_model = "meta-llama/Llama-3.2-1B-Instruct"
            print(f"Selected Guidance for LLaDA: {args.ar_guidance_model}")
        elif args.model_type == "dream":
            args.ar_guidance_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"Selected Guidance for Dream: {args.ar_guidance_model}")
    # ---------------------------------------------------

    if args.save_path is None:
        args.save_path = f"./checkpoints/policy_{args.model_type}.pt"
    if args.load_path is None:
        args.load_path = f"./checkpoints/policy_{args.model_type}.pt"

    prompts = load_prompts(args.datasets, max_samples=args.max_samples)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model_type == "llada":
        model = LLaDAModelLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        policy_net = model.model.block_policy
    else:
        model = DreamModel.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        policy_net = model.block_policy

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in policy_net.parameters():
        p.requires_grad = True

    if args.resume and args.load_path and os.path.exists(args.load_path):
        is_corrupted = False
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                is_corrupted = True
            if torch.isinf(param).any():
                is_corrupted = True
        if is_corrupted:
            print("Warning: Model parameters contain NaN or Inf. Skipping loading policy.")
        else:
            print(f"Loading policy from {os.path.abspath(args.load_path)}")
            policy_net.load_state_dict(torch.load(args.load_path, map_location=device))
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=args.lr)

    ar_guidance_model = AutoModelForCausalLM.from_pretrained(
        args.ar_guidance_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ar_guidance_model.eval()

    try:
        ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_guidance_model, trust_remote_code=True)
        # Use the optimized StaticTokenAligner
        aligner = StaticTokenAligner(ar_tokenizer, tokenizer, device=device)
        ar_guidance_model.logit_aligner = aligner
        print("Guidance Tokenizer loaded and Static Aligner attached.")
    except Exception as e:
        print(f"Warning: Could not load guidance tokenizer: {e}")
        ar_guidance_model.logit_aligner = None

    os.makedirs(args.reward_offload_dir, exist_ok=True)
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.ar_reward_model,
        dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        offload_folder=args.reward_offload_dir,
    ).eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(args.ar_reward_model, trust_remote_code=True)

    use_wandb = args.wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project="adaptive-dllm", dir="./wandb_log", config=vars(args), name=f"policy_{args.model_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if wandb.run is not None:
                print(f"WandB Run URL: {wandb.run.url}")
                print(f"WandB Project URL: {wandb.run.project_url}")
        except Exception:
            use_wandb = False
            print("wandb import failed.")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        correct_count = 0
        total_count = 0
        for i, item in enumerate(tqdm(prompts, desc="Prompts")):
            prompt = item["prompt"]
            answer = item["answer"]
            
            # Pre-compute reward cache
            prompt_cache = prepare_reward_cache(reward_model, reward_tokenizer, prompt, device)

            start_time = time.time()
            rollouts: List[RolloutResult] = []
            r_qual_list = []
            dataset_type = item.get("dataset", "gsm8k")
            
            r_speed_list = []
            r_ccd_list = []
            r_acc_list = []

            for _ in range(args.rollouts):
                if args.model_type == "llada":
                    rollout = rollout_llada(
                        model,
                        policy_net,
                        tokenizer,
                        prompt,
                        ar_guidance_model,
                        args.guidance_gamma,
                        args.guidance_temperature,
                        args.gen_length,
                        args.steps,
                        args.block_len_min,
                        args.block_len_max,
                        args.threshold,
                    )
                else:
                    rollout = rollout_dream(
                        model,
                        policy_net,
                        tokenizer,
                        prompt,
                        ar_guidance_model,
                        args.guidance_gamma,
                        args.guidance_temperature,
                        args.gen_length,
                        args.steps,
                        args.block_len_min,
                        args.block_len_max,
                        args.threshold,
                    )

                gen_text = tokenizer.decode(rollout.generated_ids[0, input_ids_len(prompt, tokenizer):], skip_special_tokens=True)
                
                # Accuracy Check
                # Accuracy Check
                is_right = is_correct_smart(gen_text, answer, dataset_type)
                if is_right:
                    correct_count += 1
                total_count += 1

                nll = compute_reward_nll(reward_model, reward_tokenizer, prompt_cache, gen_text, device=device)
                r_qual = torch.exp(-nll)
                r_acc = torch.tensor(1.0 if is_right else 0.0, device=device)
                r_speed = compute_r_speed(rollout.block_lens, args.block_len_min, args.block_len_max, device=device)
                # [FIX] Reward Gating: If Quality is too low (<0.5), force Speed Reward to 0
                if False: # r_qual < 0.5:
                    r_speed = torch.tensor(0.0, device=device)
                r_ccd = rollout.entropy_ccd

                rollouts.append(rollout)
                r_qual_list.append(r_qual)
                r_speed_list.append(r_speed)
                r_ccd_list.append(r_ccd)
                r_acc_list.append(r_acc)

            r_qual_tensor = torch.stack(r_qual_list)
            r_speed_tensor = torch.stack(r_speed_list)
            r_ccd_tensor = torch.stack(r_ccd_list)
            r_acc_tensor = torch.stack(r_acc_list)

            if r_qual_tensor.numel() > 1 and r_qual_tensor.std() > 0:
                a_q = (r_qual_tensor - r_qual_tensor.mean()) / (r_qual_tensor.std() + 1e-6)
            else:
                a_q = torch.zeros_like(r_qual_tensor)
            
            if r_speed_tensor.numel() > 1 and r_speed_tensor.std() > 0:
                a_s = (r_speed_tensor - r_speed_tensor.mean()) / (r_speed_tensor.std() + 1e-6)
            else:
                a_s = torch.zeros_like(r_speed_tensor)
                
            if r_ccd_tensor.numel() > 1 and r_ccd_tensor.std() > 0:
                a_ccd = (r_ccd_tensor - r_ccd_tensor.mean()) / (r_ccd_tensor.std() + 1e-6)
            else:
                a_ccd = torch.zeros_like(r_ccd_tensor)

            if r_acc_tensor.numel() > 1 and r_acc_tensor.std() > 0:
                a_acc = (r_acc_tensor - r_acc_tensor.mean()) / (r_acc_tensor.std() + 1e-6)
            else:
                a_acc = torch.zeros_like(r_acc_tensor)
                
            a_q_weighted = args.w1 * a_q
            a_s_weighted = args.w2 * a_s
            a_acc_weighted = args.w_acc * a_acc
            a_ccd_weighted = args.lambda_ccd * a_ccd
            a_total = a_q_weighted + a_s_weighted + a_acc_weighted - a_ccd_weighted
            

            losses = []
            for idx, rollout in enumerate(rollouts):
                new_logprob = torch.tensor(0.0, device=device)
                for (hidden, entropy, guidance_entropy), action in zip(rollout.policy_features, rollout.policy_actions):
                    logits = policy_logits(policy_net, hidden, entropy, guidance_entropy)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = new_logprob + dist.log_prob(torch.tensor(action, device=device))

                loss = ppo_loss(new_logprob, rollout.logprob.detach(), a_total[idx].detach(), args.ppo_clip)
                losses.append(loss)

            loss_total = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if i == 0 or (i + 1) % 1 == 0:
                acc_val = correct_count / total_count if total_count > 0 else 0.0
                print(f"Step {i} | Acc: {acc_val:.2%} ({correct_count}/{total_count}) | a_total: {a_total.mean():.4f} | Loss: {loss_total.item():.4f} | R_qual: {r_qual_tensor.mean():.4f} | R_speed: {r_speed_tensor.mean():.4f} | R_acc: {r_acc_tensor.mean():.4f} | R_CCD: {r_ccd_tensor.mean():.4f} | a_q: {a_q.mean():.4f} | a_s: {a_s.mean():.4f} | a_acc: {a_acc.mean():.4f} | a_ccd: {a_ccd.mean():.4f} | Time: {time.time() - start_time:.2f}s")
            if args.debug:
                print("\n" + "=" * 80)
                print("PROMPT:\n", prompt)
                print("GOLD ANSWER:\n", answer)
                print("\nPOLICY ACTIONS (block_len - 1):")
                for rollout in rollouts:
                    print(rollout.policy_actions)
                print("\nBLOCK LENGTHS:")
                for rollout in rollouts:
                    print(rollout.block_lens)
                print("\nGENERATED TEXTS:")
                for rollout in rollouts:
                    text = tokenizer.decode(rollout.generated_ids[0, input_ids_len(prompt, tokenizer):], skip_special_tokens=True)
                    valid = "CORRECT" if is_correct_smart(text, answer, dataset_type) else "WRONG"
                    print(f"[{valid}] {text}")
                print("=" * 80 + "\n")
            if use_wandb:
                wandb.log({
                    "Train/Accuracy": correct_count / total_count if total_count > 0 else 0.0,
                    "Train/a_total": a_total.mean().item(),
                    "Train/Loss": loss_total.item(),
                    "Train/R_qual": r_qual_tensor.mean().item(),
                    "Train/R_speed": r_speed_tensor.mean().item(),
                    "Train/R_acc": r_acc_tensor.mean().item(),
                    "Train/R_CCD": r_ccd_tensor.mean().item(),
                    "Train/a_q": a_q.mean().item(),
                    "Train/a_s": a_s.mean().item(),
                    "Train/a_acc": a_acc.mean().item(),
                    "Train/a_ccd": a_ccd.mean().item(),
                    "Train/a_q_weighted": a_q_weighted.mean().item(),
                    "Train/a_s_weighted": a_s_weighted.mean().item(),
                    "Train/a_acc_weighted": a_acc_weighted.mean().item(),
                    "Train/a_ccd_weighted": a_ccd_weighted.mean().item(),
                })
            if i == 0 or (i + 1) % 10 == 0:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                torch.save(policy_net.state_dict(), args.save_path)

    if use_wandb:
        wandb.finish()


def input_ids_len(prompt: str, tokenizer) -> int:
    return tokenizer(prompt, return_tensors="pt").input_ids.shape[1]


if __name__ == "__main__":
    main()
