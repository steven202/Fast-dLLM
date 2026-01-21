"""
Adaptive-dLLM GDPO Training (Robust, Merged)

Supports both LLaDA and Dream models.
Uses GSM8K + HumanEval prompts for training.
"""

import argparse
import os
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import set_seed

from llada.model.modeling_llada import LLaDAModelLM
from dream.model.modeling_dream import DreamModel


@dataclass
class RolloutResult:
    generated_ids: torch.LongTensor
    block_lens: List[int]
    logprob: torch.Tensor
    entropy_ccd: torch.Tensor
    policy_features: List[Tuple[torch.Tensor, torch.Tensor]]
    policy_actions: List[int]


def load_prompts(use_gsm8k: bool, use_humaneval: bool, max_samples: Optional[int] = None) -> List[str]:
    prompts: List[str] = []

    if use_gsm8k:
        try:
            dataset = load_dataset("gsm8k", "main", split="train")
            for item in dataset:
                prompts.append(item["question"])
        except Exception as e:
            print(f"Warning: Failed to load GSM8K: {e}")

    if use_humaneval:
        try:
            dataset = load_dataset("openai_humaneval", split="test")
            for item in dataset:
                prompts.append(item.get("prompt", ""))
        except Exception as e:
            print(f"Warning: Failed to load HumanEval: {e}")
            
    if not prompts:
        prompts = ["Explain the theory of relativity."] # Fallback

    if max_samples is not None:
        prompts = prompts[:max_samples]

    return prompts


def get_mask_id(model_type: str, model) -> int:
    if model_type == "llada":
        return 126336
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "mask_token_id"):
        return model.generation_config.mask_token_id
    if hasattr(model, "config") and hasattr(model.config, "mask_token_id"):
        return model.config.mask_token_id
    raise ValueError("Unable to infer mask token id.")


def policy_logits(policy_net: torch.nn.Module, hidden: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 3:
        hidden = hidden[:, -1, :]
    entropy = entropy.view(entropy.size(0), 1)
    x = torch.cat([hidden, entropy], dim=-1)
    x = policy_net.act(policy_net.fc1(x))
    return policy_net.fc2(x)


def sample_block_len(policy_net, hidden: torch.Tensor, entropy: torch.Tensor, min_len: int, max_len: int) -> Tuple[int, torch.Tensor]:
    logits = policy_logits(policy_net, hidden, entropy)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    block_len = action.item() + 1
    block_len = int(max(min_len, min(max_len, block_len)))
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

    p = F.softmax(logits.to(torch.float64), dim=-1)
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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask_id = 126336

    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    # --- [MODIFIED] Safe Vocab Access ---
    # LLaDA usually has 126464, while Llama-3 has 128256. 
    # We must know the backbone's limit to slice safely.
    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 126464 # Fallback
    # ------------------------------------

    nfe = 0
    current_len = 0
    steps_remaining = steps
    entropy_accum = []
    block_lens = []
    logprobs = []
    policy_features = []
    policy_actions = []

    while current_len < gen_length:
        context_len = input_ids.shape[1] + current_len
        context_ids = x[:, :context_len]
        outputs = model(context_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)

        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)

        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        guidance_target = None
        if ar_guidance_model is not None:
            # 1. Get raw logits
            ar_logits = ar_guidance_model(context_ids).logits[:, -1, :]
            
            # --- [MODIFIED] Safe Logit Slicing ---
            # If guidance has more tokens than backbone (e.g. 128k vs 126k), slice it.
            if ar_logits.shape[-1] > backbone_vocab_size:
                ar_logits = ar_logits[:, :backbone_vocab_size]
            # -------------------------------------

            guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len

        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach()))
        policy_actions.append(block_len - 1)

        i = 0
        while True:
            if i >= steps_per_block:
                break
            mask_index = (x == mask_id)
            mask_index[:, block_end:] = 0

            outputs = model(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            hidden_states = apply_guidance_llada(model, hidden_states, guidance_target, guidance_gamma)

            if model.model.config.weight_tying:
                logits = F.linear(hidden_states, model.model.transformer.wte.weight, None)
            else:
                logits = model.model.transformer.ff_out(hidden_states)

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask_id = get_mask_id("dream", model)

    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    current_len = 0
    steps_remaining = steps
    entropy_accum = []
    block_lens = []
    logprobs = []
    policy_features = []
    policy_actions = []

    # --- [MODIFIED] Safe Vocab Access for Dream ---
    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 32000 
    # ----------------------------------------------

    while current_len < gen_length:
        context_len = input_ids.shape[1] + current_len
        context_ids = x[:, :context_len]
        position_ids = torch.arange(context_len, device=device).unsqueeze(0)
        outputs = model(context_ids, position_ids=position_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        last_logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(last_logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)

        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)

        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        guidance_target = None
        if ar_guidance_model is not None:
            ar_logits = ar_guidance_model(context_ids).logits[:, -1, :]
            
            # --- [MODIFIED] Safe Logit Slicing ---
            if ar_logits.shape[-1] > backbone_vocab_size:
                ar_logits = ar_logits[:, :backbone_vocab_size]
            # -------------------------------------

            guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len

        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach()))
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
    )


def compute_reward_nll(
    reward_model,
    reward_tokenizer,
    prompt: str,
    generated_text: str,
    device: str,
) -> torch.Tensor:
    # Use torch.no_grad for efficiency
    with torch.no_grad():
        # Ensure model is on correct device
        if reward_model.device.type != "cuda" and torch.cuda.is_available():
             reward_model.to("cuda")
        
        prompt_ids = reward_tokenizer(prompt, return_tensors="pt").input_ids.to(reward_model.device)
        full_ids = reward_tokenizer(prompt + generated_text, return_tensors="pt").input_ids.to(reward_model.device)

        outputs = reward_model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        loss = loss.view(labels.size())

        prompt_len = prompt_ids.size(1) - 1
        nll = loss[:, prompt_len:].mean()

        return nll


def compute_r_speed(block_lens: List[int], min_len: int, max_len: int) -> torch.Tensor:
    L_avg = sum(block_lens) / max(1, len(block_lens))
    return torch.tensor((L_avg - min_len) / max(1e-6, (max_len - min_len)))


def ppo_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    ratio = torch.exp(new_logprob - old_logprob)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    return -torch.min(ratio * advantage, clipped * advantage).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["llada", "dream"], default="llada")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    
    # [MODIFIED] Default is None so we can set it dynamically based on model_type
    parser.add_argument("--ar_guidance_model", type=str, default=None)
    
    parser.add_argument("--ar_reward_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--reward_offload_dir", type=str, default="./offload_reward")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_gsm8k", action="store_true")
    parser.add_argument("--use_humaneval", action="store_true")
    parser.add_argument("--max_samples", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_len_min", type=int, default=8)
    parser.add_argument("--block_len_max", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)

    parser.add_argument("--guidance_gamma", type=float, default=0.2)
    parser.add_argument("--guidance_temperature", type=float, default=0.5)

    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--lambda_ccd", type=float, default=0.1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # --- [MODIFIED] Automatic Guidance Model Selection ---
    if args.ar_guidance_model is None:
        if args.model_type == "llada":
            args.ar_guidance_model = "meta-llama/Llama-3.2-1B-Instruct"
            print(f"✅ Selected Guidance for LLaDA: {args.ar_guidance_model}")
        elif args.model_type == "dream":
            args.ar_guidance_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"✅ Selected Guidance for Dream: {args.ar_guidance_model}")
    # ---------------------------------------------------

    if args.save_path is None:
        args.save_path = f"./checkpoints/policy_{args.model_type}.pt"
    if args.load_path is None:
        args.load_path = f"./checkpoints/policy_{args.model_type}.pt"
    if not args.use_gsm8k and not args.use_humaneval:
        args.use_gsm8k = True
        args.use_humaneval = True

    prompts = load_prompts(args.use_gsm8k, args.use_humaneval, max_samples=args.max_samples)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model_type == "llada":
        model = LLaDAModelLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        policy_net = model.model.block_policy
    else:
        model = DreamModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        policy_net = model.block_policy

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in policy_net.parameters():
        p.requires_grad = True

    if args.load_path and os.path.exists(args.load_path):
        policy_net.load_state_dict(torch.load(args.load_path, map_location=device))

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=args.lr)

    ar_guidance_model = AutoModelForCausalLM.from_pretrained(
        args.ar_guidance_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    ar_guidance_model.eval()

    os.makedirs(args.reward_offload_dir, exist_ok=True)
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.ar_reward_model,
        torch_dtype=torch.float16,
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
            wandb.init(project="adaptive-dllm", config=vars(args))
        except Exception:
            use_wandb = False

    for epoch in range(args.epochs):
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            rollouts: List[RolloutResult] = []
            r_qual_list = []
            r_speed_list = []
            r_ccd_list = []

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
                nll = compute_reward_nll(reward_model, reward_tokenizer, prompt, gen_text, device=device)
                r_qual = torch.exp(-nll)
                r_speed = compute_r_speed(rollout.block_lens, args.block_len_min, args.block_len_max)
                r_ccd = rollout.entropy_ccd

                rollouts.append(rollout)
                r_qual_list.append(r_qual)
                r_speed_list.append(r_speed)
                r_ccd_list.append(r_ccd)

            r_qual_tensor = torch.stack(r_qual_list)
            r_speed_tensor = torch.stack(r_speed_list)
            r_ccd_tensor = torch.stack(r_ccd_list)

            a_q = (r_qual_tensor - r_qual_tensor.mean()) / (r_qual_tensor.std() + 1e-6)
            a_s = (r_speed_tensor - r_speed_tensor.mean()) / (r_speed_tensor.std() + 1e-6)
            a_total = args.w1 * a_q + args.w2 * a_s - args.lambda_ccd * r_ccd_tensor

            losses = []
            for idx, rollout in enumerate(rollouts):
                new_logprob = torch.tensor(0.0, device=device)
                for (hidden, entropy), action in zip(rollout.policy_features, rollout.policy_actions):
                    logits = policy_logits(policy_net, hidden, entropy)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = new_logprob + dist.log_prob(torch.tensor(action, device=device))

                loss = ppo_loss(new_logprob, rollout.logprob.detach(), a_total[idx].detach(), args.ppo_clip)
                losses.append(loss)

            loss_total = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            if i % 1 == 0:
                print(f"Step {i} | Loss: {loss_total.item():.4f} | R_qual: {r_qual_tensor.mean():.4f} | Time: {time.time() - start_time:.2f}s")

            if use_wandb:
                wandb.log({
                    "Train/R_qual": r_qual_tensor.mean().item(),
                    "Train/R_speed": r_speed_tensor.mean().item(),
                    "Train/R_CCD": r_ccd_tensor.mean().item(),
                    "Train/Loss": loss_total.item(),
                })

        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(policy_net.state_dict(), args.save_path)

    if use_wandb:
        wandb.finish()


def input_ids_len(prompt: str, tokenizer) -> int:
    return tokenizer(prompt, return_tensors="pt").input_ids.shape[1]


if __name__ == "__main__":
    main()