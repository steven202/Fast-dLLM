"""
Adaptive-dLLM GDPO Training with Optuna (Auto-Tuning)

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
import optuna  # [ADDED] Import Optuna
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import set_seed, str2bool

from llada.model.modeling_llada import LLaDAModelLM
from dream.model.modeling_dream import DreamModel
from tqdm import tqdm

# [Global for Optuna to access]
ARGS = None 

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
        prompts = ["Explain the theory of relativity."] 

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


def get_transfer_index(logits: torch.Tensor, mask_index: torch.Tensor, x: torch.Tensor, threshold: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
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


def rollout_llada(model, policy_net, tokenizer, prompt, ar_guidance_model, guidance_gamma, guidance_temperature, gen_length, steps, block_len_min, block_len_max, threshold) -> RolloutResult:
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask_id = 126336
    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 126464 

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
        entropy = -(torch.log_softmax(outputs.logits[:, -1, :], dim=-1).exp() * torch.log_softmax(outputs.logits[:, -1, :], dim=-1)).sum(-1)

        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)
        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        guidance_target = None
        if ar_guidance_model is not None:
            ar_logits = ar_guidance_model(context_ids).logits[:, -1, :]
            if ar_logits.shape[-1] > backbone_vocab_size:
                ar_logits = ar_logits[:, :backbone_vocab_size]
            guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len
        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach()))
        policy_actions.append(block_len - 1)

        i = 0
        while True:
            if i >= steps_per_block: break
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
            if (x[:, block_start:block_end] == mask_id).sum() == 0: break
        current_len += block_len

    entropy_ccd = torch.stack(entropy_accum).mean() if entropy_accum else torch.tensor(0.0, device=device)
    logprob_sum = torch.stack(logprobs).sum()
    return RolloutResult(generated_ids=x, block_lens=block_lens, logprob=logprob_sum, entropy_ccd=entropy_ccd, policy_features=policy_features, policy_actions=policy_actions)


def rollout_dream(model, policy_net, tokenizer, prompt, ar_guidance_model, guidance_gamma, guidance_temperature, gen_length, steps, block_len_min, block_len_max, threshold) -> RolloutResult:
    # (Identical to rollout_llada but with apply_guidance_dream and get_mask_id logic)
    device = model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mask_id = get_mask_id("dream", model)
    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : input_ids.shape[1]] = input_ids

    if hasattr(model.config, "vocab_size"):
        backbone_vocab_size = model.config.vocab_size
    else:
        backbone_vocab_size = 32000

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
        position_ids = torch.arange(context_len, device=device).unsqueeze(0)
        outputs = model(context_ids, position_ids=position_ids, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        entropy = -(torch.log_softmax(outputs.logits[:, -1, :], dim=-1).exp() * torch.log_softmax(outputs.logits[:, -1, :], dim=-1)).sum(-1)

        block_len, logprob = sample_block_len(policy_net, last_hidden, entropy, block_len_min, block_len_max)
        block_len = min(block_len, gen_length - current_len)
        steps_per_block = max(1, int(round(steps_remaining * block_len / max(1, gen_length - current_len))))
        steps_remaining = max(0, steps_remaining - steps_per_block)

        guidance_target = None
        if ar_guidance_model is not None:
            ar_logits = ar_guidance_model(context_ids).logits[:, -1, :]
            if ar_logits.shape[-1] > backbone_vocab_size:
                ar_logits = ar_logits[:, :backbone_vocab_size]
            guidance_target = model.compute_guidance_target(ar_logits, temperature=guidance_temperature)

        block_start = input_ids.shape[1] + current_len
        block_end = block_start + block_len
        block_lens.append(block_len)
        logprobs.append(logprob)
        policy_features.append((last_hidden.detach(), entropy.detach()))
        policy_actions.append(block_len - 1)

        i = 0
        while True:
            if i >= steps_per_block: break
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
            if (x[:, block_start:block_end] == mask_id).sum() == 0: break
        current_len += block_len

    entropy_ccd = torch.stack(entropy_accum).mean() if entropy_accum else torch.tensor(0.0, device=device)
    logprob_sum = torch.stack(logprobs).sum()
    return RolloutResult(generated_ids=x, block_lens=block_lens, logprob=logprob_sum, entropy_ccd=entropy_ccd, policy_features=policy_features, policy_actions=policy_actions)


def compute_reward_nll(reward_model, reward_tokenizer, prompt, generated_text, device) -> torch.Tensor:
    with torch.no_grad():
        if reward_model.device.type != "cuda" and torch.cuda.is_available(): reward_model.to("cuda")
        prompt_ids = reward_tokenizer(prompt, return_tensors="pt").input_ids.to(reward_model.device)
        full_ids = reward_tokenizer(prompt + generated_text, return_tensors="pt").input_ids.to(reward_model.device)
        outputs = reward_model(full_ids)
        logits = outputs.logits[:, :-1, :]
        labels = full_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none")
        loss = loss.view(labels.size())
        nll = loss[:, prompt_ids.size(1) - 1:].mean()
        return nll


def compute_r_speed(block_lens: List[int], min_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    L_avg = sum(block_lens) / max(1, len(block_lens))
    return torch.tensor((L_avg - min_len) / max(1e-6, (max_len - min_len)), device=device, dtype=torch.float32)


def ppo_loss(new_logprob, old_logprob, advantage, clip):
    ratio = torch.exp(new_logprob - old_logprob)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    return -torch.min(ratio * advantage, clipped * advantage).mean()


# [ADDED] Objective function for Optuna
def objective(trial):
    # 1. Suggest Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    w1 = trial.suggest_float("w1", 0.5, 3.0) # Quality Weight
    w2 = trial.suggest_float("w2", 0.1, 1.5) # Speed Weight
    guidance_gamma = trial.suggest_float("guidance_gamma", 0.0, 0.5)
    
    # Pruning for training efficiency
    epochs_for_tuning = 2 # Reduced epochs for faster trials
    prompts_limit = 20 # Train on subset for tuning

    # 2. Setup Resources (Load Models once effectively if possible, here simplified)
    device = torch.device(ARGS.device if torch.cuda.is_available() else "cpu")
    
    # Reload Policy Net to reset weights for each trial
    if ARGS.model_type == "llada":
        # Note: In a real efficient loop, you'd cache the backbone and only reset the policy
        model = LLaDAModelLM.from_pretrained(ARGS.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, trust_remote_code=True)
        policy_net = model.model.block_policy
    else:
        model = DreamModel.from_pretrained(ARGS.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(ARGS.model_path, trust_remote_code=True)
        policy_net = model.block_policy

    model.eval()
    for p in model.parameters(): p.requires_grad = False
    for p in policy_net.parameters(): p.requires_grad = True
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr)

    # Load Helper Models (Guidance & Reward) - Ideally cached outside objective
    ar_guidance_model = AutoModelForCausalLM.from_pretrained(ARGS.ar_guidance_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    reward_model = AutoModelForCausalLM.from_pretrained(ARGS.ar_reward_model, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto", offload_folder=ARGS.reward_offload_dir).eval()
    reward_tokenizer = AutoTokenizer.from_pretrained(ARGS.ar_reward_model, trust_remote_code=True)

    prompts = load_prompts(ARGS.use_gsm8k, ARGS.use_humaneval, max_samples=prompts_limit)
    
    avg_reward = 0
    
    # 3. Training Loop
    for epoch in range(epochs_for_tuning):
        epoch_rewards = []
        for i, prompt in enumerate(prompts):
            rollouts = []
            r_qual_list, r_speed_list, r_ccd_list = [], [], []

            for _ in range(max(1, ARGS.rollouts // 2)): # Reduced rollouts for speed
                if ARGS.model_type == "llada":
                    rollout = rollout_llada(model, policy_net, tokenizer, prompt, ar_guidance_model, guidance_gamma, ARGS.guidance_temperature, ARGS.gen_length, ARGS.steps, ARGS.block_len_min, ARGS.block_len_max, ARGS.threshold)
                else:
                    rollout = rollout_dream(model, policy_net, tokenizer, prompt, ar_guidance_model, guidance_gamma, ARGS.guidance_temperature, ARGS.gen_length, ARGS.steps, ARGS.block_len_min, ARGS.block_len_max, ARGS.threshold)

                gen_text = tokenizer.decode(rollout.generated_ids[0, input_ids_len(prompt, tokenizer):], skip_special_tokens=True)
                nll = compute_reward_nll(reward_model, reward_tokenizer, prompt, gen_text, device=device)
                
                r_qual = torch.exp(-nll)
                r_speed = compute_r_speed(rollout.block_lens, ARGS.block_len_min, ARGS.block_len_max, device=device)
                r_ccd = rollout.entropy_ccd

                rollouts.append(rollout)
                r_qual_list.append(r_qual)
                r_speed_list.append(r_speed)
                r_ccd_list.append(r_ccd)

            r_qual_t = torch.stack(r_qual_list)
            r_speed_t = torch.stack(r_speed_list)
            r_ccd_t = torch.stack(r_ccd_list)

            # Optimization Logic using Trial Parameters
            a_q = (r_qual_t - r_qual_t.mean()) / (r_qual_t.std() + 1e-6)
            a_s = (r_speed_t - r_speed_t.mean()) / (r_speed_t.std() + 1e-6)
            a_total = w1 * a_q + w2 * a_s - ARGS.lambda_ccd * r_ccd_t

            losses = []
            for idx, rollout in enumerate(rollouts):
                new_logprob = torch.tensor(0.0, device=device)
                for (hidden, entropy), action in zip(rollout.policy_features, rollout.policy_actions):
                    logits = policy_logits(policy_net, hidden, entropy)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logprob = new_logprob + dist.log_prob(torch.tensor(action, device=device))
                losses.append(ppo_loss(new_logprob, rollout.logprob.detach(), a_total[idx].detach(), ARGS.ppo_clip))

            loss_total = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # Track metric: We want to maximize Quality + Speed
            current_reward = (w1 * r_qual_t.mean() + w2 * r_speed_t.mean()).item()
            epoch_rewards.append(current_reward)

        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        
        # [ADDED] Reporting and Pruning
        trial.report(avg_reward, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_reward # Optuna maximizes this


def input_ids_len(prompt: str, tokenizer) -> int:
    return tokenizer(prompt, return_tensors="pt").input_ids.shape[1]


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["llada", "dream"], default="llada")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--ar_guidance_model", type=str, default=None)
    parser.add_argument("--ar_reward_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
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
    # Hyperparams below will be overridden by Optuna
    parser.add_argument("--guidance_gamma", type=float, default=0.2)
    parser.add_argument("--guidance_temperature", type=float, default=0.5)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--lambda_ccd", type=float, default=0.1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--wandb", type=str2bool, default=True)
    
    # [ADDED] Optuna args
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    
    ARGS = parser.parse_args()
    set_seed(ARGS.seed)

    if ARGS.ar_guidance_model is None:
        if ARGS.model_type == "llada": ARGS.ar_guidance_model = "meta-llama/Llama-3.2-1B-Instruct"
        elif ARGS.model_type == "dream": ARGS.ar_guidance_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Starting Optuna Study...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ARGS.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()