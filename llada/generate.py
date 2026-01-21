# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

from torch.cuda import nvtx

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)

#     base = mask_num // steps
#     remainder = mask_num % steps

#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1

#     return num_transfer_tokens

def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    block_mask_index: (B, L) bool – which positions are masked in the current block
    returns: (B, steps) int – how many tokens to transfer at each step per batch item
    """
    device = block_mask_index.device
    dtype = torch.long

    total = block_mask_index.sum(dim=1)                  # (B,)
    base  = torch.div(total, steps, rounding_mode='floor')  # (B,)
    rem   = total - base * steps                         # (B,)

    # Start with base for all steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype)  # (B, steps)

    # Add +1 to the first `rem[b]` steps for each batch b — without tensor slicing
    cols = torch.arange(steps, device=device).unsqueeze(0)               # (1, steps)
    add_mask = cols < rem.unsqueeze(1)                                   # (B, steps)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)       # (B, steps)

    return num_transfer_tokens



@torch.no_grad()
def generate_adaptive(model, prompt, total_len=128, steps=None, ar_guidance_model=None, temperature=0.,
                      remasking='low_confidence', mask_id=126336, threshold=None, factor=None, guidance_gamma=0.2):
    '''
    Args:
        model: Mask predictor (LLaDA).
        prompt: A tensor of shape (B, L).
        total_len: Maximum length of generated answer.
        steps: Total steps budget (used to scale steps per block).
        ar_guidance_model: Small AR model for Phase 3 guidance.
    '''
    # print("Generating with Adaptive-dLLM (Refactored Loop)...")
    device = model.device
    B = prompt.shape[0]
    
    # x contains the committed tokens (initially prompt)
    x = prompt.clone()
    
    current_len = 0
    nfe = 0
    
    while current_len < total_len:
        # Re-verify 'x' shape
        L_context = x.shape[1]
        
        # 1. Policy Prediction
        outputs = model(x, output_hidden_states=True)
        # hidden_states: tuple of (B, L, H)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] # (B, H)
        
        # Entropy
        last_logits = outputs.logits[:, -1, :]
        last_probs = torch.softmax(last_logits, dim=-1)
        entropy = -torch.sum(last_probs * torch.log(last_probs + 1e-9), dim=-1, keepdim=True)
        
        # Predict block length
        if hasattr(model, 'policy_net'):
            block_len = model.policy_net.predict(last_hidden_state, entropy).item()
        else:
            block_len = 32 # Default fallback
        
        # 2. Safety Clip
        block_len = min(block_len, total_len - current_len)
        if block_len <= 0:
            break
            
        # 3. Pre-compute Guidance Target (One-Pass Optimization)
        guidance_target = None
        if ar_guidance_model is not None:
            # Run AR ONCE on the current context 'x'
            ar_logits = ar_guidance_model(x) # [B, L_context, V]
            # We specifically want targets for the FUTURE tokens. 
            # AR model predicts next tokens. 
            # ar_logits[:, -1, :] predicts the first new token.
            # But wait, AR model predicts *one* step ahead. 
            # We have a block of length > 1.
            # We can only guide the FIRST token with 1 running? Or guide with greedy/sampled rollouts?
            # 
            # "The AR model must run only ONCE per block (outside the diffusion loop) to pre-compute the guidance target."
            # If we run it once, we get logits for the *next* token (L_context).
            # For a block of size block_len, do we have targets for all?
            # 
            # Constraint check: "AR model must run only ONCE per block".
            # This implies we might only guide the *first* token, OR the AR model runs efficiently (KV cache) to project `block_len` tokens?
            # But AR is sequential. You can't get 2nd token target without picking 1st.
            # 
            # If we strictly follow "ONCE per block", we effectively only have the target for the immediate next token.
            # OR, we assume the AR model output for `current_context_ids` effectively gives us a "plan".
            # 
            # Actually, if we just use `last_token_logits`, we guide the first token. 
            # If we want to guide the whole block, we'd need to roll out AR.
            # 
            # Let's assume we use the AR prediction for the *masked positions*. 
            # But AR cannot predict position t+2 without t+1.
            # 
            # "Taylor Guidance uses Temperature-scaled Expected Embedding".
            # 
            # Let's assume `guidance_target` is computed for the prompt, which corresponds to the first new token.
            # For the rest of the block, we either have no guidance or recycled guidance?
            # 
            # Simplest interpretation: We have `guidance_target` of shape [B, 1, H] (or [B, H]).
            # We will apply it to the first masked token (or all? but that's wrong).
            # Let's apply to the *whole block* but broadcasting? No, that pushes all tokens to be the same.
            # 
            # Compromise: We guide ONLY the tokens for which we have targets.
            # Since AR runs once, we have target for x[L_context].
            # We will generate guidance_target for this position.
            
            # Using logits for the last position in x to predict x[L_context]
            # ar_logits shape: (B, L, V). Last one predicts L+1.
            
            ar_logits_last = ar_logits.logits[:, -1:, :] if hasattr(ar_logits, 'logits') else ar_logits[:, -1:, :]
            
            if hasattr(model, 'compute_guidance_target'):
                guidance_target = model.compute_guidance_target(ar_logits_last) # (B, 1, H)
            else:
                guidance_target = None # Should not happen if models updated

        # 4. Prepare Exact Position IDs
        # Relative to the start of the block
        position_ids = torch.arange(L_context, L_context + block_len, device=device).unsqueeze(0).expand(B, -1)
        
        # 5. Run Diffusion
        # Prepare input for diffusion: [Context, MASK * block_len]
        mask_tokens = torch.full((B, block_len), mask_id, dtype=torch.long, device=device)
        x_block = torch.cat([x, mask_tokens], dim=1)
        
        # Diffusion Loop for this block
        steps_this_block = block_len 
        if steps is not None:
             steps_this_block = max(1, int(steps * (block_len / total_len)))
        
        current_block_start = L_context
        current_block_end = L_context + block_len
        
        x_curr = x_block.clone()
        
        # Schedule
        block_mask_indicator = torch.ones((B, block_len), dtype=torch.bool, device=device)
        num_transfer_tokens_schedule = get_num_transfer_tokens(block_mask_indicator, steps_this_block)
        
        for t in range(steps_this_block):
            params_mask = (x_curr[:, current_block_start:] == mask_id)
            if params_mask.sum() == 0:
                break
                
            nfe += 1
            
            # Forward pass with Hidden States
            model_out = model(x_curr, output_hidden_states=True)
            logits = model_out.logits
            hidden_states = model_out.hidden_states[-1] # (B, L_total, H)
            
            # --- GUIDANCE INJECTION ---
            if guidance_target is not None:
                # Target: expectation of AR for the *first* new token.
                # Corresponds to hidden_state at 'current_block_start'.
                # G = gamma * (target - z_t)
                # z_t = hidden_states[:, current_block_start, :]
                
                z_t = hidden_states[:, current_block_start:current_block_start+1, :]
                
                # Careful: guidance_target is (B, 1, H). z_t is (B, 1, H).
                G = guidance_gamma * (guidance_target - z_t)
                
                # Update hidden state?
                # We can't update hidden state inside the model easily (it's already computed).
                # We update the *effect* of hidden state, i.e., logits.
                # Logits = (h + G) @ W_out.T
                #        = h @ W_out.T + G @ W_out.T
                #        = logits + G @ W_out.T
                
                # Need access to readout head.
                # LLaDA: model.model.transformer.ff_out (if untied) or wte (if tied).
                # Checking `model.model.transformer.ff_out` availability.
                # `LLaDAModelLM.get_output_embeddings()`
                lm_head = model.get_output_embeddings()
                
                # Compute adjustment to logits
                # Using G as "noise_pred" adjustment or directly modifying logits equivalent to shifting embedding.
                # The user instruction: "Update Noise: noise_pred = noise_pred - G"
                # If we assume `z_t` is the predicted embedding we want to shift TOWARDS `guidance_target`.
                # Then we WANT `z_final = z_t + gamma*(target - z_t)`.
                # So we ADD `G` to `z_t`.
                # `noise_pred` subtraction is for *denoising*.
                # If equation is `x_{t-1} = x_t - noise_pred`.
                # Then `noise_pred_new = noise_pred - G` means `x_{t-1} = x_t - noise_pred + G`.
                # So we are adding `G`.
                # So we simply Add `G` to hidden states, then re-project.
                
                guidance_term_logits = lm_head(G)
                
                # Apply only to the relevant token (first token of block)
                # logits is (B, L, V).
                # We add to logits[:, current_block_start, :]
                
                logits[:, current_block_start:current_block_start+1, :] += guidance_term_logits
            
            # --- END GUIDANCE ---
            
            mask_index_full = (x_curr == mask_id)
            # Only touch the current block
            mask_index_full[:, :current_block_start] = False
            mask_index_full[:, current_block_end:] = False 
            
            quota = num_transfer_tokens_schedule[:, t] if threshold is None else None
            
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index_full, x_curr, quota, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index_full, x_curr, None, factor)
            
            x_curr[transfer_index] = x0[transfer_index]
            
        # 6. Update
        x = x_curr 
        current_len += block_len
        
    return x, nfe

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe

@torch.no_grad()
def generate_with_dual_cache(
    model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
    remasking="low_confidence", mask_id=126336, threshold=None, factor=None
):
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])  # Python int, not Tensor
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    # x: (B, Lp + gen_length)
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        # Masks/indices for the current block
        block_mask_index = (x[:, s:e] == mask_id)  # (B, block_length)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # (B, steps_per_block)

        # 1) Warm KV-cache on the full prefix once per block
        out_full = model(x, use_cache=True)
        past_key_values = out_full.past_key_values
        nfe += 1

        # Build a replace_position tensor indicating the block range (static slice)
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True  # boolean mask (not a dynamic slice bound)

        # Step 0: do an initial transfer on the full logits
        global_mask_index = (x == mask_id)
        # Do not touch beyond current block in this phase
        global_mask_index[:, e:] = False

        if factor is None:
            quota0 = None if threshold is not None else num_transfer_tokens[:, 0]  # (B,)
            x0, transfer_index = get_transfer_index(
                out_full.logits, temperature, remasking, global_mask_index, x, quota0, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out_full.logits, temperature, remasking, global_mask_index, x, None, factor
            )

        # In-place update via torch.where (no tensor-slice assignment with mask)
        x = torch.where(transfer_index, x0, x)

        # 2) Semi-autoregressive refinement, fixed number of steps (graph-friendly)
        #    Each iteration runs on the current block with KV-cache and replace_position
        for i in range(1, steps_per_block):
            # Evaluate logits only for current block with cache
            if (x[:, s:e] == mask_id).sum() == 0:
                break
            logits_blk = model(
                x[:, s:e], past_key_values=past_key_values, use_cache=True, replace_position=replace_position
            ).logits  # shape expected by get_transfer_index*

            # Mask and quota for this step (all tensor ops)
            mask_blk = (x[:, s:e] == mask_id)  # (B, block_length)

            if factor is None:
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]  # (B,)
                x0_blk, transfer_idx_blk = get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold
                )
            else:
                x0_blk, transfer_idx_blk = get_transfer_index_dynamic(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], None, factor
                )

            # Merge back into x[:, s:e] using torch.where (no masked slice assignment)
            blk_old = x[:, s:e]
            blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)  # static concatenation

            nfe += 1

    return x, nfe



def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,   # (B, L) bool
    x: torch.Tensor,            # (B, L) long
    num_transfer_tokens,        # (B,) or (B,1) long tensor, or None when threshold is used
    threshold: float = None,
):
    """
    Returns:
        x0: (B, L) long — proposed tokens
        transfer_index: (B, L) bool — which positions to update this step
    """
    # 1) Sample proposal x0
    # Gumbel-noise for exploration; if temperature==0, add_gumbel_noise should no-op
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L), long

    # 2) Confidence for chosen tokens (or random)
    if remasking == "low_confidence":
        # Use higher precision for softmax stability
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (B, L), float64
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)  # (B, L)
    else:
        raise NotImplementedError(remasking)

    # Only modify masked spots; keep others as original x and set their confidence to -inf
    x0 = torch.where(mask_index, x0, x)

    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)  # (B, L)

    # 3) Pick positions to transfer (vectorized)
    if threshold is not None:
        # Transfer all masked positions whose confidence >= threshold
        # (No top-k; purely threshold-based)
        transfer_index = mask_index & (confidence >= threshold)

        # at least one token is transferred "always unmask max c^i"
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True) # (B, 1)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)

        # (Above Threshold) OR (Is Max Confidence)
        transfer_index = transfer_index | force_mask

        # Safety: do not unmask something that was not masked (consider fully unmasked rows)
        transfer_index = transfer_index & mask_index

        return x0, transfer_index

    # Else: per-row top-k with varying k (num_transfer_tokens), fully batched
    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be a tensor when threshold is None.")

    # Ensure shape (B,) long
    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    # Sort confidences descending (masked positions are valid; others are -inf)
    # idx: (B, L) gives positions in original sequence sorted by confidence
    values, idx = torch.sort(confidence, dim=1, descending=True)

    B, L = confidence.shape
    # Build a mask that is True for the first k[b] columns in each row (sorted order)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)   # (B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)                   # (B, L)
    select_sorted = cols < k_expanded                                            # (B, L) bool

    # Scatter the sorted True/False back to original column order
    # Use integer scatter then cast to bool (scatter_ on bool can be finicky across versions)
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8) # (B, L)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index  # ensure we never select unmasked

    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        num_tokens = int(num_transfer_tokens[j].item())
        if num_tokens == 0:
            continue
        
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    # model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    with torch.inference_mode():
        nvtx.range_push("INFER")

        out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    
        torch.cuda.synchronize()
        nvtx.range_pop()
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
