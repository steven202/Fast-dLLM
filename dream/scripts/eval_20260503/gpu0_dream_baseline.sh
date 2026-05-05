#!/bin/bash
# Dream Baseline — from Fast-dLLM, ALL samples, 5 datasets
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

cd /tmp
FD=/home/user/fast_dllm_h200/mixed_fast_dllm/Fast-dLLM/dream
mkdir -p $FD/evals_results_0503/dream_baseline

BASE="pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=256,diffusion_steps=256,block_length=32,add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True,dtype=bfloat16"

echo "=== Dream Baseline (ALL samples) ==="

for ds in "gsm8k 5" "humaneval 0" "minerva_math500 4" "mbpp 3" "minerva_math 4"; do
  set -- $ds; task=$1; fs=$2; out=$FD/evals_results_0503/dream_baseline/${task}-s256-all
  echo "--- $task (${fs}-shot, ALL samples) ---"
  accelerate launch $FD/eval.py --model dream --model_args ${BASE} \
    --tasks $task --num_fewshot $fs --batch_size 1 \
    --output_path $out --log_samples \
    --confirm_run_unsafe_code 2>&1 | tee /tmp/dream_bl_${task}.log
  cp /tmp/dream_bl_${task}.log $out.log
done

echo "=== Dream Baseline DONE ==="
