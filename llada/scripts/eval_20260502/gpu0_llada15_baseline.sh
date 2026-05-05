#!/bin/bash
# LLaDA-1.5 Baseline — from Fast-dLLM (original code), n=200
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

cd /tmp
FD=/home/user/fast_dllm_h200/mixed_fast_dllm/Fast-dLLM/llada
mkdir -p $FD/evals_results_0502/llada15_baseline

BASE="model_path=GSAI-ML/LLaDA-1.5,gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True"

echo "=== LLaDA-1.5 Baseline (Fast-dLLM) ==="

for ds in "gsm8k 5 200" "humaneval 0 164" "minerva_math500 4 200" "mbpp 3 200"; do
  set -- $ds; task=$1; fs=$2; lim=$3; out=$FD/evals_results_0502/llada15_baseline/${task}-s256-n200
  echo "--- $task (${fs}-shot, limit=$lim) ---"
  accelerate launch $FD/eval_llada.py --tasks $task --num_fewshot $fs --model llada_dist \
    --model_args ${BASE} --batch_size 1 --output_path $out --log_samples --limit $lim \
    --confirm_run_unsafe_code 2>&1 | tee -a $out.log
done

echo "=== DONE ==="
