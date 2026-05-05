#!/bin/bash
# GPU 7: LLaDA-8B Baseline — from Fast-dLLM, ALL samples
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=true NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1

cd /tmp
FD=/home/user/fast_dllm_h200/mixed_fast_dllm/Fast-dLLM/llada
mkdir -p $FD/evals_results_0503/llada8b_baseline

BASE="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,use_cache=True,dual_cache=True,threshold=0.9,show_speed=True"

echo "=== GPU 7: LLaDA-8B Baseline (ALL samples) ==="

for ds in "gsm8k 5" "humaneval 0" "minerva_math500 4" "mbpp 3" "minerva_math 4"; do
  set -- $ds; task=$1; fs=$2; out=$FD/evals_results_0503/llada8b_baseline/${task}-s256-all
  echo "--- $task (${fs}-shot, ALL samples) ---"
  accelerate launch $FD/eval_llada.py --tasks $task --num_fewshot $fs --model llada_dist \
    --model_args ${BASE} --batch_size 1 --output_path $out --log_samples \
    --confirm_run_unsafe_code 2>&1 | tee /tmp/gpu7_${task}.log
  cp /tmp/gpu7_${task}.log $out.log
done

echo "=== GPU 7: DONE ==="
