#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# export CUDA_VISIBLE_DEVICES=2


LOG_DIR="./log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Handle Ctrl-C
cleanup() {
	echo -e "\nInterrupted (Ctrl-C). Exiting..."
	exit 130
}
trap cleanup SIGINT SIGTERM

run_eval() {
	local name="$1"
	shift
	local LOG_FILE="$LOG_DIR/eval_${name}_${policy_name}_${threshold_impl}_${TIMESTAMP}.log"
	echo "Logging to $LOG_FILE"
	"$@" 2>&1 | tee "$LOG_FILE"
}

task=gsm8k
length=256
block_len_min=8
block_len_max=64
num_fewshot=5
steps=256
factor=1.0
threshold_impl="llada"
model_path='GSAI-ML/LLaDA-8B-Instruct'

policy_path="./checkpoints/policy_llada_gsm8k_Qwen_Qwen3-0.6B_Qwen_Qwen3-8B_static_20260125_231016.pt"

policy_name=$(basename "$policy_path" .pt)
echo "Running llada gsm8k qwen3-0.6b qwen3-8b static"
run_eval "llada_gsm8k_qwen3-0.6b_qwen3-8b_static" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,dual_cache=True,factor=${factor},threshold=0.9,threshold_impl=${threshold_impl},gumbel_temperature=0.0,show_speed=True,show_actions=True,wandb=True
