
#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

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
	local LOG_FILE="$LOG_DIR/eval_${name}_${policy_name}_${TIMESTAMP}.log"
	echo "Logging to $LOG_FILE"
	"$@" 2>&1 | tee "$LOG_FILE"
}

task=gsm8k
length=256
block_len_min=8
block_len_max=64
num_fewshot=5
steps=128
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
policy_path='./checkpoints/policy_llada.pt'
policy_name=$(basename "$policy_path" .pt)


# # 1. Baseline (Adaptive with full steps allowance)
# echo "Running Baseline (steps=${length})..."
# run_eval "llada_gsm8k_baseline" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},threshold=0.9,show_speed=True 


# # 2. Prefix cache (Note: Adaptive Policy currently warns about cache support)
# echo "Running Prefix Cache..."
# run_eval "llada_gsm8k_prefix_cache" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,show_speed=True 


# # 3. Parallel (Constrained steps - forces policy to be aggressive)
# echo "Running Parallel (steps=${steps})..."
# run_eval "llada_gsm8k_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},threshold=0.9,show_speed=True

# # 4. Parallel factor
# echo "Running Parallel Factor..."
# run_eval "llada_gsm8k_parallel_factor" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},factor=${factor},threshold=0.9,show_speed=True


# # 5. Prefix cache+parallel
# echo "Running Prefix Cache + Parallel..."
# run_eval "llada_gsm8k_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,threshold=0.9,show_speed=True

# # 6. Dual cache+parallel
# echo "Running Dual Cache + Parallel..."
# run_eval "llada_gsm8k_dual_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

# # 7. Prefix cache+parallel factor
# echo "Running Prefix Cache + Parallel Factor..."
# run_eval "llada_gsm8k_cache_parallel_factor" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --model adaptive_llada \
# --model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,factor=${factor},threshold=0.9,show_speed=True

# 8. Dual cache+parallel factor
echo "Running Dual Cache + Parallel Factor..."
run_eval "llada_gsm8k_dual_cache_parallel_factor" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,dual_cache=True,factor=${factor},threshold=0.9,show_speed=True,wandb=True
