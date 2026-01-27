
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
model="Dream-org/Dream-v0-Base-7B"
policy_path='./checkpoints/policy_dream.pt'
policy_name=$(basename "$policy_path" .pt)


# 1. Baseline
echo "Running Baseline..."
run_eval "dream_gsm8k_baseline" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},threshold=0.9,show_speed=True 

# 2. Prefix Cache
echo "Running Prefix Cache..."
run_eval "dream_gsm8k_prefix_cache" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,show_speed=True 

# 3. Parallel
echo "Running Parallel..."
run_eval "dream_gsm8k_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},threshold=0.9,show_speed=True 

# 4. Prefix Cache + Parallel
echo "Running Prefix Cache + Parallel..."
run_eval "dream_gsm8k_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,threshold=0.9,show_speed=True 

# 5. Dual Cache + Parallel
echo "Running Dual Cache + Parallel..."
run_eval "dream_gsm8k_dual_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True
