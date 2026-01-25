
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

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"
policy_path='./checkpoints/policy_dream.pt'
policy_name=$(basename "$policy_path" .pt)


# 1. Baseline
echo "Running Baseline..."
run_eval "dream_humaneval_baseline" accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},show_speed=True \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# 2. Prefix Cache
echo "Running Prefix Cache..."
run_eval "dream_humaneval_prefix_cache" accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,show_speed=True \
--output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# 3. Parallel
echo "Running Parallel..."
run_eval "dream_humaneval_parallel" accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},threshold=0.9,show_speed=True \
--output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

# 4. Prefix Cache + Parallel
echo "Running Prefix Cache + Parallel..."
run_eval "dream_humaneval_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

# 5. Dual Cache + Parallel
echo "Running Dual Cache + Parallel..."
run_eval "dream_humaneval_dual_cache_parallel" accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
echo "Run postprocess (example for baseline)..."
# python dream/postprocess_code.py {the samples_xxx.jsonl file under output_path}
