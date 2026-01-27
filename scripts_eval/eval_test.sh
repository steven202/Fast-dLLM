
#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# export CUDA_VISIBLE_DEVICES=0

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
num_fewshot=5
# Sweep configs: evaluate the SAME policy under multiple (block_len_max, steps) pairs.
# Rationale:
# - Larger block_len_max generally needs more denoising steps.
# - (64, 128) can be under-denoised; (64, 256) is a safer pairing.
# Format: "block_len_max:steps".
EVAL_CONFIGS=(
	"32:128"
	"64:128"
	"64:256"
)
factor=1.0
threshold_impl="llada"
model_path='GSAI-ML/LLaDA-8B-Instruct'

eval_one() {
	local label="$1"
	local policy_path="$2"
	local pair="$3"

	local block_len_max="${pair%%:*}"
	local steps="${pair##*:}"

	policy_name=$(basename "$policy_path" .pt)
	local run_name="${label}_B${block_len_max}_S${steps}"
	echo "Running ${run_name}"

	run_eval "${run_name}" accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
		--model adaptive_llada \
		--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_min=${block_len_min},block_len_max=${block_len_max},use_cache=True,dual_cache=True,factor=${factor},threshold=0.9,threshold_impl=${threshold_impl},gumbel_temperature=0.0,show_speed=True,show_actions=True,wandb=True
}

policy_path="./checkpoints/policy_llada_gsm8k_facebook_MobileLLM-R1.5-140M_Qwen_Qwen3-8B_static_20260125_142549.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_mobile140m_qwen3-8b_static" "$policy_path" "$pair"
done

policy_path="./checkpoints/policy_llada_gsm8k_Qwen_Qwen3-0.6B_Qwen_Qwen3-8B_cached_20260125_154118.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_qwen3-0.6b_qwen3-8b_cached" "$policy_path" "$pair"
done

policy_path="./checkpoints/policy_llada_gsm8k_Qwen_Qwen3-0.6B_Alibaba-Apsara_DASD-4B-Thinking_static_20260125_153711.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_qwen3-0.6b_dasd-4b_static" "$policy_path" "$pair"
done

policy_path="./checkpoints/policy_llada_gsm8k_Qwen_Qwen3-0.6B_Qwen_Qwen3-8B.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_qwen3-0.6b_qwen3-8b" "$policy_path" "$pair"
done

policy_path="./checkpoints/policy_llada_gsm8k_Qwen_Qwen3-0.6B_Qwen_Qwen3-8B_static_20260125_231016.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_qwen3-0.6b_qwen3-8b_static" "$policy_path" "$pair"
done

policy_path="./checkpoints/policy_llada_gsm8k_meta-llama_Llama-3.2-1B-Instruct_Qwen_Qwen3-8B_static_20260126_002854.pt"
for pair in "${EVAL_CONFIGS[@]}"; do
	eval_one "llada_gsm8k_llama-3.2-1b_qwen3-8b_static" "$policy_path" "$pair"
done