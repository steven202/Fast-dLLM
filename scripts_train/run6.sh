#!/usr/bin/env bash
# set -e
set -euo pipefail

LOG_DIR="./log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"


# parser.add_argument("--ar_guidance_model", type=str, choices=["meta-llama/Llama-3.2-1B-Instruct", "facebook/MobileLLM-R1.5-140M", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen3-0.6B", "Alibaba-Apsara/DASD-4B-Thinking"], default="Qwen/Qwen3-0.6B")
# parser.add_argument("--ar_reward_model", type=str, choices=["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-8B", "Alibaba-Apsara/DASD-4B-Thinking"], default="Qwen/Qwen3-8B")
# parser.add_argument("--max_samples", type=int, default=2000)
# Training parameters (must match train_adaptive_policy.py defaults or your custom values)
model_type="dream"
datasets="gsm8k"
# Use '_' instead of '/' in these variables (so bash/log filenames don't create directories).
# The script will convert '_' -> '/' when passing them to Python/HF.
ar_reward_model_raw="Qwen_Qwen3-8B"
ar_guidance_model_raw="TinyLlama_TinyLlama-1.1B-Chat-v1.0"
threshold_impl="dream"
model_path="Dream-org/Dream-v0-Base-7B"
steps=8
use_fewshot=False
# Convert underscore-form -> HuggingFace repo id form (e.g., Qwen_Qwen3-8B -> Qwen/Qwen3-8B)
# Rules:
# - If the value already contains '/', keep it as-is.
# - Otherwise, convert ONLY the first '_' into '/'.
to_hf_repo_id() {
    local raw="$1"
    if [[ "$raw" == *"/"* ]]; then
        echo "$raw"
        return 0
    fi
    if [[ "$raw" == *"_"* ]]; then
        echo "${raw/_/\/}"
        return 0
    fi
    echo "$raw"
}

ar_reward_model="$(to_hf_repo_id "$ar_reward_model_raw")"
ar_guidance_model="$(to_hf_repo_id "$ar_guidance_model_raw")"
block_len_max=32
block_len_min=8
# ar_reward_model="mobile140m"
aligner_type="static"
max_samples=3000
gen_length=256
wandb_hist_every=10
# Create descriptive log filename
LOG_FILE="$LOG_DIR/train_${model_type}_${datasets}_${ar_reward_model_raw}_${ar_guidance_model_raw}_${aligner_type}_${threshold_impl}_${max_samples}_S${steps}_fewshot${use_fewshot}_${TIMESTAMP}.log"
echo "Logging to $LOG_FILE"

# Handle Ctrl-C
cleanup() {
    echo -e "\nInterrupted (Ctrl-C). Exiting..."
    exit 130
}
trap cleanup SIGINT SIGTERM

# Run command with stdout + stderr logged
CUDA_LAUNCH_BLOCKING=1 python train_adaptive_policy.py \
--aligner_type "${aligner_type}" \
--block_len_min "${block_len_min}" \
--block_len_max "${block_len_max}" \
--datasets "${datasets}" \
--model_type "${model_type}" \
--model_path "${model_path}" \
--ar_guidance_model "${ar_guidance_model}" \
--ar_reward_model "${ar_reward_model}" \
--wandb_hist_every "${wandb_hist_every}" \
--max_samples "${max_samples}" \
--threshold_impl "${threshold_impl}" \
--steps "${steps}" \
--use_fewshot "${use_fewshot}" 2>&1 | tee "$LOG_FILE"
# --use_cache True \
# --dual_cache True 2>&1 | tee "$LOG_FILE"
# No! 0.58