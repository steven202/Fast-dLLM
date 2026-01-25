#!/usr/bin/env bash
# set -e
set -euo pipefail

LOG_DIR="./log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
# Training parameters (must match train_adaptive_policy.py defaults or your custom values)
model_type="llada"
datasets="gsm8k"
ar_reward_model="Qwen_Qwen3-8B"
# ar_reward_model="mobile140m"
aligner_type="static"

# Create descriptive log filename
LOG_FILE="$LOG_DIR/train_${model_type}_${datasets}_${ar_reward_model}_${aligner_type}_${TIMESTAMP}.log"
echo "Logging to $LOG_FILE"

# Handle Ctrl-C
cleanup() {
    echo -e "\nInterrupted (Ctrl-C). Exiting..."
    exit 130
}
trap cleanup SIGINT SIGTERM

# Run command with stdout + stderr logged
CUDA_LAUNCH_BLOCKING=1 PYTHONUNBUFFERED=1 python train_adaptive_policy.py \
--aligner_type "${aligner_type}" \
2>&1 | tee "$LOG_FILE"