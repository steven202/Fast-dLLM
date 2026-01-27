#!/bin/bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
DATE=$(date +%Y%m%d_%H%M%S)
echo "Running evaluation for LLaDA..."
python evaluate_adaptive_policy.py \
    --model llada \
    --policy_path ./checkpoints/policy_llada.pt \
    > eval_llada_$DATE.log 2>&1

echo "Running evaluation for Dream..."
python evaluate_adaptive_policy.py \
    --model dream \
    --policy_path ./checkpoints/policy_dream.pt \
    > eval_dream_$DATE.log 2>&1

echo "Evaluation complete. Check eval_llada_$DATE.log and eval_dream_$DATE.log."