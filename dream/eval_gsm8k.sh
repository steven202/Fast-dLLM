# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=256
model="Dream-org/Dream-v0-Base-7B"

# Clean old results
rm -rf evals_results/baseline/gsm8k-s256 2>/dev/null
rm -rf evals_results/baseline/gsm8k-s256.log 2>/dev/null

mkdir -p evals_results/baseline/gsm8k-s256

# dual cache+parallel
accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true,dual_cache=true,show_speed=True \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1  \
    --output_path evals_results/baseline/gsm8k-s256 --log_samples \
    --confirm_run_unsafe_code --limit 100 2>&1 | tee -a evals_results/baseline/gsm8k-s256.log
