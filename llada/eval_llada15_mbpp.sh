# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='GSAI-ML/LLaDA-1.5'

# Clean old results
rm -rf evals_results/baseline/llada15-mbpp-s256 2>/dev/null
rm -rf evals_results/baseline/llada15-mbpp-s256.log 2>/dev/null

mkdir -p evals_results/baseline/llada15-mbpp-s256

# dual cache+parallel
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--batch_size 1  \
--output_path evals_results/baseline/llada15-mbpp-s256 --log_samples --limit 100 2>&1 | tee -a evals_results/baseline/llada15-mbpp-s256.log

echo "post-processed accuracy:" 2>&1 | tee -a evals_results/baseline/llada15-mbpp-s256.log
## NOTICE: use postprocess for mbpp
python postprocess_code.py evals_results/baseline/llada15-mbpp-s256/*/samples_llada15-mbpp*.jsonl 2>&1 | tee -a evals_results/baseline/llada15-mbpp-s256.log