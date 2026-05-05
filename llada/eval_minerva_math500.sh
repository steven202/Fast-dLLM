# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0
task=minerva_math500
length=256
block_length=32
num_fewshot=4
steps=256
model_path='GSAI-ML/LLaDA-8B-Instruct'

# Clean old results
rm -rf evals_results/baseline/minerva_math500-s256 2>/dev/null
rm -rf evals_results/baseline/minerva_math500-s256.log 2>/dev/null

mkdir -p evals_results/baseline/minerva_math500-s256

# dual cache+parallel
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--batch_size 1  \
--output_path evals_results/baseline/minerva_math500-s256 --log_samples --limit 100 2>&1 | tee -a evals_results/baseline/minerva_math500-s256.log
