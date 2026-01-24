
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=humaneval
length=256
block_length=32
steps=$((length / block_length))
model="Dream-org/Dream-v0-Base-7B"
policy_path='./checkpoints/policy_dream.pt'


# 1. Baseline
echo "Running Baseline..."
accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},show_speed=True \
--output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# 2. Prefix Cache
echo "Running Prefix Cache..."
accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,show_speed=True \
--output_path evals_results/prefix_cache/humaneval-ns0-${length} --log_samples

# 3. Parallel
echo "Running Parallel..."
accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},threshold=0.9,show_speed=True \
--output_path evals_results/parallel/humaneval-ns0-${length} --log_samples

# 4. Prefix Cache + Parallel
echo "Running Prefix Cache + Parallel..."
accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/cache_parallel/humaneval-ns0-${length} --log_samples

# 5. Dual Cache + Parallel
echo "Running Dual Cache + Parallel..."
accelerate launch eval_adaptive.py --tasks ${task} \
--confirm_run_unsafe_code --model adaptive_dream \
--model_args pretrained=${model},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True \
--output_path evals_results/dual_cache_parallel/humaneval-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
echo "Run postprocess (example for baseline)..."
# python dream/postprocess_code.py {the samples_xxx.jsonl file under output_path}
