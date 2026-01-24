
# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
policy_path='./checkpoints/policy_llada.pt'


# 1. Baseline (Adaptive with full steps allowance)
echo "Running Baseline (steps=${length})..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},threshold=0.9,show_speed=True 


# 2. Prefix cache (Note: Adaptive Policy currently warns about cache support)
echo "Running Prefix Cache..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,show_speed=True 


# 3. Parallel (Constrained steps - forces policy to be aggressive)
echo "Running Parallel (steps=${steps})..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},threshold=0.9,show_speed=True

# 4. Parallel factor
echo "Running Parallel Factor..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},factor=${factor},threshold=0.9,show_speed=True


# 5. Prefix cache+parallel
echo "Running Prefix Cache + Parallel..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,threshold=0.9,show_speed=True

# 6. Dual cache+parallel
echo "Running Dual Cache + Parallel..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,dual_cache=True,threshold=0.9,show_speed=True

# 7. Prefix cache+parallel factor
echo "Running Prefix Cache + Parallel Factor..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${steps},block_len_max=${block_length},use_cache=True,factor=${factor},threshold=0.9,show_speed=True

# 8. Dual cache+parallel factor
echo "Running Dual Cache + Parallel Factor..."
accelerate launch eval_adaptive.py --tasks ${task} --num_fewshot ${num_fewshot} \
--model adaptive_llada \
--model_args pretrained=${model_path},policy_path=${policy_path},gen_length=${length},steps=${length},block_len_max=${block_length},use_cache=True,dual_cache=True,factor=${factor},threshold=0.9,show_speed=True
