#!/usr/bin/env bash
set -euo pipefail

# Runs 4 baseline commands sequentially in this order:
# 1) humaneval @ 256
# 2) humaneval @ 512
# 3) gsm8k     @ 256
# 4) gsm8k     @ 512

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

model_path="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
block_length="${BLOCK_LENGTH:-32}"
threshold="${THRESHOLD:-0.9}"
mode="${MODE:-new}"               # new | resume
run_tag="${RUN_TAG:-$(date '+%Y%m%d-%H%M%S')}"
limit="${LIMIT:-}"
baseline_root="${BASELINE_ROOT:-evals_results/baseline_dual_cache_parallel_4runs}"

extra_cli=()
if [[ -n "$limit" ]]; then
  extra_cli+=(--limit "$limit")
fi

if [[ "$mode" != "new" && "$mode" != "resume" ]]; then
  echo "Unsupported MODE: $mode (expected new or resume)"
  exit 1
fi

log_dir="${LOG_DIR:-$script_dir/evals_results/logs}"
mkdir -p "$log_dir"
log_file="${LOG_FILE:-$log_dir/baseline_4runs_${run_tag}.log}"
touch "$log_file"
ln -sfn "$(basename "$log_file")" "$log_dir/latest_baseline_4runs.log"
exec > >(tee -a "$log_file") 2>&1

echo "============================================================"
echo "Run tag: $run_tag | Mode: $mode"
echo "Model: $model_path | block_length=$block_length | threshold=$threshold"
echo "Baseline root: $baseline_root"
echo "Order: humaneval(256,512) -> gsm8k(256,512)"
echo "Log file: $log_file"
echo "============================================================"

find_latest_sample_jsonl () {
  local out_dir="$1"
  local candidate
  candidate="$(find "$out_dir" -type f -name 'samples_humaneval_*.jsonl' -print 2>/dev/null | sort | tail -n 1)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  find "$out_dir" -type f -name 'samples_*.jsonl' -print 2>/dev/null | sort | tail -n 1
}

run_humaneval_postprocess () {
  local out_dir="$1"
  local score_log="$out_dir/cleaned_output.log"
  local sample_file
  sample_file="$(find_latest_sample_jsonl "$out_dir")"

  if [[ -z "$sample_file" ]]; then
    echo "[Postprocess][humaneval] No sample file found under $out_dir"
    {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] humaneval"
      echo "sample_file=NOT_FOUND"
      echo "pass@1=N/A"
      echo
    } | tee -a "$score_log"
    return
  fi

  echo "[Postprocess][humaneval] sample: $sample_file"
  local pp_output
  pp_output="$(python postprocess_code.py "$sample_file")"
  local pp_score
  pp_score="$(echo "$pp_output" | awk 'NF{last=$0} END{print last}')"

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] humaneval"
    echo "sample_file=$sample_file"
    echo "pass@1=$pp_score"
    echo
  } | tee -a "$score_log"
}

run_humaneval () {
  local length="$1"
  local task="humaneval"
  local num_fewshot=0
  local output_path=""
  local save_dir=""

  if [[ "$mode" == "new" ]]; then
    output_path="${baseline_root}/runs/${run_tag}/humaneval-ns0-${length}"
    save_dir="${output_path}/resume"
  else
    output_path="${baseline_root}/humaneval-ns0-${length}"
    save_dir="${output_path}/resume"
  fi

  echo
  echo "[humaneval][len=${length}] dual_cache+parallel"
  accelerate launch eval_llada.py --tasks "$task" \
    --num_fewshot "$num_fewshot" \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=${threshold},show_speed=True,save_dir=${save_dir}" \
    --output_path "$output_path" --log_samples "${extra_cli[@]}"

  run_humaneval_postprocess "$output_path"
}

run_gsm8k () {
  local length="$1"
  local task="gsm8k"
  local num_fewshot=5
  local save_dir=""

  if [[ "$mode" == "new" ]]; then
    save_dir="${baseline_root}/runs/${run_tag}/gsm8k-l${length}-b${block_length}"
  else
    save_dir="${baseline_root}/gsm8k-l${length}-b${block_length}"
  fi

  echo
  echo "[gsm8k][len=${length}] dual_cache+parallel"
  accelerate launch eval_llada.py --tasks "$task" --num_fewshot "$num_fewshot" \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args "model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},use_cache=True,dual_cache=True,threshold=${threshold},show_speed=True,save_dir=${save_dir}" \
    "${extra_cli[@]}"
}

# Required order by request:
run_humaneval 256
run_humaneval 512
run_gsm8k 256
run_gsm8k 512

echo
echo "Completed all 4 runs."
echo "Main log: $log_file"
