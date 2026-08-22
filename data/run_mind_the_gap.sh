#!/usr/bin/env bash
set -euo pipefail
set -x

source /home/liting/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-verl070}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

INPUT="${INPUT:-data/DeepScaler/Qwen2d5_math_7b/train_800.success_rate_k8.right.parquet}"
OUTPUT="${OUTPUT:-data/DeepScaler/Qwen2d5_math_7b/train_800.mind_the_gap.parquet}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B}"
SAMPLES="${SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"
START="${START:-0}"

log_dir="${LOG_DIR:-outputs/mind_the_gap}"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_dir}/rewrite.log") 2>&1

args=(
  --input "${INPUT}"
  --output "${OUTPUT}"
  --model "${MODEL_PATH}"
  --samples "${SAMPLES}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --start "${START}"
)

if [[ -n "${END:-}" ]]; then
  args+=(--end "${END}")
fi

python3 data/data.py "${args[@]}" "$@"
