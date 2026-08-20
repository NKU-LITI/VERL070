#!/usr/bin/env bash

set -x
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

PROJECT_NAME="${PROJECT_NAME:-scaf-grpo}"
EXP_NAME="${EXP_NAME:-outputs/qwen25_math7b_grpo}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-7B}"
DATA_SEED="${DATA_SEED:-42}"

mkdir -p "${EXP_NAME}"
exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

CONDA_SH="${CONDA_SH:-/home/liting/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl070}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

data_dir="${DATA_DIR:-data/DeepScaler/Qwen2d5_math_7b}"
data_train_path="${DATA_TRAIN_PATH:-${data_dir}/train_800.success_rate_k8.right.parquet}"
data_val_path="${DATA_VAL_PATH:-${data_dir}/val_200.success_rate_k8.right.parquet}"

# ------------------------------------------------------------------------
# Train
nnodes="${NNODES:-1}"
n_gpus_per_node="${N_GPUS_PER_NODE:-2}"
epoch="${EPOCH:-10}"
lr="${LR:-1e-6}"
wd="${WEIGHT_DECAY:-0.0}"
warmup_steps="${WARMUP_STEPS:-5}"
save_freq="${SAVE_FREQ:--1}"
test_freq="${TEST_FREQ:-5}"

# Batch / sequence sizes.
train_batchsize="${TRAIN_BATCH_SIZE:-64}"
val_batchsize="${VAL_BATCH_SIZE:-64}"
ppo_mini_batchsize="${PPO_MINI_BATCH_SIZE:-16}"
micro_batchsize_per_gpu="${MICRO_BATCH_SIZE_PER_GPU:-2}"
max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_response_length="${MAX_RESPONSE_LENGTH:-4096}"

# Rollout
n_rollout="${N_ROLLOUT:-8}"
val_n_rollout="${VAL_N_ROLLOUT:-8}"
train_temp="${TRAIN_TEMPERATURE:-1.0}"
vllm_gpu_memory_util="${VLLM_GPU_MEMORY_UTIL:-0.30}"
vllm_max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
vllm_tp="${VLLM_TENSOR_MODEL_PARALLEL_SIZE:-2}"
vllm_enforce_eager="${VLLM_ENFORCE_EAGER:-True}"

if ! nvidia-smi; then
    echo "ERROR: nvidia-smi failed. GPU/driver is not available; aborting before training."
    exit 9
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="${data_train_path}" \
    data.val_files="${data_val_path}" \
    data.shuffle=True \
    data.seed="${DATA_SEED}" \
    data.train_batch_size="${train_batchsize}" \
    data.val_batch_size="${val_batchsize}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batchsize}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_batchsize_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${micro_batchsize_per_gpu}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${micro_batchsize_per_gpu}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature="${train_temp}" \
    actor_rollout_ref.rollout.n="${n_rollout}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${val_n_rollout}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${vllm_gpu_memory_util}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${vllm_tp}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${vllm_max_num_batched_tokens}" \
    actor_rollout_ref.rollout.enforce_eager="${vllm_enforce_eager}" \
    actor_rollout_ref.actor.optim.lr="${lr}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay="${wd}" \
    reward_manager.name=naive \
    reward_model.reward_manager=remote \
    reward_model.num_workers=8 \
    trainer.nnodes="${nnodes}" \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.total_epochs="${epoch}" \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.val_before_train=True \
    trainer.with_hint=True \
    trainer.with_luffy_expert=True \
    trainer.luffy_expert_every_group=False \
    trainer.luffy_expert_key=qwen_expert_trajectory \
    trainer.replace_hint_prompt=False \
    trainer.replace_num=1 \
    trainer.rollout_data_dir="${EXP_NAME}/rollout_log/training" \
    trainer.validation_data_dir="${EXP_NAME}/rollout_log/validation" \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    trainer.hint_data_dir="${EXP_NAME}/rollout_log/hint" \
    trainer.warmup_steps="${warmup_steps}" \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.tracking_dir="${EXP_NAME}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    "$@"
