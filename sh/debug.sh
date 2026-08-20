#!/usr/bin/env bash



set -x
set -euo pipefail

source /home/liting/miniconda3/etc/profile.d/conda.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_USE_V1=1

PROJECT_NAME="scaf-grpo-expert-sft" 
EXP_NAME="${EXP_NAME:-outputs/qwen25_math7b_verl_070_expert_test}"
MODEL_PATH="${MODEL_PATH:-/workplace/nankai/liting_space/LLM/Qwen2.5-Math-1.5B}"
DATA_SEED="${DATA_SEED:-42}"


mkdir -p "${EXP_NAME}"
exec > >(tee -a "${EXP_NAME}/train.log") 2>&1

printf '\n===== restart %s =====\n' "$(date '+%Y-%m-%d %H:%M:%S')"

reward_tag="math-verify"
prompt_tag="system-p1"


data_dir="${DATA_DIR:-data/DeepScaler/Qwen2d5_math_7b}"
data_train_path="${DATA_TRAIN_PATH:-${data_dir}/train_800.success_rate_k8.right.parquet}"
data_val_path="${DATA_VAL_PATH:-${data_dir}/val_200.success_rate_k8.right.parquet}"




# ------------------------------------------------------------------------
### train
nnodes=1
n_gpus_per_node=1 # 8 
tensor_model_parallel_size=1 # 2
vllm_gpu_memory_util=0.35 # 0.8
epoch=10 # 100
lr=1e-6
wd=0.0
n_rollout=8
train_temp=1.0
train_batchsize=64 # 256
ppo_mini_batchsize=32 # 64
micro_batchsize_per_gpu=2 # 16
warmup_steps=5 # 50

### val
val_batchsize=64 # 512

###
save_freq=-1
test_freq=5 # 10


# epoch=2, step=24, warmup_steps内lr线性增加到设置的值
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_train_path \
    data.val_files="$data_val_path" \
    data.shuffle=True \
    data.seed="${DATA_SEED}" \
    data.train_batch_size=${train_batchsize} \
    data.val_batch_size=${val_batchsize} \
    data.max_prompt_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batchsize} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batchsize_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.off_policy_cliprange=0.2 \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.use_off_policy_probs=False \
    actor_rollout_ref.actor.use_off_policy_clip=False \
    actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
    actor_rollout_ref.actor.off_policy_reshape_weight=0.1 \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    actor_rollout_ref.actor.use_ppo_kl_loss=False \
    actor_rollout_ref.actor.ppo_kl_loss_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.temperature=${train_temp} \
    actor_rollout_ref.rollout.n=${n_rollout} \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_memory_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_model_parallel_size} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=${wd} \
    reward_manager.name=naive \
    reward_model.reward_manager=remote \
    reward_model.num_workers=1 \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.total_epochs=${epoch} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=True \
    trainer.with_hint=True \
    trainer.with_luffy_expert=True \
    trainer.luffy_expert_every_group=True \
    trainer.luffy_expert_key=qwen_expert_trajectory \
    trainer.replace_hint_prompt=False \
    trainer.replace_num=1 \
    trainer.rollout_data_dir="${EXP_NAME}/rollout_log/training" \
    trainer.validation_data_dir="${EXP_NAME}/rollout_log/validation" \
    trainer.default_local_dir="${EXP_NAME}/checkpoints" \
    trainer.hint_data_dir="${EXP_NAME}/rollout_log/hint" \
    trainer.warmup_steps=$warmup_steps \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.tracking_dir="${EXP_NAME}" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME $@
