#!/bin/bash
set -x

# SAT2 Spatial Reasoning with Qwen2.5-VL-3B using TTRL
# This script implements Test-Time Reinforcement Learning for multi-modal spatial reasoning

# ===========================
# Configuration Parameters
# ===========================

TASK="SAT2-SPATIAL"
BACKBONE="Qwen2.5-VL-3B-Instruct"
ADVANTAGE="grpo"  # Use GRPO (Group Relative Policy Optimization) advantage estimator

# Model path - adjust this to your model location
BACKBONE_PATH=${BACKBONE_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}

# Data paths - adjust to your data location
DATA_LOCAL_DIR=${DATA_LOCAL_DIR:-"$HOME/data"}

# Response length configuration
K=2  # Response length multiplier
MAX_PROMPT_LENGTH=1024  # Max tokens for prompt (including image tokens)
MAX_RESPONSE_LENGTH=$((512 * $K))  # 1024 tokens for response

# Validation configuration
N=16  # Number of samples for validation

# Training configuration
EPISODE=50  # Total training epochs
DATA_TRAIN_BATCH_SIZE=4  # Number of prompts per batch (reduced for VL models)

# TTRL-specific configuration
N_VOTES_PER_PROMPT=32  # Number of samples to generate for majority voting
N_SAMPLES_PER_PROMPT=16  # Number of samples to use for actual training

# PPO configuration
MINI_BATCH_SIZE=1  # PPO mini-batch size
MICRO_BATCH_SIZE=2  # Micro-batch size for gradient accumulation

# Hardware configuration
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}  # Number of GPUs per node
NNODES=${NNODES:-1}  # Number of nodes

# Rollout configuration
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}  # vLLM tensor parallelism
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.6}  # GPU memory for vLLM

# ===========================
# Run TTRL Training
# ===========================

python -m verl.trainer.main_ppo \
  --config-name='ppo_trainer_ttrl.yaml' \
  \
  # ===========================
  # Data Configuration
  # ===========================
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.image_key=images \
  \
  # ===========================
  # Model Configuration
  # ===========================
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  \
  # ===========================
  # Actor Training Configuration
  # ===========================
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$((DATA_TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT / MINI_BATCH_SIZE)) \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  \
  # ===========================
  # Rollout Configuration (vLLM)
  # ===========================
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.top_p=0.9 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
  \
  # ===========================
  # Reference Model Configuration
  # ===========================
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  \
  # ===========================
  # Algorithm Configuration
  # ===========================
  algorithm.adv_estimator=$ADVANTAGE \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.use_kl_in_reward=False \
  \
  # ===========================
  # Custom Reward Function
  # ===========================
  custom_reward_function.path="./verl/utils/reward_score/sat2/__init__.py" \
  custom_reward_function.name=reward_func \
  \
  # ===========================
  # TTRL Configuration
  # ===========================
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  \
  # ===========================
  # Validation Configuration
  # ===========================
  val.n=$N \
  val.generation_config.temperature=0.6 \
  val.generation_config.top_p=0.9 \
  val.generation_config.max_new_tokens=$MAX_RESPONSE_LENGTH \
  \
  # ===========================
  # Trainer Configuration
  # ===========================
  trainer.total_epochs=$EPISODE \
  trainer.save_freq=10 \
  trainer.test_freq=5 \
  trainer.critic_warmup=0 \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$NNODES \
  \
  # ===========================
  # Logging Configuration
  # ===========================
  trainer.logger=['console','wandb'] \
  trainer.project_name='ttrl_sat2_spatial_reasoning' \
  trainer.experiment_name="${BACKBONE}_ttrl_n${N_VOTES_PER_PROMPT}_k${N_SAMPLES_PER_PROMPT}" \
  \
  $@
