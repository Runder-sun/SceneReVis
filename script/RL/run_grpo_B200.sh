#!/bin/bash
export OPENSSL_FIPS=0 
export OPENSSL_CONF=/dev/null
export DEEPSPEED_DISABLE_TRITON=1
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True 
export VERL_LOGGING_LEVEL=INFO
export VLLM_USE_V1=1

# 1. 每次启动前清理环境，防止显存碎片
pkill -f "ray" || true
pkill -f "sglang" || true
sleep 5

set -x
ulimit -n 65535

# 路径配置
INTERACTION_CONFIG=/path/to/workspace/llmscene/script/RL/config/scene_editing_interaction_B200.yaml
MODEL_PATH=/path/to/data/ckpt/sft/sft_output_b200_8card_50k/v1-20251210-195003/checkpoint-849
TRAIN_DATA=[/path/to/data/datasets/rl/scene_editing_train_v2.parquet,/path/to/data/datasets/rl/scene_editing_train_ood.parquet]
VAL_DATA=[/path/to/data/datasets/rl/scene_editing_val_v2.parquet,/path/to/data/datasets/rl/scene_editing_val_ood.parquet]

MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=40960
ACTOR_MAX_TOKEN=51200
INFER_MAX_TOKEN=51200

# === [修改 1] 使用 4 卡 ===
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
    custom_reward_function.path="scene_reward.py" \
    custom_reward_function.name="compute_score" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${INTERACTION_CONFIG}" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=16 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=true \
    data.return_raw_chat=true \
    data.return_multi_modal_inputs=false \
    data.image_key=images \
    data.trust_remote_code=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN} \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.004 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN} \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=7 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.nccl_timeout=100000 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=vlmscene_grpo_B200_v2 \
    trainer.experiment_name=qwen2.5-vl-7b-b200-4card-speedup \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.default_local_dir=/path/to/data/rl_output_scene_editing/B200_v7/ckpt \
    trainer.save_freq=10 \
    trainer.test_freq=40 \
    trainer.total_epochs=2