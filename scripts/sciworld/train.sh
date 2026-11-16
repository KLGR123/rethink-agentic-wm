#!/usr/bin/env bash
# GRPO training of Qwen2.5 model on ScienceWorld (single node, 8×A100 80GB).
#
# Prerequisites (run before this script)
# - Prepare parquet data: `python scripts/sciworld/prepare_rl_data.py`
# - Start the env server (separate terminal): `bash scripts/sciworld/start_server.sh`

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export SCIWORLD_ENV_URL="http://127.0.0.1:8002"
# Must match --num-envs of the env server (start_env_server.sh reads the same
# variable). The agent loop divides this by agent.num_workers to cap episodes
# in flight, so a mismatch either starves the pool or leaves it idle.
# Every ScienceWorld env slot is a JVM (~220 MB RSS) and they all live on the
# env-server host, not on the training GPUs. 64 slots is ~14 GB — fine on this
# dev node, and enough to keep 32 episodes in flight busy.
export SCIWORLD_NUM_ENVS="64"
# ScienceWorld allows up to 100 simulator moves; the loop stops at the smaller
# of this and the response token budget.
export SCIWORLD_MAX_TURNS="30"

MODEL_PATH="/fs_mol/liujiarun/models/qwen2.5-3b-instruct"
DATA_DIR="/fs_mol/liujiarun/rethink-agentic-wm/datasets/sciworld"
TRAIN_FILES="${DATA_DIR}/train.parquet"
VAL_FILES="${DATA_DIR}/val.parquet"
AGENT_CONFIG="$(pwd)/scripts/sciworld/agent.yaml"

PROJECT_NAME="sciworld"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-qwen2.5-3b-instruct}"
# Resolve the run name before constructing output paths.
TRAIN_ARGS=()
while (($#)); do
    case "$1" in
        --experiment-name) EXPERIMENT_NAME="${2:?Missing experiment name}"; shift 2 ;;
        --experiment-name=*|trainer.experiment_name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done
[[ "$EXPERIMENT_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || { echo "Invalid experiment name" >&2; exit 2; }
export EXPERIMENT_NAME
CKPT_DIR="/fs_mol/liujiarun/rethink-agentic-wm/outputs/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="/fs_mol/liujiarun/rethink-agentic-wm/outputs/trajectories/${PROJECT_NAME}/${EXPERIMENT_NAME}"

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/fs_mol/liujiarun/rethink-agentic-wm/outputs/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0


python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.grad_alignment.enable=True \
    actor_rollout_ref.actor.grad_alignment.interval=50 \
    actor_rollout_ref.actor.grad_alignment.output_dir="$(pwd)/outputs/results/grad_alignment/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.agent.default_agent_loop=sciworld_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.filter_groups.enable=False \
    algorithm.use_kl_in_reward=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.total_epochs=5 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.logger="[console,tensorboard]" \
    "${TRAIN_ARGS[@]}"
