# Rethink Agentic World Models

Environment code and experiment entry points live together in `scripts/{alfworld,sciworld,webshop}/`: `server.py`, `env_client.py`, `agent_loop.py`, prompts, `agent.yaml`, data preparation and training scripts.

Run commands from `/fs_mol/liujiarun/rethink-agentic-wm`. Training uses `/fs_mol/liujiarun/envs/verl-0.9.1`; each environment server uses its own conda environment. Run one model job per node.

| Environment | Server conda environment | Port |
|---|---|---|
| ALFWorld | `/fs_mol/liujiarun/envs/alfworld` | 8001 |
| SciWorld | `/fs_mol/liujiarun/envs/sciworld` | 8002 |
| WebShop | `/fs_mol/liujiarun/envs/webshop` | 8000 |

## Prepare RL data

Use the corresponding server environment:

```bash
python scripts/alfworld/prepare_rl_data.py --output-dir datasets/alfworld --val-split valid_seen
python scripts/sciworld/prepare_rl_data.py --output-dir datasets/sciworld --val-split dev
bash scripts/webshop/build_search_index.sh
python scripts/webshop/prepare_rl_data.py --webshop-path /fs_mol/liujiarun/WebShop \
  --output-dir datasets/webshop --train-range 1500 6910 --val-range 500 1500
```

## Server and RL

Start the server and training in separate tmux sessions. Example for ALFWorld:

```bash
conda activate /fs_mol/liujiarun/envs/alfworld
bash scripts/alfworld/start_server.sh
```

```bash
conda activate /fs_mol/liujiarun/envs/verl-0.9.1
bash scripts/alfworld/train.sh --experiment-name grpo-qwen2.5-3b-instruct-new \
  trainer.total_training_steps=100 \
  trainer.save_freq=10 trainer.max_actor_ckpt_to_keep=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.grad_alignment.interval=10
```

Substitute `sciworld` or `webshop` for other environments. Check existing processes and server capacity before launching. Each `train.sh` contains its environment settings and baseline recipe; extra Hydra arguments override training settings.

All training scripts accept `--experiment-name NAME`, `EXPERIMENT_NAME=NAME`, or `trainer.experiment_name=NAME`. Command-line names take precedence; default output paths follow the resolved name. Explicit output-path overrides remain authoritative. Use a new name for a new experiment.

## SFT and SFT-initialized RL

Audited demonstrations are in `datasets/<env>/sft/`: `train.parquet`, `metadata.json` and `splits.json`. ALFWorld has 82 unique demonstrations / 88 training rows; SciWorld has 98 / 104. See metadata for coverage and repeated examples. Original successful and failed demonstrations are retained in source/raw/. RL scene indices remain in the parent dataset directory.

```bash
conda activate /fs_mol/liujiarun/envs/verl-0.9.1
TASK=alfworld  # or sciworld
RUN=sft-qwen2.5-3b-instruct-new
bash scripts/$TASK/train_sft.sh datasets/$TASK/sft outputs/checkpoints/$TASK/$RUN \
  --experiment-name "$RUN"
```

SFT uses native `torchrun` + `verl.trainer.sft_trainer`: 8 GPUs, full-parameter FSDP2, one epoch, action-only supervision. The two positional paths precede name/Hydra options. To initialize RL from SFT:

```bash
bash scripts/alfworld/train.sh --experiment-name grpo-qwen2.5-3b-instruct-w-sft-new \
  actor_rollout_ref.model.path=/absolute/path/to/sft/global_step_N/huggingface \
  trainer.resume_mode=disable trainer.total_training_steps=100
```

Use the matching prompt, observation and server configuration. SciWorld's `sft_prompts.py` preserves the SFT representation and differs from default `prompts.py`; changing weights alone does not select that representation. ALFWorld `valid_unseen` stays excluded from training and tuning.

## Build new demonstrations

Collect only official train scenes in the corresponding simulator environment, then review reset/replayed trajectories and failures. Use fresh paths; do not overwrite a published dataset.

```bash
# ALFWorld: six task types × 20 scenes, at most 30 actions.
/fs_mol/liujiarun/envs/alfworld/bin/python scripts/alfworld/collect_sft_demos.py \
  --output-dir datasets/alfworld/sft/source-new --per-class 20 --seed 20261102 --max-actions 30
# SciWorld: 30 task types × four variations, at most 100 actions.
/fs_mol/liujiarun/envs/sciworld/bin/python scripts/sciworld/collect_sft_demos.py \
  --output-dir datasets/sciworld/sft/source-new --per-class 4 --seed 20261029 --max-actions 100

conda activate /fs_mol/liujiarun/envs/verl-0.9.1
TASK=alfworld  # or sciworld
python scripts/$TASK/prepare_sft_data.py --data-dir datasets/$TASK/sft/source-new \
  --output-dir datasets/$TASK/sft/new-version \
  --model /fs_mol/liujiarun/models/qwen2.5-3b-instruct
```

## Results

| Path | Contents |
|---|---|
| `outputs/checkpoints/<env>/<run>/` | Retained weights and optimizer state |
| `outputs/trajectories/<env>/<run>/` | Action/observation transcripts and scores |
| `outputs/tensorboard/<env>/<run>/` | Training metrics |
| `outputs/results/grad_alignment/<env>/<run>/` | Gradient CSVs and plots |

```bash
tensorboard --logdir outputs/tensorboard --port 6006
python scripts/visualize_grad_alignment.py \
  --csv_dir outputs/results/grad_alignment/sciworld/grpo-qwen2.5-3b-instruct-w-sft
python -m pytest -q tests/utils/test_env_episode_helpers.py tests/utils/test_online_grad_alignment.py
```

RL weights are FSDP shards under `global_step_N/actor/`; `data.pt` is dataloader state. `trainer.max_actor_ckpt_to_keep` limits retained weight checkpoints. The gradient probe compares complete actor loss with WM observation prediction using turn-mean reduction; zero-norm cosine is NaN. Disable with `actor_rollout_ref.actor.grad_alignment.enable=False`.
