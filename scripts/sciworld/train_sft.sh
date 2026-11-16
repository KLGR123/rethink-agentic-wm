#!/usr/bin/env bash
# Native verl SFT; no custom optimizer/trainer. Use a fresh output directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DATA_DIR="${1:?Usage: train_sft.sh AUDITED_DATA_DIR OUTPUT_DIR [--experiment-name NAME] [Hydra overrides]}"
OUTPUT_DIR="${2:?Specify a new checkpoint directory}"
shift 2
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$OUTPUT_DIR")}"
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
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$(pwd)/outputs/tensorboard/sciworld/${EXPERIMENT_NAME}}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
STEPS=$(python - "$DATA_DIR" <<'PYDATA'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
p = d/'metadata.json'
if not p.exists():
    p = d/'training-audit.json'  # Legacy isolated experiments remain usable.
audit = json.loads(p.read_text())
expected = audit.get('train_sha256') or audit.get('train_parquet_sha256')
assert expected and hashlib.sha256((d/'train.parquet').read_bytes()).hexdigest() == expected, 'SFT dataset hash mismatch'
rows = audit['training_rows']
assert rows > 0 and rows % 8 == 0, 'Dataset must match the global batch size of 8'
steps = audit.get('updates', rows // 8)
assert steps == rows // 8, 'Dataset update count mismatch'
print(steps)
PYDATA
)
torchrun --standalone --nproc_per_node=8 -m verl.trainer.sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" data.val_files=null data.messages_key=messages \
  data.train_batch_size=8 data.micro_batch_size_per_gpu=1 data.use_dynamic_bsz=False \
  data.max_token_len_per_gpu=20480 data.max_length=20480 data.pad_mode=no_padding \
  data.truncation=error data.num_workers=0 data.enable_thinking_default=null \
  model.path="/fs_mol/liujiarun/models/qwen2.5-3b-instruct" \
  model.use_remove_padding=True model.enable_gradient_checkpointing=True \
  engine=fsdp engine.strategy=fsdp2 engine.use_torch_compile=False engine.ulysses_sequence_parallel_size=1 \
  optim.lr=1e-5 optim.lr_scheduler_type=constant optim.lr_warmup_steps=1 optim.weight_decay=0.01 optim.clip_grad=1 \
  trainer.total_epochs=1 trainer.total_training_steps="$STEPS" trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.resume_mode=disable trainer.max_ckpt_to_keep=1 trainer.logger='[console,tensorboard]' \
  trainer.seed=20261029 trainer.project_name=sciworld-sft trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="$OUTPUT_DIR" trainer.nnodes=1 trainer.n_gpus_per_node=8 \
  checkpoint.save_contents='[model,optimizer,extra,hf_model]' "${TRAIN_ARGS[@]}"
