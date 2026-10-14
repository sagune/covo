#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
DATA="$ROOT/data/processed/magicdata_original_covo"
OUT="$ROOT/outputs/qwen35_9b_magicdata_hardneg_balanced_1epoch_20260820"

mkdir -p "$OUT"
cd "$ROOT"

train="$DATA/train.hardneg.balanced.qwen.jsonl"
expected=313473
actual=$(wc -l < "$train")
if [[ "$actual" -ne "$expected" ]]; then
  echo "[error] unexpected train rows: $actual/$expected" >&2
  exit 1
fi

echo "[$(date '+%F %T')] START train_rows=$actual epochs=1 expected_steps=39185"
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="$ROOT/src" \
"$PY" scripts/train_lora_sft.py \
  --train-file "$train" \
  --output-dir "$OUT" \
  --model-name-or-path "$BASE" \
  --input-format qwen-messages \
  --max-length 1024 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --lr-scheduler-type linear \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-steps 4900 \
  --preprocessing-num-workers 1 \
  --dataloader-num-workers 0 \
  --dataloader-prefetch-factor 2 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --bf16 --gradient-checkpointing --disable-thinking

echo "[$(date '+%F %T')] ALL_DONE"
