#!/usr/bin/env bash
set -euo pipefail

COVO=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
TRAIN="$COVO/data/processed/stcmds_chinesehp_sensevoice/train.cb-hardnegative.qwen.jsonl"
DEV="$COVO/data/processed/stcmds_chinesehp_sensevoice/dev.qwen.jsonl"
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
OUT="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817"

cd "$COVO"
echo "[$(date '+%F %T')] START train_rows=$(wc -l < "$TRAIN") dev_rows=$(wc -l < "$DEV")"
"$PY" scripts/train_lora_sft.py \
  --train-file "$TRAIN" \
  --eval-file "$DEV" \
  --output-dir "$OUT" \
  --model-name-or-path "$BASE" \
  --input-format qwen-messages \
  --max-length 1024 \
  --epochs 2 \
  --learning-rate 2e-4 \
  --lr-scheduler-type linear \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-steps 4300 \
  --eval-steps 4300 \
  --preprocessing-num-workers 1 \
  --dataloader-num-workers 0 \
  --dataloader-prefetch-factor 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking
echo "[$(date '+%F %T')] ALL_DONE"
