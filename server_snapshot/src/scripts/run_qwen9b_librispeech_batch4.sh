#!/usr/bin/env bash
set -euo pipefail

PY=/root/autodl-tmp/great/bin/python
COVO=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
TRAIN=/root/autodl-tmp/src/logs/librispeech_train-clean-100_sensevoice_covo_balanced_sft_20260812.jsonl
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
OUTPUT=$COVO/outputs/qwen35_9b_librispeech_sensevoice_nbest10_balanced_batch4_1epoch_20260812
LOG=/root/autodl-tmp/src/logs/train_qwen35_9b_librispeech_sensevoice_balanced_batch4_20260812.log

cd "$COVO"
exec "$PY" scripts/train_lora_sft.py \
  --train-file "$TRAIN" \
  --output-dir "$OUTPUT" \
  --model-name-or-path "$BASE" \
  --max-length 768 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --save-steps 1000 \
  --logging-steps 10 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking \
  > "$LOG" 2>&1
