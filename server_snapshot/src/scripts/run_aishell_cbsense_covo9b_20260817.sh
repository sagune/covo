#!/usr/bin/env bash
set -euo pipefail

SRC=/root/autodl-tmp/src
COVO=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
INPUT="$SRC/logs/cb_sensevoice_aishell_error_targeted_covo_messages_full_20260727.jsonl"
PRED="$SRC/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_predictions_20260817.jsonl"
SUMMARY="$SRC/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_summary_20260817.json"
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"

cd "$COVO"
echo "[$(date '+%F %T')] START rows=$(wc -l < "$INPUT")"
"$PY" scripts/infer_lora_text.py \
  --input "$INPUT" --output "$PRED" \
  --model-name-or-path "$BASE" --adapter-path "$ADAPTER" \
  --batch-size 8 --max-new-tokens 128 --progress-every 256 --disable-thinking
"$PY" scripts/evaluate_correction_jsonl.py --input "$PRED" > "$SUMMARY"
cat "$SUMMARY"
echo "[$(date '+%F %T')] ALL_DONE"
