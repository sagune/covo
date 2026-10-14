#!/usr/bin/env bash
# ST-CMDS held-out 9B correction inference, mirroring the THCHS-30 suite settings.
set -euo pipefail

WORKSPACE=/root/autodl-tmp
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WORKSPACE/great/bin/python"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
OUT="$WORKSPACE/.dsh_checks/stcmds"
IN="$OUT/stcmds_selector.messages.jsonl"
PRED="$OUT/stcmds_9b_stcmds_adapter.predictions.jsonl"
LOG="$OUT/infer.log"

cd "$COVO"
echo "[$(date -Is)] START rows=$(wc -l < "$IN") adapter=$ADAPTER" >> "$LOG"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
  --input "$IN" --output "$PRED.inprogress" \
  --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
  --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1

mv -f "$PRED.inprogress" "$PRED"
echo "[$(date -Is)] DONE rows=$(wc -l < "$PRED")" >> "$LOG"
touch "$OUT/INFER_DONE"
