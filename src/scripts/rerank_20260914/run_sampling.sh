#!/usr/bin/env bash
# 5 temperature samples over the damaged+control subset (no seed in the inference script,
# so repeated runs give independent samples).
set -euo pipefail

WORKSPACE=/root/autodl-tmp
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WORKSPACE/great/bin/python"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
OUT="$WORKSPACE/.dsh_checks/sampling"
IN="$OUT/subset.messages.jsonl"
LOG="$OUT/sampling.log"

mkdir -p "$OUT"
"$PY" "$WORKSPACE/.dsh_checks/build_sample_subset.py" >> "$LOG" 2>&1
echo "[$(date -Is)] subset rows=$(wc -l < "$IN")" >> "$LOG"

cd "$COVO"
for k in 1 2 3 4 5; do
  PRED="$OUT/sample_$k.predictions.jsonl"
  [[ -s "$PRED" ]] && continue
  echo "[$(date -Is)] SAMPLE $k START" >> "$LOG"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$IN" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
    --batch-size 8 --max-new-tokens 96 --temperature 0.8 --top-p 0.95 \
    --progress-every 128 --disable-thinking >> "$LOG" 2>&1
  mv -f "$PRED.inprogress" "$PRED"
  echo "[$(date -Is)] SAMPLE $k DONE" >> "$LOG"
done

echo "[$(date -Is)] ALL_SAMPLES_DONE" >> "$LOG"
touch "$OUT/SAMPLING_DONE"
