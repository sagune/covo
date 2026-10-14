#!/usr/bin/env bash
set -euo pipefail
SRC=/root/autodl-tmp/src
COVO=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
BASE=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
ADAPTER=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_9b_librispeech_sensevoice_nbest10_balanced_batch4_2epoch_20260812
LOG=$SRC/logs/experiment_librispeech_qwen9b_adapter_eval_20260813.log
exec >>"$LOG" 2>&1
echo "[$(date -Is)] eval_start"
for split in dev-clean dev-other test-clean test-other; do
  input="$SRC/logs/librispeech_${split}_sensevoice_covo_eval_20260812.jsonl"
  output="$SRC/logs/librispeech_${split}_qwen9b_sensevoice_predictions_20260813.jsonl"
  summary="$SRC/logs/librispeech_${split}_qwen9b_sensevoice_summary_20260813.json"
  echo "[$(date -Is)] start_${split}"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$input" \
    --output "$output" \
    --model-name-or-path "$BASE" \
    --adapter-path "$ADAPTER" \
    --batch-size 4 \
    --max-new-tokens 256 \
    --progress-every 100 \
    --disable-thinking
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/evaluate_correction_jsonl.py" \
    --input "$output" >"$summary"
  echo "[$(date -Is)] summary_${split}"
  cat "$summary"
done
echo "[$(date -Is)] eval_complete"
