#!/usr/bin/env bash
set -euo pipefail

SRC=/root/autodl-tmp/src
COVO=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo
PY=/root/autodl-tmp/great/bin/python
GEN_PID=40796
ROUTE_LOG="$SRC/logs/experiment_librispeech_sensevoice_to_qwen9b_route_20260812.log"
TRAIN_LOG="$SRC/logs/train_qwen35_9b_librispeech_sensevoice_balanced_20260812.log"
TRAIN_EVIDENCE="$SRC/logs/librispeech_train-clean-100_sensevoice_nbest10_20260812.jsonl"
TRAIN_SFT="$SRC/logs/librispeech_train-clean-100_sensevoice_covo_balanced_sft_20260812.jsonl"
TRAIN_SUMMARY="$SRC/logs/librispeech_train-clean-100_sensevoice_covo_balanced_sft_summary_20260812.json"
OUTPUT_DIR="$COVO/outputs/qwen35_9b_librispeech_sensevoice_nbest10_balanced_1epoch_20260812"

exec >> "$ROUTE_LOG" 2>&1
echo "[$(date -Is)] waiting_for_train_evidence pid=$GEN_PID"
while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep 60
done

train_rows=$(wc -l < "$TRAIN_EVIDENCE")
echo "[$(date -Is)] train_evidence_rows=$train_rows"
if [[ "$train_rows" -ne 28539 ]]; then
  echo "train evidence incomplete" >&2
  exit 2
fi

cd "$SRC"
for split in dev-clean dev-other test-clean test-other; do
  echo "[$(date -Is)] generating_$split"
  "$PY" analysis/generate_sensevoice_librispeech.py \
    --input "logs/librispeech_${split}_manifest_20260812.jsonl" \
    --output "logs/librispeech_${split}_sensevoice_nbest10_20260812.jsonl" \
    --beam-size 16 \
    --token-topk 32 \
    --max-nbest 10 \
    --progress-every 100
done

echo "[$(date -Is)] building_train_sft"
"$PY" analysis/build_librispeech_covo_sft.py \
  --input "$TRAIN_EVIDENCE" \
  --output "$TRAIN_SFT" \
  --summary "$TRAIN_SUMMARY" \
  --train \
  --preserve-ratio 0.30

for split in dev-clean dev-other test-clean test-other; do
  "$PY" analysis/build_librispeech_covo_sft.py \
    --input "logs/librispeech_${split}_sensevoice_nbest10_20260812.jsonl" \
    --output "logs/librispeech_${split}_sensevoice_covo_eval_20260812.jsonl" \
    --summary "logs/librispeech_${split}_sensevoice_covo_eval_summary_20260812.json"
done

echo "[$(date -Is)] validating_sft"
cd "$COVO"
"$PY" scripts/train_lora_sft.py \
  --train-file "$TRAIN_SFT" \
  --output-dir /tmp/librispeech_qwen9b_dryrun \
  --model-name-or-path dummy \
  --max-length 768 \
  --disable-thinking \
  --dry-run >/tmp/librispeech_qwen9b_dryrun.txt

free_kb=$(df --output=avail /root/autodl-tmp | tail -1)
if [[ "$free_kb" -lt 8388608 ]]; then
  echo "insufficient disk before training: ${free_kb}KB" >&2
  exit 3
fi

echo "[$(date -Is)] starting_qwen9b_lora output=$OUTPUT_DIR"
"$PY" scripts/train_lora_sft.py \
  --train-file "$TRAIN_SFT" \
  --output-dir "$OUTPUT_DIR" \
  --model-name-or-path "$COVO/../models/Qwen3.5-9B" \
  --max-length 768 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 1000 \
  --logging-steps 10 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking \
  > "$TRAIN_LOG" 2>&1

echo "[$(date -Is)] route_complete"
