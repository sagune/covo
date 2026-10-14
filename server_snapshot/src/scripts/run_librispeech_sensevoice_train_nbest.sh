#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/src
OUT=logs/librispeech_train-clean-100_sensevoice_nbest10_20260812.jsonl
LOG=logs/experiment_librispeech_train-clean-100_sensevoice_nbest10_20260812.log

/root/autodl-tmp/great/bin/python analysis/generate_sensevoice_librispeech.py \
  --input logs/librispeech_train-clean-100_manifest_20260812.jsonl \
  --output "$OUT" \
  --beam-size 16 \
  --token-topk 32 \
  --max-nbest 10 \
  --progress-every 100 \
  >> "$LOG" 2>&1
