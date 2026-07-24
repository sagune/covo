#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/great/bin/python}"
CONFIG="${CONFIG:-configs/cb-sensevoice-aishell.yaml}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-17}"
LOG_DIR="${LOG_DIR:-logs}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cb_sensevoice_w14_train_evidence}"

mkdir -p "$LOG_DIR"

for shard in $(seq "$START_SHARD" "$END_SHARD"); do
  split=$(printf "train_covo_shard%02d" "$shard")
  evidence="$LOG_DIR/${OUTPUT_PREFIX}_shard$(printf "%02d" "$shard").jsonl"
  stdout_log="$LOG_DIR/experiment_${OUTPUT_PREFIX}_shard$(printf "%02d" "$shard").log"
  expected=1000
  if [[ "$shard" == "17" ]]; then
    expected=301
  fi

  if [[ -f "$evidence" ]]; then
    lines=$(wc -l < "$evidence")
    if [[ "$lines" -eq "$expected" ]]; then
      echo "[skip] $split lines=$lines"
      continue
    fi
    backup="${evidence}.incomplete.$(date +%s)"
    mv "$evidence" "$backup"
    echo "[resume] moved incomplete $lines/$expected rows to $backup"
  fi

  echo "[start] $split $(date -Is)"
  TRANSFORMERS_VERBOSITY=error \
  TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  CBW_EVIDENCE_ONLY=1 \
  CBW_EVIDENCE_OUT="$evidence" \
  "$PYTHON_BIN" run_CLI.py test \
    --config "$CONFIG" \
    --data.init_args.test_split="$split" \
    --model.init_args.split="$split" \
    --model.init_args.oracle_nbest_diagnostic=false \
    > "$stdout_log" 2>&1

  lines=$(wc -l < "$evidence")
  if [[ "$lines" -ne "$expected" ]]; then
    echo "[error] $split produced $lines/$expected rows" >&2
    exit 1
  fi
  echo "[done] $split lines=$lines $(date -Is)"
done

merged="$LOG_DIR/${OUTPUT_PREFIX}_full.jsonl"
temporary="${merged}.tmp"
: > "$temporary"
for shard in $(seq 0 17); do
  evidence="$LOG_DIR/${OUTPUT_PREFIX}_shard$(printf "%02d" "$shard").jsonl"
  [[ -f "$evidence" ]] || {
    echo "[error] missing $evidence" >&2
    exit 1
  }
  cat "$evidence" >> "$temporary"
done
mv "$temporary" "$merged"

lines=$(wc -l < "$merged")
if [[ "$lines" -ne 17301 ]]; then
  echo "[error] merged evidence has $lines/17301 rows" >&2
  exit 1
fi
echo "[all-done] $merged lines=$lines $(date -Is)"
