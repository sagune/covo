#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/great/bin/python}"
CONFIG="${CONFIG:-configs/cb-whisper-aishell-v3-kws.yaml}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-17}"
EXPECTED_LINES="${EXPECTED_LINES:-1000}"
LAST_SHARD_EXPECTED_LINES="${LAST_SHARD_EXPECTED_LINES:-301}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "$LOG_DIR"

for shard in $(seq "$START_SHARD" "$END_SHARD"); do
  split=$(printf "train_full_shard%02d" "$shard")
  evidence="$LOG_DIR/cbwhisper_covo_evidence_${split}.jsonl"
  stdout_log="$LOG_DIR/experiment_cbwhisper_covo_evidence_${split}_stdout.log"
  expected="$EXPECTED_LINES"
  if [[ "$shard" == "17" ]]; then
    expected="$LAST_SHARD_EXPECTED_LINES"
  fi

  if [[ -f "$evidence" ]]; then
    lines=$(wc -l < "$evidence")
    if [[ "$lines" -ge "$expected" ]]; then
      echo "[skip] $split evidence already has $lines/$expected lines"
      continue
    fi
    echo "[resume] $split evidence has only $lines/$expected lines; rebuilding"
    rm -f "$evidence"
  fi

  echo "[start] $split $(date -Is)"
  TRANSFORMERS_VERBOSITY=error \
  CBW_EVIDENCE_ONLY=1 \
  CBW_EVIDENCE_OUT="$evidence" \
  "$PYTHON_BIN" cb-whisper.py test \
    --config "$CONFIG" \
    --data.init_args.test_split="$split" \
    --model.init_args.split="$split" \
    --model.init_args.oracle_nbest_diagnostic=false \
    > "$stdout_log" 2>&1
  lines=$(wc -l < "$evidence")
  echo "[done] $split lines=$lines $(date -Is)"
done

echo "[all-done] $(date -Is)"
