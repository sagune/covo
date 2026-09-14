#!/usr/bin/env bash
# Verification gap: the THCHS-30 transfer check compared the relaxed run against the
# 2026-08-11 evidence without proving the decode path is unchanged since then.
# This reruns THCHS-30 with config defaults and checks reproduction.
set -euo pipefail

WS=/root/autodl-tmp
SRC="$WS/src"
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
LOG="$R/thchs_base.log"
ORIG="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"
BASE="$R/thchs_base.jsonl"

cd "$SRC"
if [[ ! -s "$BASE" ]]; then
  echo "[$(date -Is)] thchs BASE (config defaults) START" >> "$LOG"
  CBW_EVIDENCE_ONLY=1 CBW_EVIDENCE_OUT="$BASE" \
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$PY" run_CLI.py test --config configs/cb-sensevoice-thchs30-error-hotwords.yaml \
    --data.init_args.num_workers=0 \
    --model.init_args.oracle_nbest_diagnostic=false \
    --model.init_args.oracle_nbest_detail_path="$R/thchs_base_oracle_detail.csv" \
    --model.init_args.oracle_nbest_summary_path="$R/thchs_base_oracle_summary.csv" >> "$LOG" 2>&1
  echo "[$(date -Is)] thchs BASE DONE rows=$(wc -l < "$BASE" 2>/dev/null || echo 0)" >> "$LOG"
fi

echo "=== reproduction check: THCHS-30 base rerun vs 2026-08-11 evidence ===" >> "$LOG"
md5sum "$ORIG" "$BASE" >> "$LOG" 2>&1
if cmp -s "$ORIG" "$BASE"; then
  echo "BYTE-IDENTICAL: no code drift; the THCHS transfer delta is attributable to the relaxed parameters" >> "$LOG"
else
  echo "DIFFERS: report the metric delta below and note the drift caveat" >> "$LOG"
  "$PY" "$WS/.dsh_checks/compare_pool_generic.py" \
    --uttid "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid" \
    --aligned "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt" \
    --label "THCHS-30 base reproduction" \
    --evidence "original(08-11)=$ORIG" --evidence "base-rerun=$BASE" >> "$LOG" 2>&1
fi

touch "$R/THCHS_BASE_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
