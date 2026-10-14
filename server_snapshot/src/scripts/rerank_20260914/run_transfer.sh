#!/usr/bin/env bash
# Transfer check: forced-CTC scoring + reranking for ST-CMDS and THCHS-30 (front-end only).
set -euo pipefail

WS=/root/autodl-tmp
PY="$WS/great/bin/python"
SCORER="$WS/src/analysis/score_sensevoice_candidate_evidence.py"
R="$WS/.dsh_checks/rerank"
LOG="$R/transfer.log"

score_one() {
  local name="$1" input="$2" manifest="$3" out="$4"
  if [[ -s "$out" ]]; then
    echo "[$(date -Is)] $name scores already present ($(wc -l < "$out") rows)" >> "$LOG"
    return
  fi
  echo "[$(date -Is)] $name scoring start" >> "$LOG"
  PYTHONPATH="$WS/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    "$PY" "$SCORER" --input "$input" --manifest "$manifest" --output "$out" \
    --max-nbest 16 --progress-every 200 >> "$LOG" 2>&1
  echo "[$(date -Is)] $name scoring done ($(wc -l < "$out") rows)" >> "$LOG"
}

score_one stcmds "$R/stcmds_ctc_input.jsonl" "$R/stcmds_manifest.jsonl" "$R/stcmds_ctc_scores.jsonl"
score_one thchs30 "$R/thchs30_ctc_input.jsonl" "$R/thchs30_manifest.jsonl" "$R/thchs30_ctc_scores.jsonl"

echo "[$(date -Is)] rerank start" >> "$LOG"
{
  "$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
    --evidence "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
    --ctc-scores "$R/stcmds_ctc_scores.jsonl" \
    --uttid "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid" \
    --aligned "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/aligned.txt" \
    --label "ST-CMDS held-out (front-end transfer)" \
    --out "$R/stcmds_reranked.jsonl"
  "$PY" "$WS/.dsh_checks/apply_rerank_generic.py" \
    --evidence "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" \
    --ctc-scores "$R/thchs30_ctc_scores.jsonl" \
    --uttid "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid" \
    --aligned "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt" \
    --label "THCHS-30 test (front-end transfer)" \
    --out "$R/thchs30_reranked.jsonl"
} 2>&1 | tee -a "$LOG"

touch "$R/TRANSFER_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
