#!/usr/bin/env bash
# Policy sweep over the three datasets, all at ORIGINAL admission.
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
SW="$WS/.dsh_checks/rerank_policy_sweep.py"
LOG="$R/policy_sweep.log"
: > "$LOG"

run() {
  local label="$1" ev="$2" ctc="$3" utt="$4" alg="$5"
  echo "================ $label ================" | tee -a "$LOG"
  "$PY" "$SW" --evidence "$ev" --ctc-scores "$ctc" --uttid "$utt" --aligned "$alg" \
    --label "$label" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}

run "AISHELL dev (original admission)" \
  "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" \
  "$R/ctc_scores.jsonl" \
  "$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev/uttid" \
  "$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev/aligned.txt"

run "THCHS-30 test (original admission)" \
  "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" \
  "$R/thchs30_ctc_scores.jsonl" \
  "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid" \
  "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt"

run "ST-CMDS held-out (original admission)" \
  "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
  "$R/stcmds_ctc_scores.jsonl" \
  "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid" \
  "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/aligned.txt"

echo "[$(date -Is)] POLICY_SWEEP_DONE" | tee -a "$LOG"
touch "$R/POLICY_SWEEP_DONE"
