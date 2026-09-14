#!/usr/bin/env bash
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
SW="$WS/.dsh_checks/rerank_policy_sweep4.py"
LOG="$R/policy_sweep4.log"
: > "$LOG"
run() {
  echo "================ $1 ================" | tee -a "$LOG"
  "$PY" "$SW" --evidence "$2" --ctc-scores "$3" --uttid "$4" --aligned "$5" --label "$1" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}
run "AISHELL dev" \
  "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" "$R/ctc_scores.jsonl" \
  "$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev/uttid" \
  "$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev/aligned.txt"
run "THCHS-30 test" \
  "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" "$R/thchs30_ctc_scores.jsonl" \
  "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid" \
  "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt"
run "ST-CMDS held-out" \
  "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" "$R/stcmds_ctc_scores.jsonl" \
  "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid" \
  "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test/aligned.txt"
echo "[$(date -Is)] POLICY_SWEEP4_DONE" | tee -a "$LOG"
touch "$R/POLICY_SWEEP4_DONE"
