#!/usr/bin/env bash
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
SW="$WS/.dsh_checks/rerank_policy_sweep6.py"
LOG="$R/policy_sweep6.log"
: > "$LOG"
run() {
  echo "================ $1 ================" | tee -a "$LOG"
  "$PY" "$SW" --evidence "$2" --ctc-scores "$3" --uttid "$4" --aligned "$5" --label "$1" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
run "AISHELL dev / original" "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
run "AISHELL dev / relaxed" "$R/dev_relax.jsonl" "$R/relax_ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
run "THCHS-30 / original" "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
run "THCHS-30 / relaxed" "$R/thchs_relax.jsonl" "$R/thchs_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
run "ST-CMDS / original" "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt"
echo "[$(date -Is)] POLICY_SWEEP6_DONE" | tee -a "$LOG"
touch "$R/POLICY_SWEEP6_DONE"
