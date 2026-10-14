#!/usr/bin/env bash
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
BS="$WS/.dsh_checks/bootstrap_delta.py"
LOG="$R/bootstrap.log"
: > "$LOG"
run() {
  "$PY" "$BS" --evidence "$2" --ctc-scores "$3" --uttid "$4" --aligned "$5" --label "$1" 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
}
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
run "AISHELL dev / original admission" "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
run "AISHELL dev / relaxed admission" "$R/dev_relax.jsonl" "$R/relax_ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
run "THCHS-30 / original admission" "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
run "THCHS-30 / relaxed admission" "$R/thchs_relax.jsonl" "$R/thchs_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
run "ST-CMDS held-out / original admission" "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt"
echo "[$(date -Is)] BOOTSTRAP_DONE" | tee -a "$LOG"
touch "$R/BOOTSTRAP_DONE"
