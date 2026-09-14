#!/usr/bin/env bash
# Idea A: full cross-domain transfer matrix for the learned ranker (CPU only).
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
P="$WS/.dsh_checks/learned_ranker_pilot2.py"
LOG="$R/learned_ranker.log"
: > "$LOG"

A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
AE="$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl"
TE="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"
SE="$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl"

xfer() {  # name, tev, tctc, tutt, talg, trev, trctc, trutt, tralg
  echo "########## $1 ##########" | tee -a "$LOG"
  timeout 900 "$PY" "$P" --evidence "$2" --ctc-scores "$3" --uttid "$4" --aligned "$5" \
    --label "$1" --models lin_ce,lin_ece \
    --train-evidence "$6" --train-ctc "$7" --train-uttid "$8" --train-aligned "$9" 2>&1 \
    | grep -vE "pynvml|FutureWarning|import pynvml" | tee -a "$LOG"
  echo | tee -a "$LOG"
}

echo "########## in-domain CV, 3 seeds (sanity: is the AISHELL gain fold luck?) ##########" | tee -a "$LOG"
for sd in 1 2 3; do
  echo "--- seed $sd ---" | tee -a "$LOG"
  timeout 900 "$PY" "$P" --evidence "$AE" --ctc-scores "$R/ctc_scores.jsonl" \
    --uttid "$A/uttid" --aligned "$A/aligned.txt" --label "AISHELL dev CV seed$sd" \
    --models lin_ce --seed "$sd" 2>&1 | grep -vE "pynvml|FutureWarning|import pynvml" | tee -a "$LOG"
done
echo | tee -a "$LOG"

xfer "AISHELL -> THCHS-30"  "$TE" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt" "$AE" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
xfer "AISHELL -> ST-CMDS"   "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt" "$AE" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt"
xfer "THCHS-30 -> AISHELL"  "$AE" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt" "$TE" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
xfer "THCHS-30 -> ST-CMDS"  "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt" "$TE" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt"
xfer "ST-CMDS -> AISHELL"   "$AE" "$R/ctc_scores.jsonl" "$A/uttid" "$A/aligned.txt" "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt"
xfer "ST-CMDS -> THCHS-30"  "$TE" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt" "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt"

echo "[$(date -Is)] LEARNED_RANKER_DONE" | tee -a "$LOG"
touch "$R/LEARNED_RANKER_DONE"
