#!/usr/bin/env bash
# Reusability ablation for the learned ranker (CPU only, minutes).
#
# Q1 does the module still work if the forced-CTC features are removed?  Those need
#    an extra acoustic forward pass at inference, i.e. a real deployment cost.
# Q2 how few features suffice?  A smaller feature set is easier to explain and less
#    likely to break when the domain shifts.
# Q3 does the AISHELL-trained model still transfer without the CTC features?
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
P="$WS/.dsh_checks/learned_ranker_pilot2.py"
LOG="$R/ranker_ablation.log"
: > "$LOG"

A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
AE="$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl"
TE="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl"
SE="$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl"

cv() {  # tag, drop
  echo "########## CV AISHELL dev  [drop: ${2:-none}] ##########" | tee -a "$LOG"
  timeout 900 "$PY" "$P" --evidence "$AE" --ctc-scores "$R/ctc_scores.jsonl" \
    --uttid "$A/uttid" --aligned "$A/aligned.txt" --label "CV AISHELL ${2:-full}" \
    --models lin_ce --drop-features "$2" 2>&1 | grep -vE "pynvml|FutureWarning|import pynvml" | tee -a "$LOG"
  echo | tee -a "$LOG"
}
xf() {  # tag, test-ev, test-ctc, test-utt, test-alg, drop
  echo "########## TRANSFER AISHELL -> $1  [drop: ${6:-none}] ##########" | tee -a "$LOG"
  timeout 900 "$PY" "$P" --evidence "$2" --ctc-scores "$3" --uttid "$4" --aligned "$5" \
    --label "$1" --models lin_ce --drop-features "$6" \
    --train-evidence "$AE" --train-ctc "$R/ctc_scores.jsonl" --train-uttid "$A/uttid" --train-aligned "$A/aligned.txt" \
    2>&1 | grep -vE "pynvml|FutureWarning|import pynvml" | tee -a "$LOG"
  echo | tee -a "$LOG"
}

# Q1: no forced-CTC features at all (only the model's own logged scores + ranks)
cv "no-ctc" "ctc_,ntok"
xf "ST-CMDS no-ctc"    "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt" "ctc_,ntok"
xf "THCHS-30 no-ctc"   "$TE" "$R/thchs30_ctc_scores.jsonl" "$T/uttid" "$T/aligned.txt" "ctc_,ntok"

# Q2: minimal set -- the three things a human would use
cv "minimal" "ctc_,ntok,asr_rank,asr_margin,search,total,hotword,phon,consensus"
xf "ST-CMDS minimal"   "$SE" "$R/stcmds_ctc_scores.jsonl" "$S/uttid" "$S/aligned.txt" "ctc_,ntok,asr_rank,asr_margin,search,total,hotword,phon,consensus"

# Q3: no per-utterance edit-distance feature (the only one needing the hypothesis text)
cv "no-ed_old" "ed_old"

echo "[$(date -Is)] RANKER_ABLATION_DONE" | tee -a "$LOG"
touch "$R/RANKER_ABLATION_DONE"
