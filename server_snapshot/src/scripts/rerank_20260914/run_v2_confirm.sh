#!/usr/bin/env bash
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
V2="$WS/.dsh_checks/apply_rerank_v2.py"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/v2_confirm.log"
: > "$LOG"

echo "######## original admission ########" | tee -a "$LOG"
"$PY" "$V2" --evidence "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" \
  --ctc-scores "$R/ctc_scores.jsonl" --uttid "$A/uttid" --aligned "$A/aligned.txt" \
  --label "AISHELL dev / original" --out "$R/tmp_ai.jsonl" 2>&1 | tee -a "$LOG"
"$PY" "$V2" --evidence "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" \
  --ctc-scores "$R/thchs30_ctc_scores.jsonl" --uttid "$T/uttid" --aligned "$T/aligned.txt" \
  --label "THCHS-30 / original" --out "$R/tmp_th.jsonl" 2>&1 | tee -a "$LOG"
"$PY" "$V2" --evidence "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
  --ctc-scores "$R/stcmds_ctc_scores.jsonl" --uttid "$S/uttid" --aligned "$S/aligned.txt" \
  --label "ST-CMDS / original" --out "$R/tmp_st.jsonl" 2>&1 | tee -a "$LOG"
echo "### same, with the old shipped policy (no floor) for the ablation" | tee -a "$LOG"
"$PY" "$V2" --evidence "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
  --ctc-scores "$R/stcmds_ctc_scores.jsonl" --uttid "$S/uttid" --aligned "$S/aligned.txt" \
  --label "ST-CMDS / original, floor OFF" --out "$R/tmp_st_nofloor.jsonl" --no-floor 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "######## relaxed admission ########" | tee -a "$LOG"
"$PY" "$V2" --evidence "$R/dev_relax.jsonl" --ctc-scores "$R/relax_ctc_scores.jsonl" \
  --uttid "$A/uttid" --aligned "$A/aligned.txt" \
  --label "AISHELL dev / relaxed" --out "$R/dev_relax_fixed.jsonl" 2>&1 | tee -a "$LOG"
"$PY" "$V2" --evidence "$R/thchs_relax.jsonl" --ctc-scores "$R/thchs_ctc_scores.jsonl" \
  --uttid "$T/uttid" --aligned "$T/aligned.txt" \
  --label "THCHS-30 / relaxed" --out "$R/thchs_relax_fixed.jsonl" 2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "######## spurious-insertion cost (fir_check_generic) ########" | tee -a "$LOG"
"$PY" "$WS/.dsh_checks/fir_check_generic.py" --uttid "$A/uttid" \
  --evidence "AISHELL original=$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" \
  --evidence "AISHELL relaxed=$R/dev_relax.jsonl" \
  --evidence "AISHELL relaxed+fixed=$R/dev_relax_fixed.jsonl" 2>&1 | tee -a "$LOG"
"$PY" "$WS/.dsh_checks/fir_check_generic.py" --uttid "$T/uttid" \
  --evidence "THCHS original=$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" \
  --evidence "THCHS relaxed=$R/thchs_relax.jsonl" \
  --evidence "THCHS relaxed+fixed=$R/thchs_relax_fixed.jsonl" 2>&1 | tee -a "$LOG"
"$PY" "$WS/.dsh_checks/fir_check_generic.py" --uttid "$S/uttid" \
  --evidence "STCMDS original=$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" \
  --evidence "STCMDS orig+fixed=$R/tmp_st.jsonl" \
  --evidence "STCMDS orig+oldpolicynofloor=$R/tmp_st_nofloor.jsonl" 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] V2_CONFIRM_DONE" | tee -a "$LOG"
touch "$R/V2_CONFIRM_DONE"
