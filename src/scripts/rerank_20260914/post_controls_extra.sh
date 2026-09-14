#!/usr/bin/env bash
# Adapter-identity disambiguation, CPU only (safe to run while any GPU arm is busy).
#
# run_controls.sh evaluates the EXISTING **AISHELL** adapter ($ADAPTER_OLD =
# qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022) on the ST-CMDS
# prompt file and labels that block "ST-CMDS, existing adapter", while the surrounding
# prose calls the 4.9570 arm "the control above".  Those are two DIFFERENT adapters:
#
#   e2eSTCMDS_VA.predictions.jsonl  <- ST-CMDS adapter  (checkpoint-34400) -> 4.9570
#   e2eSTCMDS_VC.predictions.jsonl  <- ST-CMDS adapter  (checkpoint-34400) -> 4.9356
#   stcmds_OLD.predictions.jsonl    <- AISHELL adapter  (checkpoint-30022) -> see below
#
# This script measures all of them in one harness invocation with unambiguous labels so
# the trained adapter can be compared against the right baseline for each question:
#   "did training on AISHELL help?"     -> vs the AISHELL-adapter arm
#   "did we beat our best ST-CMDS arm?" -> vs 4.9356 / 4.9570
set -uo pipefail
WS=/root/autodl-tmp
PY="$WS/great/bin/python"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
IDF="$R/compare_adapter_identity.txt"
LOG="$R/adapter_identity.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "============ ADAPTER IDENTITY REPORT START ============"
{
  echo "########################################################################"
  echo "# Which adapter produced which ST-CMDS number  (all policy=restore[deployable])"
  echo "# generated $(date -Is)"
  echo "########################################################################"
  echo
  echo "The three shipped arms differ ONLY in which adapter was loaded."
  echo "Input CER is identical across all of them (5.2722%) because the message"
  echo "file is the same, which is the sanity check that this table is fair."
  echo
} > "$IDF"

ev() {  # predictions, label
  [[ -s "$1" ]] || { echo "# $2 -- MISSING ($1)" >> "$IDF"; return; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" \
    --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$IDF" 2>&1
  "$PY" "$WS/.dsh_checks/covo_error_anatomy.py" --records "$1" --label "$2" >> "$IDF" 2>&1
}

ev "$R/e2eSTCMDS_VA.predictions.jsonl" "ST-CMDS, ST-CMDS adapter (34400), V-A prompt"
ev "$R/e2eSTCMDS_VC.predictions.jsonl" "ST-CMDS, ST-CMDS adapter (34400), V-C prompt  [best: 4.9356]"
ev "$R/e2eSTCMDSFIX.predictions.jsonl" "ST-CMDS, ST-CMDS adapter (34400), relaxed admission + FIXED rerank"
[[ -s "$OUT/stcmds_OLD.predictions.jsonl" ]] && \
  ev "$OUT/stcmds_OLD.predictions.jsonl" "ST-CMDS, AISHELL adapter (30022)  [what run_controls labels 'existing adapter']"
[[ -s "$OUT/stcmds_final.predictions.jsonl" ]] && \
  ev "$OUT/stcmds_final.predictions.jsonl" "ST-CMDS, TRAINED adapter (final), V-A prompt"
[[ -s "$OUT/stcmds_best.predictions.jsonl" ]] && \
  ev "$OUT/stcmds_best.predictions.jsonl" "ST-CMDS, TRAINED adapter (best ckpt), V-A prompt"
[[ -s "$OUT/stcmds_oldiface.predictions.jsonl" ]] && \
  ev "$OUT/stcmds_oldiface.predictions.jsonl" "ST-CMDS, ST-CMDS adapter (34400), ITS OWN training interface"

{
  echo
  echo "### Reference table (hand-maintained, restore[deployable]) ###"
  echo "  baseline (no rescorer, original admission + anchor)   5.0300 / 93.48 / destroyed   0"
  echo "  shipped rescorer, original admission                  6.2657 / 92.90 / destroyed   0"
  echo "  hand fix, original admission                          5.5214 / 92.90 / destroyed   0"
  echo "  relaxed admission + shipped rescorer                  5.1279 / 94.00 / destroyed   0"
  echo "  relaxed admission + FIXED rerank                      see above"
  echo "  learned ranker, old interface                         5.0033 / 95.93 / destroyed   0"
  echo "  learned ranker, V-A interface                         4.9570 / 95.93 / destroyed   0"
  echo "  learned ranker, V-C interface  (best before training)  4.9356 / 95.98 / destroyed   0"
  echo "  TARGET                                                4.5000"
  echo
  echo "### Adapter provenance (verified from training_metadata.json) ###"
  echo "  ST-CMDS adapter : covo/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
  echo "      train_file  : data/processed/stcmds_chinesehp_sensevoice/train.cb-hardnegative.qwen.jsonl (138,180 rows)"
  echo "      system msg  : 100-char 'conservative error corrector ... keep the first candidate'"
  echo "      max_length  : 1024"
  echo "  AISHELL adapter : covo/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
  echo "      train_file  : data/processed/chinesehp_aishell1/train_text_rewrite_hardneg_dropout.qwen.jsonl (120,082 rows)"
  echo "      system msg  : 127-char conservative variant"
  echo "  => the ST-CMDS number was NEVER a clean out-of-domain transfer: the ST-CMDS"
  echo "     adapter saw ST-CMDS training text (though under a different interface)."
} >> "$IDF"
say "identity report written to $IDF"
touch "$R/ADAPTER_IDENTITY_DONE"

for i in $(seq 1 900); do
  [ -f "$R/CONTROLS_DONE" ] && break
  sleep 60
done
{
  echo
  echo "NOTE (cross-reference): the block 'ST-CMDS, existing adapter' above was produced"
  echo "by the **AISHELL** adapter (checkpoint-30022), NOT by the ST-CMDS adapter."
  echo "The 4.9570 / 4.9356 arms came from the ST-CMDS adapter (checkpoint-34400)."
  echo "Both are measured side by side, with explicit labels, in:"
  echo "  $IDF"
} >> "$R/compare_old_vs_new.txt"
say "cross-reference note appended to compare_old_vs_new.txt"
say "============ ADAPTER IDENTITY REPORT DONE ============"
