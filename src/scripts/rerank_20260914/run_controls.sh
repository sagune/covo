#!/usr/bin/env bash
# ============================================================================
# Post-training control + comparison, runs after the training plan finishes.
#
# Why this is needed: eval_aishell.messages.jsonl and eval_thchs.messages.jsonl were
# rendered fresh for this plan, so the EXISTING adapter has never been evaluated on
# those exact files.  Without that control, any AISHELL/THCHS difference would be
# confounded with the interface change rather than attributable to training.  For
# ST-CMDS the control already exists (4.9570 on e2eSTCMDS_VA.messages.jsonl), and it
# is re-run here anyway so all three datasets come from one harness invocation.
#
# Produces compare_old_vs_new.txt with, per dataset and per adapter:
#   input / raw / verifier / restore[deployable] / restore[annotation]
# including the beyond-N-best partition (the editing-ability KPI).
# ============================================================================
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
ADAPTER_OLD="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
CMP="$R/compare_old_vs_new.txt"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/controls.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "================ CONTROLS START ================"
for i in $(seq 1 900); do
  { [ -f "$R/TRAIN_PLAN2_DONE" ] || [ -f "$R/TRAIN_FAILED" ]; } && break
  sleep 60
done
say "training plan settled: DONE=$([[ -f "$R/TRAIN_PLAN2_DONE" ]] && echo yes || echo no) FAILED=$([[ -f "$R/TRAIN_FAILED" ]] && echo yes || echo no)"
for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 60
done

infer() {  # messages, predictions, adapter, label
  local msg="$1" pred="$2" ada="$3" label="$4"
  [[ -s "$pred" ]] && { say "SKIP ($label)"; return 0; }
  [[ -s "$msg" ]] || { say "MISSING messages for ($label): $msg"; return 1; }
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  say "infer done ($label) rows=$(wc -l < "$pred" 2>/dev/null || echo 0)"
  cd "$WS"
}
ev() {  # predictions, label, aligned, uttid  -> append to CMP
  [[ -s "$1" ]] || { echo "# $2 (missing)" >> "$CMP"; return; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" \
    --aligned "$3" --uttid-file "$4" >> "$CMP" 2>&1
}

# --- controls with the EXISTING adapter, on the same prompt files -----------
infer "$OUT/eval_aishell.messages.jsonl" "$OUT/dev_OLD.predictions.jsonl" "$ADAPTER_OLD" "AISHELL dev, existing adapter"
infer "$OUT/eval_thchs.messages.jsonl"   "$OUT/thchs_OLD.predictions.jsonl" "$ADAPTER_OLD" "THCHS-30, existing adapter"
infer "$R/e2eSTCMDS_VA.messages.jsonl"   "$OUT/stcmds_OLD.predictions.jsonl" "$ADAPTER_OLD" "ST-CMDS, existing adapter"

# --- compose the report ------------------------------------------------------
{
  echo "########################################################################"
  echo "# Backend training: existing adapter vs trained adapter"
  echo "# generated $(date -Is)"
  echo "########################################################################"
  echo
  echo "measured training throughput: $(grep -aoE 'measured steps/s = [0-9.]+' "$R/train_run_v2.log" | tail -1)"
  echo "checkpoints: $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null | wc -l)"
  echo
  echo "=== AISHELL dev (tuning set) ==========================================="
  echo "--- existing adapter (control on the same prompt file) ---"
} >> "$CMP"
ev "$OUT/dev_OLD.predictions.jsonl" "AISHELL dev, existing adapter" "$A/aligned.txt" "$A/uttid"
for f in "$OUT"/dev_*.predictions.jsonl; do
  case "$f" in *dev_OLD*) continue;; esac
  ev "$f" "AISHELL dev, $(basename "$f" .predictions.jsonl | sed 's/^dev_//')" "$A/aligned.txt" "$A/uttid"
done

{
  echo
  echo "=== ST-CMDS held-out (the target) ====================================="
  echo "--- existing adapter ---"
} >> "$CMP"
ev "$OUT/stcmds_OLD.predictions.jsonl" "ST-CMDS, existing adapter" "$S/aligned.txt" "$S/uttid"
{
  echo "--- trained adapter ---"
} >> "$CMP"
ev "$OUT/stcmds_final.predictions.jsonl" "ST-CMDS, trained adapter" "$S/aligned.txt" "$S/uttid"
{
  echo
  echo "  reference points for ST-CMDS end to end (restore[deployable]):"
  echo "    baseline (no rescorer, original admission + anchor)    5.0300 / 93.48 / destroyed 0"
  echo "    shipped rescorer, original admission                   6.2657 / 92.90 / destroyed 0"
  echo "    hand fix, original admission                           5.5214 / 92.90 / destroyed 0"
  echo "    learned ranker, old interface                          5.0033 / 95.93 / destroyed 0"
  echo "    learned ranker, V-A interface  (= the control above)   4.9570 / 95.93 / destroyed 0"
  echo "    learned ranker, V-C interface  (best so far)           4.9356 / 95.98 / destroyed 0"
  echo "    TARGET                                                 4.5000"
  echo
  echo "=== THCHS-30 (transfer) =============================================="
  echo "--- existing adapter (control on the same prompt file) ---"
} >> "$CMP"
ev "$OUT/thchs_OLD.predictions.jsonl" "THCHS-30, existing adapter" "$T/aligned.txt" "$T/uttid"
{
  echo "--- trained adapter ---"
} >> "$CMP"
ev "$OUT/thchs_final.predictions.jsonl" "THCHS-30, trained adapter" "$T/aligned.txt" "$T/uttid"
{
  echo
  echo "=== how to read this ================================================="
  echo "  * 'input' is the front end COVO receives; it must be identical across"
  echo "    adapters within a dataset (same message file)."
  echo "  * 'restore[deployable]' is the deployable output."
  echo "  * editing-ability KPI = the beyond-N-best partition of that line:"
  echo "    imp/wor counts on rows whose reference is not in the pool."
  echo "    existing adapter on ST-CMDS was imp 94 / wor 22 (+0.178 pp), which is"
  echo "    about half of the backend total gain; if imp collapses the training"
  echo "    traded editing for selection and must be rolled back."
  echo "  * JSON parse failures show up as predictions equal to the input."
} >> "$CMP"

say "report written to $CMP"
touch "$R/CONTROLS_DONE"
say "================ CONTROLS DONE ================"
