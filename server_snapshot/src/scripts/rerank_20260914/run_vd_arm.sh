#!/usr/bin/env bash
# Extra arm: the trained adapter on the VD prompt (the one that carries rerank=).
#
# Why: training used the V-A prompt (self-consistent ranks, no rerank= field), and the
# best ST-CMDS number so far (4.9356) came from the EXISTING adapter on the VD prompt
# (e2eSTCMDS_VC).  Without this arm the trained adapter could only be compared with
# 4.9570 (V-A), and we would not know whether the rescorer-score field still helps
# once the adapter has been trained on the same interface family.
#
# Control for this exact file already exists: the existing adapter on the VD prompt is
# the V-C arm, 4.9356 / 95.98 / destroyed 0.
#
# Waits for CONTROLS_DONE so it never contends with the control runs.
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
CMP="$R/compare_old_vs_new.txt"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/vd_arm.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "================ VD ARM START ================"
for i in $(seq 1 900); do
  { [ -f "$R/CONTROLS_DONE" ] || [ -f "$R/TRAIN_FAILED" ]; } && break
  sleep 60
done
[[ -s "$OUT/final/adapter_model.safetensors" ]] || { say "no trained adapter; aborting"; exit 1; }
for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 60
done

PRED="$OUT/stcmds_vd.predictions.jsonl"
if [[ ! -s "$PRED" ]]; then
  say "infer START (trained adapter on the VD prompt)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$R/e2eSTCMDS_VD.messages.jsonl" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$OUT/final" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [[ -s "$PRED.inprogress" ]] && mv -f "$PRED.inprogress" "$PRED"
  say "infer done rows=$(wc -l < "$PRED" 2>/dev/null || echo 0)"
  cd "$WS"
fi

{
  echo
  echo "=== ST-CMDS on the VD prompt (rerank= present) ======================="
  echo "--- existing adapter  (this is the 4.9356 arm) ---"
} >> "$CMP"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$R/e2eSTCMDS_VC.predictions.jsonl" \
  --label "ST-CMDS VD, existing adapter" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1
{
  echo "--- trained adapter ---"
} >> "$CMP"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" \
  --label "ST-CMDS VD, trained adapter" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1

say "================ VD ARM DONE ================"
touch "$R/VD_ARM_DONE"
