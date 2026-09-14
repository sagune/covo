#!/usr/bin/env bash
# Safety net for the overnight training chain.
#
# Two jobs, deliberately separated so neither delays the other:
#   1. REPORT (CPU only, starts the moment training ends) - consolidates the per-arm
#      restore_eval blocks in train_run_v2.log into TRAIN_SUMMARY.txt.  The orchestrator
#      never gates on restore_eval.py, so if an evaluation silently crashed this is where
#      it becomes visible instead of the plan just announcing DONE.
#   2. REPAIR (GPU, after the whole chain is idle) - re-runs any arm whose predictions
#      file or result line is missing, and if AISHELL dev says `final` was not the best
#      checkpoint, re-runs the ST-CMDS/THCHS transfer with the winner.
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
LOG="$R/train_run_v2.log"
PLOG="$R/post_train.log"
CMP="$R/compare_old_vs_new.txt"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
say() { echo "[$(date -Is)] $*" >> "$PLOG"; }

say "============ POST-TRAIN WATCHER START ============"
for i in $(seq 1 1400); do
  { [ -f "$R/TRAIN_PLAN2_DONE" ] || [ -f "$R/TRAIN_FAILED" ]; } && break
  sleep 60
done
if [ -f "$R/TRAIN_FAILED" ]; then
  say "TRAIN_FAILED present - reporting what exists, no repair"
else
  say "training done; consolidating"
fi
"$PY" "$WS/.dsh_checks/consolidate_train.py" --log "$LOG" --out "$R/TRAIN_SUMMARY.txt" >> "$PLOG" 2>&1
say "summary written"
touch "$R/POST_REPORT_DONE"

# ---------------------------------------------------------------- repair (GPU)
for i in $(seq 1 600); do
  [ -f "$R/OLDIFACE_DONE" ] && break
  [ -f "$R/TRAIN_FAILED" ] && { say "train failed; skipping repair"; exit 1; }
  sleep 60
done
for i in $(seq 1 40); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 60
done
[[ -s "$OUT/final/adapter_model.safetensors" ]] || { say "no adapter; nothing to repair"; exit 1; }

infer() {
  local msg="$1" pred="$2" ada="$3" label="$4"
  if [[ -s "$pred" ]]; then say "SKIP ($label) exists"; return 0; fi
  say "infer START ($label) adapter=$ada"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$PLOG" 2>&1
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  say "infer done ($label) rows=$(wc -l < "$pred" 2>/dev/null || echo 0)"
  cd "$WS"
}

# 1. missing transfer arms
if [[ ! -s "$OUT/stcmds_final.predictions.jsonl" ]]; then
  infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_final.predictions.jsonl" "$OUT/final" "ST-CMDS final (repair)"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/stcmds_final.predictions.jsonl" \
    --label "ST-CMDS, trained adapter (repaired)" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1
fi
if [[ ! -s "$OUT/thchs_final.predictions.jsonl" ]]; then
  infer "$OUT/eval_thchs.messages.jsonl" "$OUT/thchs_final.predictions.jsonl" "$OUT/final" "THCHS final (repair)"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/thchs_final.predictions.jsonl" \
    --label "THCHS-30, trained adapter (repaired)" --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$CMP" 2>&1
fi

# 2. if AISHELL dev says a checkpoint beat `final`, redo the transfer with it
BEST=$(grep -a "^BEST_CKPT=" "$R/TRAIN_SUMMARY.txt" 2>/dev/null | head -1 | cut -d= -f2)
VERDICT=$(grep -a "^VERDICT=" "$R/TRAIN_SUMMARY.txt" 2>/dev/null | head -1 | cut -d= -f2)
say "best ckpt from AISHELL dev: ${BEST:-none} (verdict: ${VERDICT:-none})"
if [[ -n "${BEST:-}" && "${BEST:-}" != "AISHELL dev, final" && "${VERDICT:-}" == *"NOT best"* ]]; then
  TAG=$(echo "$BEST" | sed 's/^AISHELL dev, //')
  CKB="$OUT/final/$TAG"
  [[ -d "$CKB" ]] || CKB="$OUT/$TAG"
  if [[ -s "$CKB/adapter_model.safetensors" ]]; then
    say "re-running ST-CMDS with $TAG"
    infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_best.predictions.jsonl" "$CKB" "ST-CMDS best=$TAG"
    "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/stcmds_best.predictions.jsonl" \
      --label "ST-CMDS, BEST ckpt ($TAG)" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1
    "$PY" "$WS/.dsh_checks/covo_error_anatomy.py" --records "$OUT/stcmds_best.predictions.jsonl" \
      --label "ST-CMDS, BEST ckpt ($TAG)" >> "$CMP" 2>&1
  else
    say "best ckpt dir not found: $CKB"
  fi
else
  say "final stands; no re-run needed"
fi

"$PY" "$WS/.dsh_checks/consolidate_train.py" --log "$LOG" --out "$R/TRAIN_SUMMARY.txt" >> "$PLOG" 2>&1
say "============ POST-TRAIN WATCHER DONE ============"
touch "$R/POST_TRAIN_DONE"
