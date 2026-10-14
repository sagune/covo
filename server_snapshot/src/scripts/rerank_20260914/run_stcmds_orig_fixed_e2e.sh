#!/usr/bin/env bash
# Cleanest acceptance line for the reranker fix itself: ST-CMDS held-out at the
# ORIGINAL admission (no relaxation at all), fixed reranker, no trust anchor.
# If this holds baseline CER while raising recall, the rerank fix is downstream-safe
# on the one dataset where the shipped reranker broke it.
set -uo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
LOG="$R/stcmds_orig_fixed_e2e.log"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"

for i in $(seq 1 600); do
  [[ -f "$R/FIXED_RERANK_E2E_DONE" ]] && break
  sleep 60
done
echo "[$(date -Is)] start ST-CMDS original+fixed e2e" >> "$LOG"

[[ -s "$R/stcmds_orig_fixed.jsonl" ]] || cp -f "$R/tmp_st.jsonl" "$R/stcmds_orig_fixed.jsonl"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

MSG="$R/e2eSTCMDSORIGFIX.messages.jsonl"
PRED="$R/e2eSTCMDSORIGFIX.predictions.jsonl"
if [[ ! -s "$MSG" ]]; then
  cd "$WS"
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$R/stcmds_orig_fixed.jsonl" \
    --output "$MSG.inprogress" "${COMMON[@]}" >> "$LOG" 2>&1
  mv -f "$MSG.inprogress" "$MSG"
fi
if [[ ! -s "$PRED" ]]; then
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$MSG" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADA_ST" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  mv -f "$PRED.inprogress" "$PRED"
fi
echo "=== ST-CMDS original admission + FIXED rerank, end-to-end ===" >> "$LOG"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" --label "ST-CMDS orig+fixed e2e" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$LOG" 2>&1

touch "$R/STCMDS_ORIG_FIXED_E2E_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
