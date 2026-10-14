#!/usr/bin/env bash
# Isolate the "safe subset" on ST-CMDS: original admission (unchanged) + no trust anchor
# + protect/forced-CTC rerank. This separates the universally-safe changes from the
# relaxed admission, which hurt ST-CMDS.
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
D="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
EV="$R/stcmds_reranked.jsonl"
MSG="$R/e2eSTCMDS_SAFE.messages.jsonl"
PRED="$R/e2eSTCMDS_SAFE.predictions.jsonl"
LOG="$R/stcmds_safe.log"

for i in $(seq 1 240); do
  [ -f "$R/STCMDS_RR_DONE" ] && break
  sleep 60
done
echo "[$(date -Is)] safe arm start" >> "$LOG"

cd "$WS"
if [[ ! -s "$MSG" ]]; then
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$EV" --output "$MSG.inprogress" \
    --max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6 \
    --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin \
    --prompt-mode selector --protect-supported-hotwords --clean-nbest \
    --include-consensus-spans --include-hotword-evidence >> "$LOG" 2>&1
  mv -f "$MSG.inprogress" "$MSG"
fi
cd "$COVO"
if [[ ! -s "$PRED" ]]; then
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$MSG" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADA_ST" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  mv -f "$PRED.inprogress" "$PRED"
fi

cd "$WS"
echo "=== ST-CMDS SAFE SUBSET (original admission + no anchor + rerank) ===" >> "$LOG"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" --label "ST-CMDS safe subset" \
  --aligned "$D/aligned.txt" --uttid-file "$D/uttid" >> "$LOG" 2>&1

touch "$R/STCMDS_SAFE_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
