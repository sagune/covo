#!/usr/bin/env bash
# P5: end-to-end acceptance. Bridge the reranked evidence and re-run COVO.
set -euo pipefail

WORKSPACE=/root/autodl-tmp
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WORKSPACE/great/bin/python"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
OUT="$WORKSPACE/.dsh_checks/rerank"
EVID="$OUT/dev_reranked.jsonl"
MSGS="$OUT/dev_reranked_selector.messages.jsonl"
PRED="$OUT/dev_reranked.predictions.jsonl"
LOG="$OUT/p5.log"

cd "$WORKSPACE"
if [[ ! -s "$MSGS" ]]; then
  echo "[$(date -Is)] BRIDGE start" >> "$LOG"
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare \
    --input "$EVID" --output "$MSGS.inprogress" \
    --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 \
    --hotword-source prompt --include-pinyin --prompt-mode selector \
    --protect-supported-hotwords --clean-nbest --include-consensus-spans \
    --include-hotword-evidence --prefer-same-length --trust-asr-top1 \
    --preserve-anchor-digits >> "$LOG" 2>&1
  mv -f "$MSGS.inprogress" "$MSGS"
  echo "[$(date -Is)] BRIDGE done rows=$(wc -l < "$MSGS")" >> "$LOG"
fi

cd "$COVO"
echo "[$(date -Is)] INFER start rows=$(wc -l < "$MSGS")" >> "$LOG"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
  --input "$MSGS" --output "$PRED.inprogress" \
  --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
  --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
mv -f "$PRED.inprogress" "$PRED"
echo "[$(date -Is)] INFER done rows=$(wc -l < "$PRED")" >> "$LOG"
touch "$OUT/P5_DONE"
