#!/usr/bin/env bash
# AISHELL dev re-run with the SAME selector prompt route used for THCHS-30 / ST-CMDS,
# so the three-dataset table shares one prompt family and one adapter family.
set -euo pipefail

WORKSPACE=/root/autodl-tmp
SRC="$WORKSPACE/src"
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WORKSPACE/great/bin/python"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"

OUT="$WORKSPACE/.dsh_checks/aishell_selector"
EVIDENCE="$SRC/logs/hotword_lora_9b_20260908/dev.evidence.jsonl"
INPUT="$OUT/aishell_selector.messages.jsonl"
PRED="$OUT/aishell_9b_aishell_adapter.predictions.jsonl"
LOG="$OUT/infer.log"

mkdir -p "$OUT"
cd "$WORKSPACE"

if [[ ! -s "$INPUT" ]]; then
  "$PY" "$SRC/analysis/cbsensevoice_covo_bridge.py" prepare \
    --input "$EVIDENCE" --output "$INPUT.inprogress" \
    --max-nbest 8 --max-pinyin 5 --max-hotwords 12 \
    --max-prompt-hotwords 6 --max-candidates-with-scores 8 \
    --hotword-source prompt --include-pinyin --prompt-mode selector \
    --protect-supported-hotwords --clean-nbest --include-consensus-spans \
    --include-hotword-evidence --prefer-same-length --trust-asr-top1 \
    --preserve-anchor-digits
  mv -f "$INPUT.inprogress" "$INPUT"
fi
echo "[$(date -Is)] messages rows=$(wc -l < "$INPUT")" >> "$LOG"

cd "$COVO"
echo "[$(date -Is)] START rows=$(wc -l < "$INPUT") adapter=$ADAPTER" >> "$LOG"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
  --input "$INPUT" --output "$PRED.inprogress" \
  --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
  --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1

mv -f "$PRED.inprogress" "$PRED"
echo "[$(date -Is)] DONE rows=$(wc -l < "$PRED")" >> "$LOG"
touch "$OUT/INFER_DONE"
