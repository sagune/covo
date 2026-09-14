#!/usr/bin/env bash
# THCHS-30 end-to-end acceptance, run in parallel with the long ST-CMDS decode.
# e2eTHCHS.messages.jsonl is already bridged (no trust anchor) from thchs_relax_reranked.jsonl.
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_AI="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
LOG="$R/thchs_e2e.log"
MSG="$R/e2eTHCHS.messages.jsonl"
PRED="$R/e2eTHCHS.predictions.jsonl"

if [[ ! -s "$PRED" ]]; then
  echo "[$(date -Is)] THCHS e2e infer START" >> "$LOG"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$MSG" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADA_AI" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  mv -f "$PRED.inprogress" "$PRED"
  echo "[$(date -Is)] THCHS e2e infer DONE rows=$(wc -l < "$PRED")" >> "$LOG"
fi

echo "=== THCHS-30 end-to-end (new config: relax + no anchor + protect/CTC rerank) ===" >> "$LOG"
"$PY" "$WS/.dsh_checks/restore_eval.py" \
  --records "$PRED" --label "THCHS e2e new-config" \
  --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$LOG" 2>&1

touch "$R/THCHS_E2E_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
