#!/usr/bin/env bash
# Round 2 (cont.): arms D and E — replace the prompt-visible acoustic score with forced-CTC.
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
LOG="$R/round2de.log"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

"$PY" "$WS/.dsh_checks/build_armE.py" >> "$LOG" 2>&1

bridge() {
  local ev="$1" out="$2"
  [[ -s "$out" ]] && return
  cd "$WS"
  "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$ev" --output "$out.inprogress" \
    "${COMMON[@]}" >> "$LOG" 2>&1
  mv -f "$out.inprogress" "$out"
  echo "[$(date -Is)] bridged $(basename "$out") rows=$(wc -l < "$out")" >> "$LOG"
}

infer() {
  local inp="$1" out="$2"
  [[ -s "$out" ]] && return
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$inp" --output "$out.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  mv -f "$out.inprogress" "$out"
  echo "[$(date -Is)] inferred $(basename "$out") rows=$(wc -l < "$out")" >> "$LOG"
}

bridge "$R/dev_ctcscored.jsonl" "$R/armD.messages.jsonl"
infer "$R/armD.messages.jsonl" "$R/armD.predictions.jsonl"
bridge "$R/dev_reranked_ctcscored.jsonl" "$R/armE.messages.jsonl"
infer "$R/armE.messages.jsonl" "$R/armE.predictions.jsonl"

touch "$R/ROUND2DE_DONE"
echo "[$(date -Is)] DONE" >> "$LOG"
