#!/usr/bin/env bash
# Round 2: let the language side select. Two arms, single-variable vs the round-1 control.
#
#   A (control, already measured): baseline evidence + trust-asr-top1 ON
#   B: baseline evidence, trust-asr-top1 OFF  (and the two same-length/digit anchors OFF)
#   C: reranked evidence (protect + forced-CTC order), same flags as B
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
LOG="$R/round2.log"

# flags shared by both arms: no trust anchor, no same-length/digit anchors
COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

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

B_EV="$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl"
C_EV="$R/dev_reranked.jsonl"
B_MSG="$R/armB.messages.jsonl"; B_PRED="$R/armB.predictions.jsonl"
C_MSG="$R/armC.messages.jsonl"; C_PRED="$R/armC.predictions.jsonl"

bridge "$B_EV" "$B_MSG"
infer "$B_MSG" "$B_PRED"
bridge "$C_EV" "$C_MSG"
infer "$C_MSG" "$C_PRED"

touch "$R/ROUND2_DONE"
echo "[$(date -Is)] ROUND2 DONE" >> "$LOG"
