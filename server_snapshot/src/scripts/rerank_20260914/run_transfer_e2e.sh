#!/usr/bin/env bash
# Transfer acceptance, end-to-end: does the new configuration (relaxed admission +
# no trust anchor + protect/forced-CTC rerank) hold up on THCHS-30 and ST-CMDS?
set -euo pipefail

WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
R="$WS/.dsh_checks/rerank"
LOG="$R/transfer_e2e.log"
ADA_AI="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"

COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence)

# wait for any CB-SenseVoice decode to release the GPU (cap 8 h, then warn)
for i in $(seq 1 960); do
  pgrep -f "run_CLI.py test" >/dev/null || break
  sleep 30
done
if pgrep -f "run_CLI.py test" >/dev/null; then
  echo "[$(date -Is)] WARNING: decode still running after the wait cap; proceeding anyway" >> "$LOG"
fi
echo "[$(date -Is)] gpu free, starting e2e" >> "$LOG"

run_arm() {
  local ev="$1" tag="$2" adapter="$3"
  local msg="$R/e2e$tag.messages.jsonl" pred="$R/e2e$tag.predictions.jsonl"
  [[ -s "$ev" ]] || { echo "[$(date -Is)] skip $tag: no $ev" >> "$LOG"; return; }
  if [[ ! -s "$msg" ]]; then
    cd "$WS"
    "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare --input "$ev" --output "$msg.inprogress" \
      "${COMMON[@]}" >> "$LOG" 2>&1
    mv -f "$msg.inprogress" "$msg"
  fi
  if [[ ! -s "$pred" ]]; then
    cd "$COVO"
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
      --input "$msg" --output "$pred.inprogress" \
      --model-name-or-path "$MODEL" --adapter-path "$adapter" \
      --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
    mv -f "$pred.inprogress" "$pred"
  fi
  echo "[$(date -Is)] $tag ready rows=$(wc -l < "$pred")" >> "$LOG"
}

run_arm "$R/thchs_relax_reranked.jsonl" THCHS "$ADA_AI"
run_arm "$R/stcmds_relax.jsonl" STCMDS "$ADA_ST"

touch "$R/TRANSFER_E2E_DONE"
echo "[$(date -Is)] ALL DONE" >> "$LOG"
