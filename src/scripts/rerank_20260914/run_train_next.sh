#!/usr/bin/env bash
# ============================================================================
# One-command follow-up runner, so whichever branch the first SFT result implies
# can start immediately.  Usage:  run_train_next.sh {dpo|epoch2|selector}
#
#   dpo       preference pairs (chosen = reference, rejected = front-end top-1, or
#             the pool's best on rows the pool cannot fix) against the trained adapter
#   epoch2    continue SFT for a second epoch from the trained adapter
#   selector  train the candidate-selector ablation (target {"choice": k}) and map
#             the indices back to text before evaluating
#
# All three evaluate on the same three prompt files and append to the same report, so
# results stay comparable with the first run and with the existing-adapter controls.
# ============================================================================
set -uo pipefail
MODE="${1:-}"
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
ADAPTER_OLD="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
BASE="$R/train_aishell_v1"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/train_next_${MODE}.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
[[ -n "$MODE" ]] || { echo "usage: $0 {dpo|epoch2|selector}"; exit 2; }
[[ -f "$R/TRAIN_PLAN2_DONE" ]] || { echo "the first run has not finished yet"; exit 2; }
[[ -s "$BASE/final/adapter_model.safetensors" ]] || { echo "no trained adapter to build on"; exit 2; }

say "================ NEXT RUN ($MODE) ================"
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 120
done
cd "$COVO"

OUT="$R/next_$MODE"
mkdir -p "$OUT"
case "$MODE" in
  dpo)
    PAIRS="$R/dpo_pairs_aishell.jsonl"
    [[ -s "$PAIRS" ]] || { say "FATAL: $PAIRS missing"; exit 1; }
    # max-prompt-length must exceed the real prompt length or the system message is
    # truncated away (default 896 does exactly that: measured 98.2% of prompts exceed
    # it, and the truncation is from the LEFT).
    #
    # Memory: train_lora_dpo_text.py loads TWO full copies of the 9B model (policy and
    # reference, ~18GB each = ~36GB of weights) on a 49GB card, so the micro-batch has
    # to stay small.  1 x 16 keeps the effective batch at 16 exactly as 2 x 8 did, and
    # max-length 2048 still covers the longest pair (measured max prompt 1690 tokens
    # + a ~20-token completion), so this is purely a memory change.
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_lora_dpo_text.py" \
      --train-file "$PAIRS" --output-dir "$OUT/adapter" \
      --model-name-or-path "$MODEL" --adapter-path "$BASE/final" \
      --max-prompt-length 2048 --max-length 2048 \
      --max-steps 800 --learning-rate 5e-6 --beta 0.1 --sft-weight 0.1 \
      --per-device-train-batch-size 1 --gradient-accumulation-steps 16 \
      --warmup-steps 50 --logging-steps 20 --save-steps 200 \
      --bf16 --gradient-checkpointing --disable-thinking > "$R/next_dpo_train.log" 2>&1
    ADA="$OUT/adapter"
    ;;
  epoch2)
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_lora_sft.py" \
      --train-file "$BASE/train_sft.jsonl" --output-dir "$OUT/adapter" \
      --model-name-or-path "$MODEL" --adapter-path "$BASE/final" \
      --input-format qwen-messages --max-length 2048 \
      --epochs 1 --learning-rate 1e-5 --lr-scheduler-type linear --warmup-ratio 0.01 \
      --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
      --logging-steps 20 --save-steps 500 \
      --bf16 --gradient-checkpointing --disable-thinking > "$R/next_epoch2_train.log" 2>&1
    ADA="$OUT/adapter"
    ;;
  selector)
    SEL="$R/train_aishell_selector.jsonl"
    [[ -s "$SEL" ]] || { say "FATAL: $SEL missing"; exit 1; }
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_lora_sft.py" \
      --train-file "$SEL" --output-dir "$OUT/adapter" \
      --model-name-or-path "$MODEL" --adapter-path "$ADAPTER_OLD" \
      --input-format qwen-messages --max-length 2048 \
      --epochs 1 --learning-rate 2e-5 --lr-scheduler-type linear --warmup-ratio 0.03 \
      --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
      --logging-steps 20 --save-steps 500 \
      --bf16 --gradient-checkpointing --disable-thinking > "$R/next_selector_train.log" 2>&1
    ADA="$OUT/adapter"
    # the selector needs its own prompt file and an index->text mapping step
    "$PY" "$WS/src/analysis/cbsensevoice_covo_bridge.py" prepare --input "$R/train_aishell_evidence.jsonl" \
      --output "$OUT/ignore.jsonl" --max-nbest 16 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6 \
      --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin --prompt-mode selector \
      --protect-supported-hotwords --clean-nbest --include-consensus-spans --include-hotword-evidence \
      --rank-mode transmitted >> "$LOG" 2>&1 || true
    ;;
  *) say "FATAL: unknown mode $MODE"; exit 2;;
esac
say "training exit=$? ; adapter=$ADA exists=$([[ -s "$ADA/adapter_model.safetensors" ]] && echo yes || echo NO)"
[[ -s "$ADA/adapter_model.safetensors" ]] || { say "FATAL: no adapter produced"; touch "$R/NEXT_${MODE}_FAILED"; exit 1; }

infer() {
  local msg="$1" pred="$2" ad="$3" label="$4"
  [[ -s "$pred" ]] && { say "SKIP ($label)"; return 0; }
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ad" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  cd "$WS"
}
ev() {
  [[ -s "$1" ]] || { echo "# $2 (missing)" >> "$R/compare_old_vs_new.txt"; return; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" --aligned "$3" --uttid-file "$4" \
    >> "$R/compare_old_vs_new.txt" 2>&1
}

say "---------- evaluation ----------"
infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds.predictions.jsonl" "$ADA" "ST-CMDS $MODE"
ev "$OUT/stcmds.predictions.jsonl" "ST-CMDS, $MODE adapter" "$S/aligned.txt" "$S/uttid"
infer "$BASE/eval_aishell.messages.jsonl" "$OUT/aishell.predictions.jsonl" "$ADA" "AISHELL dev $MODE"
ev "$OUT/aishell.predictions.jsonl" "AISHELL dev, $MODE adapter" "$A/aligned.txt" "$A/uttid"
infer "$BASE/eval_thchs.messages.jsonl" "$OUT/thchs.predictions.jsonl" "$ADA" "THCHS $MODE"
ev "$OUT/thchs.predictions.jsonl" "THCHS-30, $MODE adapter" "$T/aligned.txt" "$T/uttid"

say "================ NEXT RUN ($MODE) DONE ================"
touch "$R/NEXT_${MODE}_DONE"
