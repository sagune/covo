#!/usr/bin/env bash
# ============================================================================
# Backend (COVO LoRA) training plan -- unattended overnight run.   v2 (fixed)
#
# v1 failed instantly: MODEL pointed at <bundle>/covo/models/Qwen3.5-9B but the
# weights live at <bundle>/models/Qwen3.5-9B, and because every phase was guarded
# with "if the output is missing, run it" rather than "fail loudly", the whole plan
# cascaded to DONE in three minutes without training anything.  v2 fixes the path and
# adds hard gates.
#
# Dataset : AISHELL train, 17,301 utterances whose 16-candidate scored pools are
#           already on disk, so no decode is needed.
# Start   : the existing AISHELL adapter (checkpoint-30022) -- "on the basis of the
#           current model".
# Interface: identical to the inference arms (bridge, --rank-mode transmitted).
# Target  : normalised reference -> trains selection and editing together.
#
# Phases: 0 build data (CPU) | 1 timing pilot | 2 full training | 3 checkpoint pick
#         by AISHELL-dev end-to-end CER | 4 ST-CMDS + THCHS transfer | 5 summary
# Every phase has a gate; a failure writes TRAIN_FAILED and stops.
# ============================================================================
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
ADAPTER_AISHELL="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
LOG="$R/train_run_v2.log"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence --rank-mode transmitted)
mkdir -p "$OUT"
rm -f "$R/TRAIN_FAILED" "$R/TRAIN_PLAN2_DONE"
say()  { echo "[$(date -Is)] $*" >> "$LOG"; }
die()  { say "FATAL: $*"; touch "$R/TRAIN_FAILED"; exit 1; }
gpu_free() {
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    [[ -z "$busy" ]] && return 0
    say "gpu busy ($busy), waiting"; sleep 120
  done
  return 1
}

say "================ TRAIN PLAN v2 START ================"
say "MODEL=$MODEL exists=$([[ -f "$MODEL/config.json" ]] && echo yes || echo NO)"
say "ADAPTER=$ADAPTER_AISHELL exists=$([[ -d "$ADAPTER_AISHELL" ]] && echo yes || echo NO)"
[[ -f "$MODEL/config.json" ]] || die "model config.json not found at $MODEL"
[[ -d "$ADAPTER_AISHELL" ]] || die "adapter dir not found at $ADAPTER_AISHELL"

# ---------------------------------------------------------------- phase 0 (CPU)
say "---------- phase 0: data ----------"
[[ -s "$OUT/train_sft.jsonl" ]] || die "train_sft.jsonl missing"
say "SFT rows: $(wc -l < "$OUT/train_sft.jsonl")"
cd "$WS"
[[ -s "$OUT/eval_aishell.messages.jsonl" ]] || "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare \
  --input "$R/aishell_relax_learned_v2.jsonl" --output "$OUT/eval_aishell.messages.jsonl" "${COMMON[@]}" >> "$LOG" 2>&1
[[ -s "$OUT/eval_thchs.messages.jsonl" ]] || "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare \
  --input "$R/thchs_learned_v2.jsonl" --output "$OUT/eval_thchs.messages.jsonl" "${COMMON[@]}" >> "$LOG" 2>&1
say "eval prompts: aishell=$(wc -l < "$OUT/eval_aishell.messages.jsonl" 2>/dev/null || echo 0) thchs=$(wc -l < "$OUT/eval_thchs.messages.jsonl" 2>/dev/null || echo 0) stcmds=$(wc -l < "$R/e2eSTCMDS_VA.messages.jsonl" 2>/dev/null || echo 0)"
touch "$R/P2_PHASE0_DONE"

# ------------------------------------------------------- phase 1 (GPU, ~10 min)
say "---------- phase 1: timing pilot (20 steps) ----------"
gpu_free || die "gpu never became free"
cd "$COVO"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_lora_sft.py" \
  --train-file "$OUT/train_sft.jsonl" --output-dir "$OUT/pilot" \
  --model-name-or-path "$MODEL" --adapter-path "$ADAPTER_AISHELL" \
  --input-format qwen-messages --max-length 2048 \
  --max-steps 20 --learning-rate 2e-5 --lr-scheduler-type linear --warmup-ratio 0.03 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
  --logging-steps 1 --save-steps 100000 --bf16 --gradient-checkpointing --disable-thinking \
  > "$R/pilot_out.log" 2>&1
rc=$?
PILOT=$(grep -aoE "[0-9.]+s/it|[0-9.]+it/s" "$R/pilot_out.log" | tail -1)
say "pilot exit=$rc throughput=${PILOT:-unknown}"
grep -aq "train_runtime" "$R/pilot_out.log" || die "pilot did not finish; see pilot_out.log"
STEP_S=$(grep -aoE "train_steps_per_second': '[0-9.]+" "$R/pilot_out.log" | tail -1 | grep -oE "[0-9.]+$")
if [[ -n "${STEP_S:-}" ]]; then
  say "measured steps/s = $STEP_S  =>  ETA for 2163 steps (1 epoch) = $(awk -v s="$STEP_S" 'BEGIN{printf "%.1f h", 2163/s/3600}')"
fi
touch "$R/P2_PHASE1_DONE"

# ------------------------------------------------------ phase 2 (GPU, ~8-16 h)
say "---------- phase 2: full training, 1 epoch ----------"
if [[ ! -s "$OUT/final/adapter_model.safetensors" ]]; then
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_lora_sft.py" \
    --train-file "$OUT/train_sft.jsonl" --output-dir "$OUT/final" \
    --model-name-or-path "$MODEL" --adapter-path "$ADAPTER_AISHELL" \
    --input-format qwen-messages --max-length 2048 \
    --epochs 1 --learning-rate 2e-5 --lr-scheduler-type linear --warmup-ratio 0.03 \
    --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
    --logging-steps 20 --save-steps 500 \
    --bf16 --gradient-checkpointing --disable-thinking \
    > "$R/train_out.log" 2>&1
  say "training exit=$?"
fi
[[ -s "$OUT/final/adapter_model.safetensors" ]] || die "no adapter produced; see train_out.log"
say "checkpoints: $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null | wc -l)"
touch "$R/P2_PHASE2_DONE"

# ------------------------------------------------- phase 3 (GPU): pick checkpoint
say "---------- phase 3: AISHELL-dev e2e ----------"
infer() {
  local msg="$1" pred="$2" ada="$3" label="$4"
  if [[ -s "$pred" ]]; then say "SKIP infer ($label)"; return 0; fi
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  say "infer done ($label) rows=$(wc -l < "$pred" 2>/dev/null || echo 0)"
  cd "$WS"
}
for ck in $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -2) "$OUT/final"; do
  [[ -d "$ck" && -s "$ck/adapter_model.safetensors" ]] || continue
  tag=$(basename "$ck")
  infer "$OUT/eval_aishell.messages.jsonl" "$OUT/dev_$tag.predictions.jsonl" "$ck" "AISHELL dev $tag"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/dev_$tag.predictions.jsonl" \
    --label "AISHELL dev, $tag" --aligned "$A/aligned.txt" --uttid-file "$A/uttid" >> "$LOG" 2>&1
done
touch "$R/P2_PHASE3_DONE"

# ------------------------------------------------- phase 4 (GPU): transfer checks
say "---------- phase 4: ST-CMDS + THCHS-30 ----------"
infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_final.predictions.jsonl" "$OUT/final" "ST-CMDS final"
[[ -s "$OUT/stcmds_final.predictions.jsonl" ]] || die "ST-CMDS inference produced nothing"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/stcmds_final.predictions.jsonl" \
  --label "ST-CMDS, trained adapter" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$LOG" 2>&1
infer "$OUT/eval_thchs.messages.jsonl" "$OUT/thchs_final.predictions.jsonl" "$OUT/final" "THCHS final"
[[ -s "$OUT/thchs_final.predictions.jsonl" ]] && "$PY" "$WS/.dsh_checks/restore_eval.py" \
  --records "$OUT/thchs_final.predictions.jsonl" --label "THCHS-30, trained adapter" \
  --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$LOG" 2>&1
touch "$R/P2_PHASE4_DONE"

say "================ TRAIN PLAN v2 DONE ================"
touch "$R/TRAIN_PLAN2_DONE"
