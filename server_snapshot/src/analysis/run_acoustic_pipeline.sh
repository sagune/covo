#!/usr/bin/env bash
# ============================================================================
# AC-COVO unattended pipeline  (acoustic-conditioned COVO backend)
#
# Everything after "START" runs without any further human or ssh session, so it
# survives the laptop being closed.  Ordered as:
#   P0  wait for the acoustic exports, then LINK and verify the pairing
#   P1  zero-init integration check (must PASS: at init the acoustic model is
#       numerically the text-only model) + span-location failure rate
#   P2  timing pilot (20 steps) -> real ETA for the full epoch
#   P3  full training, checkpoint every 250 steps
#   P4  mid-course: AISHELL-dev end-to-end CER + edit precision per checkpoint
#   P5  one-shot transfer on the best checkpoint: ST-CMDS + THCHS-30
#   P6  summary
#
# Pre-registered gates (RESULTS section 10.12 / 15): edit precision >= 70%,
# net edit characters > 0, designated-hotword recall >= 95.98%, protected-span
# destructions <= 0, ST-CMDS CER < 4.9356%.  If the gate precision lands at the
# ~54% break-even again, that is the pre-declared falsification of the joint-gate
# hypothesis and the run stops without further patches.
# ============================================================================
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
R="$WS/.dsh_checks/rerank"
OUT="$R/acoustic_v1"
LOG="$R/acoustic_pipeline.log"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
ACBIN="$R/ac_aishell_train.bin"
ACIDX="$R/ac_aishell_train.jsonl"
# Defined OUTSIDE the P0 guard: P0 also assigns it, but when P0 is skipped the
# later phases still need it, and `set -u` turns that into a fatal unbound variable.
TRAINLINK="$OUT/train_sft_ac.jsonl"
COMMON=(--max-nbest 8 --max-pinyin 5 --max-hotwords 12 --max-prompt-hotwords 6
        --max-candidates-with-scores 8 --hotword-source prompt --include-pinyin
        --prompt-mode selector --protect-supported-hotwords --clean-nbest
        --include-consensus-spans --include-hotword-evidence --rank-mode transmitted)

mkdir -p "$OUT"
rm -f "$R/ACOUSTIC_FAILED" "$R/ACOUSTIC_PIPELINE_DONE"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
die() { say "FATAL: $*"; touch "$R/ACOUSTIC_FAILED"; exit 1; }

say "================ AC-COVO PIPELINE START ================"
[[ -f "$MODEL/config.json" ]] || die "model not found at $MODEL"
[[ -d "$ADAPTER" ]] || die "adapter not found at $ADAPTER"

# ------------------------------------------------------------------- P0 link
if [[ ! -f "$R/AC_P0_DONE" ]]; then
say "---------- P0: wait for exports and link ----------"
for i in $(seq 1 360); do
  n=$(wc -l < "$ACIDX" 2>/dev/null || echo 0)
  [[ "$n" -ge 17301 ]] && break
  sleep 30
done
say "training acoustic rows: $n"
[[ "$n" -ge 17301 ]] || die "training acoustic export incomplete ($n/17301)"
[[ -s "$ACBIN" ]] || die "acoustic bin missing"

TRAINLINK="$OUT/train_sft_ac.jsonl"
[[ -s "$TRAINLINK" ]] || "$PY" "$WS/src/analysis/link_acoustic_sft.py" \
[[ -s "$TRAINLINK" ]] || die "linking the training file failed"
say "linked training rows: $(wc -l < "$TRAINLINK")"

# eval prompts (reuse the previous run's if present, else rebuild identically)
[[ -s "$OUT/eval_aishell.messages.jsonl" ]] || cp -f "$R/train_aishell_v1/eval_aishell.messages.jsonl" "$OUT/" 2>/dev/null
[[ -s "$OUT/eval_thchs.messages.jsonl" ]] || cp -f "$R/train_aishell_v1/eval_thchs.messages.jsonl" "$OUT/" 2>/dev/null
for f in eval_aishell eval_thchs; do
  if [[ ! -s "$OUT/$f.messages.jsonl" ]]; then
    src="$R/aishell_relax_learned_v2.jsonl"; [[ "$f" == "eval_thchs" ]] && src="$R/thchs_learned_v2.jsonl"
    ( cd "$WS" && "$PY" src/analysis/cbsensevoice_covo_bridge.py prepare \
        --input "$src" --output "$OUT/$f.messages.jsonl" "${COMMON[@]}" ) >> "$LOG" 2>&1
  fi
done
say "eval prompts: aishell=$(wc -l < "$OUT/eval_aishell.messages.jsonl" 2>/dev/null || echo 0) thchs=$(wc -l < "$OUT/eval_thchs.messages.jsonl" 2>/dev/null || echo 0)"

# wait for the eval-set acoustic exports, then link them positionally
# Wait for COMPLETE eval exports, not merely non-empty ones.  `-s` is true as soon
# as a file has any bytes, so an earlier version linked THCHS-30 while its export
# was still being appended to and silently wrote a 216-row file for a 2495-row set.
for i in $(seq 1 240); do
  nd=$(wc -l < "$R/ac_aishell_dev.jsonl" 2>/dev/null || echo 0)
  ns=$(wc -l < "$R/ac_stcmds.jsonl" 2>/dev/null || echo 0)
  nt=$(wc -l < "$R/ac_thchs30.jsonl" 2>/dev/null || echo 0)
  [[ "$nd" -ge 1334 && "$ns" -ge 5130 && "$nt" -ge 2495 ]] && break
  sleep 30
done
say "eval acoustic rows: dev=$nd stcmds=$ns thchs=$nt (need 1334/5130/2495)"
link_eval() {
  local name="$1" rows="$2" idx="$3"
  [[ -s "$idx" ]] || { say "WARN: $name acoustic index missing; $name will run WITHOUT acoustics"; return 0; }
  "$PY" "$WS/src/analysis/link_acoustic_sft.py" --sft "$rows" --ac-index "$idx" \
     --out "${rows%.jsonl}.ac.jsonl" --min-match 0.99 >> "$LOG" 2>&1 \
     || say "WARN: $name acoustic pairing failed verification; will run WITHOUT acoustics"
}
link_eval "aishell_dev" "$OUT/eval_aishell.messages.jsonl" "$R/ac_aishell_dev.jsonl"
link_eval "stcmds"      "$R/e2eSTCMDS_VA.messages.jsonl"    "$R/ac_stcmds.jsonl"
link_eval "thchs30"     "$OUT/eval_thchs.messages.jsonl"    "$R/ac_thchs30.jsonl"
say "P0 done"
touch "$R/AC_P0_DONE"
fi

# --------------------------------------------------- P1 integration check
if [[ ! -f "$R/AC_P1_DONE" ]]; then
say "---------- P1: zero-init integration check ----------"
cd "$COVO"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_acoustic_covo.py" \
  --train-file "$TRAINLINK" --output-dir "$OUT/verify" --model-name-or-path "$MODEL" \
  --adapter-path "$ADAPTER" --acoustic-bin "$ACBIN" --acoustic-index "$ACIDX" \
  --bf16 --gradient-checkpointing --disable-thinking --limit 8 --max-length 1024 \
  --verify-zero-init > "$R/ac_verify.log" 2>&1
VRC=$?
grep -a '"verdict": "PASS"' "$R/ac_verify.log" >/dev/null || {
  cat "$R/ac_verify.log" >> "$LOG"; die "zero-init integration check FAILED (rc=$VRC)"; }
say "zero-init check PASS: $(grep -a verify_zero_init "$R/ac_verify.log" | tail -1)"
touch "$R/AC_P1_DONE"
fi

# ------------------------------------------------------------- P2 pilot
if [[ ! -f "$R/AC_P2_DONE" ]]; then
say "---------- P2: timing pilot (20 steps) ----------"
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_acoustic_covo.py" \
  --train-file "$TRAINLINK" --output-dir "$OUT/pilot" --model-name-or-path "$MODEL" \
  --adapter-path "$ADAPTER" --acoustic-bin "$ACBIN" --acoustic-index "$ACIDX" \
  --bf16 --gradient-checkpointing --disable-thinking --max-steps 20 \
  --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 --save-steps 100000 --logging-steps 1 --limit 400 \
  > "$R/ac_pilot.log" 2>&1
grep -a '"trainer_log"' "$R/ac_pilot.log" | tail -3 >> "$LOG"
SPS=$(grep -aoE '"train_steps_per_second": [0-9.]+' "$R/ac_pilot.log" | tail -1 | grep -oE '[0-9.]+$')
if [[ -n "${SPS:-}" ]]; then
  say "pilot throughput ${SPS} steps/s => ETA for 2163 steps = $(awk -v s="$SPS" 'BEGIN{printf "%.1f h", 2163/s/3600}')"
else
  say "WARN: pilot did not report throughput"
fi
touch "$R/AC_P2_DONE"
fi

# ------------------------------------------------------------- P3 train
say "---------- P3: full training, 1 epoch (2163 steps) ----------"
if [[ ! -s "$OUT/final/acoustic_projector.pt" ]]; then
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/train_acoustic_covo.py" \
    --train-file "$TRAINLINK" --output-dir "$OUT/final" --model-name-or-path "$MODEL" \
    --adapter-path "$ADAPTER" --acoustic-bin "$ACBIN" --acoustic-index "$ACIDX" \
    --bf16 --gradient-checkpointing --disable-thinking --epochs 1 \
    --per-device-train-batch-size 4 --gradient-accumulation-steps 2 \
    --learning-rate 2e-5 --lr-scheduler-type linear --warmup-ratio 0.03 \
    --logging-steps 20 --save-steps 250 > "$R/ac_train.log" 2>&1
  say "training exit=$?"
fi
[[ -s "$OUT/final/acoustic_projector.pt" ]] || die "no projector produced; see ac_train.log"
say "checkpoints: $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null | wc -l)"
touch "$R/AC_P3_DONE"

# ------------------------------------------------- P4 mid-course dev eval
MET="$OUT/dev_metrics.tsv"
[[ -s "$MET" ]] || printf 'checkpoint\tCER\tlabel\n' > "$MET"
dev_infer() {
  local weights="$1" proj="$2" tag="$3"
  local pred="$OUT/dev_$tag.predictions.jsonl"
  [[ -s "$pred" ]] && { say "SKIP dev $tag"; return 0; }
  local rows="$OUT/eval_aishell.messages.jsonl"
  [[ -s "${rows%.jsonl}.ac.jsonl" ]] && rows="${rows%.jsonl}.ac.jsonl"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_acoustic_covo.py" \
    --input "$rows" --output "$pred.inprogress" --model-name-or-path "$MODEL" \
    --adapter-path "$weights" --projector-path "$proj" --acoustic-bin "$R/ac_aishell_dev.bin" \
    --acoustic-index "$R/ac_aishell_dev.jsonl" --batch-size 4 --max-new-tokens 96 \
    --disable-thinking >> "$LOG" 2>&1
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  cd "$WS"
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$pred" --label "AISHELL dev $tag" \
    --aligned "$A/aligned.txt" --uttid-file "$A/uttid" >> "$LOG" 2>&1
  local cer
  # restore_eval prints "CER  7.3109% | ..." with NO equals sign; an earlier
  # version grepped for "CER=" and filled dev_metrics.tsv with `none`.
  cer=$(grep -a "restore\[deployable\]" "$LOG" | tail -1 | grep -oE "CER +[0-9.]+%" | grep -oE "[0-9.]+")
  echo -e "$tag\t${cer:-none}" >> "$MET"
  say "dev eval $tag CER=${cer:-none}"
}
say "---------- P4: mid-course AISHELL-dev evaluation ----------"
for ck in $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  [[ -s "$ck/acoustic_projector.pt" ]] || continue
  dev_infer "$ck" "$ck/acoustic_projector.pt" "$(basename "$ck")"
done
dev_infer "$OUT/final" "$OUT/final/acoustic_projector.pt" "final"
say "dev metrics:"; cat "$MET" >> "$LOG"
touch "$R/AC_P4_DONE"

# ------------------------------------------------------- P5 one-shot transfer
say "---------- P5: one-shot transfer (ST-CMDS + THCHS-30) ----------"
transfer() {
  local name="$1" rows="$2" bin="$3" idx="$4" aligned="$5" uttid="$6"
  local pred="$OUT/${name}.predictions.jsonl"
  [[ -s "$pred" ]] && { say "SKIP $name"; return 0; }
  [[ -s "$rows" ]] || { say "WARN: $name prompt file missing ($rows)"; return 0; }
  local usebin="$bin" useidx="$idx"
  if [[ ! -s "$idx" ]]; then say "WARN: $name has no acoustics; running without"; usebin=""; fi
  cd "$COVO"
  if [[ -n "$usebin" ]]; then
    PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_acoustic_covo.py" \
      --input "$rows" --output "$pred.inprogress" --model-name-or-path "$MODEL" \
      --adapter-path "$OUT/final" --projector-path "$OUT/final/acoustic_projector.pt" \
      --acoustic-bin "$usebin" --acoustic-index "$useidx" \
      --batch-size 4 --max-new-tokens 96 --disable-thinking >> "$LOG" 2>&1
  fi
  [[ -s "$pred.inprogress" ]] && mv -f "$pred.inprogress" "$pred"
  cd "$WS"
  [[ -s "$pred" ]] || { say "WARN: $name produced nothing"; return 0; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$pred" --label "$name (AC-COVO)" \
    --aligned "$aligned" --uttid-file "$uttid" >> "$LOG" 2>&1
}
transfer "stcmds" "${R}/e2eSTCMDS_VA.messages.ac.jsonl" "$R/ac_stcmds.bin" "$R/ac_stcmds.jsonl" "$S/aligned.txt" "$S/uttid"
transfer "thchs30" "$OUT/eval_thchs.messages.ac.jsonl" "$R/ac_thchs30.bin" "$R/ac_thchs30.jsonl" "$T/aligned.txt" "$T/uttid"
touch "$R/AC_P5_DONE"

# ------------------------------------------------------------- P6 summary
say "---------- P6: summary ----------"
say "AC-COVO pipeline finished"
touch "$R/ACOUSTIC_PIPELINE_DONE"
