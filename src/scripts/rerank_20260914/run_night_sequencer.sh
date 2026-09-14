#!/usr/bin/env bash
# ============================================================================
# Night sequencer: single owner of the GPU after training finishes.
#
# Why one owner.  The arms were originally launched as independent waiters, each
# waiting on the previous one's marker.  That is fine when the plan is fixed, but it
# forces a bad order when the result is bad: if the trained adapter misses the 4.5%
# target, the DPO fallback (the thing the goal explicitly asks for on failure) would
# sit behind three control arms costing ~5 h of GPU - controls, VD arm, old-interface
# arm - none of which can improve the number.
#
#   1. report     as soon as training settles, CPU only (consolidate_train.py)
#   2. ensure     the headline numbers exist, re-running any missing transfer arm
#   3. decide     read the trained adapter's ST-CMDS restore[deployable] CER
#   4. run        target met    -> controls -> VD -> old-interface   (the fixed plan)
#                 target missed -> DPO -> controls -> VD -> old-interface
#   5. repair     if AISHELL dev says `final` was not the best checkpoint, redo the
#                 ST-CMDS transfer with the winner (selection is on AISHELL dev only,
#                 never on ST-CMDS)
#   6. morning    write MORNING_REPORT.txt with the verdict in one screen
# ============================================================================
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
TRLOG="$R/train_run_v2.log"
LOG="$R/sequencer.log"
CMP="$R/compare_old_vs_new.txt"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
TARGET=4.50
BASELINE=4.9356
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

# ------------------------------------------------------------------ single instance
# Two sequencers would both wake on TRAIN_PLAN2_DONE, both grab the GPU, and both write
# the same <pred>.inprogress files - which corrupts arms rather than merely slowing them.
# This happened once (a relaunch without killing the previous copy).
#
# Use flock, not `pgrep -f`: an argv pattern has to be anchored just right or it matches
# the `bash -c` launcher (or its own earlier copy) and then REFUSES TO START for ever -
# which is what a pgrep version of this guard actually did.
#
# -w 120, not -n: bash's `exec 9>file` does not set close-on-exec, so the CHILDREN of a
# killed sequencer inherit the lock fd.  The wait loops spawn `sleep 60`, and `pkill -f
# run_night_sequencer.sh` kills the script but not the sleep, so a dead sequencer can hold
# the lock for up to a minute (observed: pid 182878, a `sleep 60` with ppid 1).  -n then
# refuses to start even though no sequencer exists.  Waiting 120s covers the stale case
# while still refusing a genuine second instance.
LOCK="$R/sequencer.lock"
exec 9>"$LOCK"
if ! flock -w 120 9; then
  say "REFUSING TO START: another run_night_sequencer.sh holds $LOCK after a 120s wait (pid $$)"
  exit 1
fi
say "instance lock acquired ($LOCK, pid $$)"

gpu_wait() {
  for i in $(seq 1 60); do
    busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
    [ -z "$busy" ] && return 0
    say "gpu busy ($busy), waiting"; sleep 60
  done
  return 1
}

infer() {  # messages, predictions, adapter, label
  local msg="$1" pred="$2" ada="$3" label="$4"
  local want got
  want=$(wc -l < "$msg" 2>/dev/null || echo 0)
  # A prediction file is only reusable if it has EVERY row.  `infer_lora_text.py` writes
  # to <pred>.inprogress and the old guard promoted it whenever it was non-empty, so a
  # crash halfway through would be adopted as a finished arm - and every later run would
  # then SKIP it, permanently reporting numbers over a subset.  Compare row counts.
  if [ -s "$pred" ]; then
    got=$(wc -l < "$pred")
    if [ "$got" = "$want" ]; then say "SKIP infer ($label): complete ($got rows)"; return 0; fi
    say "infer ($label): existing file has $got of $want rows - RE-RUNNING"
  fi
  [ -s "$msg" ] || { say "MISSING messages for ($label): $msg"; return 1; }
  [ -d "$ada" ] || { say "MISSING adapter for ($label): $ada"; return 1; }
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  rc=$?
  cd "$WS"
  got=$(wc -l < "$pred.inprogress" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then
    mv -f "$pred.inprogress" "$pred"
    say "infer done ($label) rc=$rc rows=$got"
  else
    say "infer INCOMPLETE ($label) rc=$rc got $got of $want rows - NOT promoting the partial file"
    rm -f "$pred.inprogress"
    return 1
  fi
}

ensure_transfer() {
  # The two one-shot transfer arms must exist, be SCORED, and be COMPLETE.  `infer` now
  # self-guards on the row count, so calling it unconditionally is idempotent and also
  # repairs a partial file left by a crashed phase 4 - gating on `-s` here would have let
  # a half-written file through, because a partial file is non-empty.
  infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_final.predictions.jsonl" "$OUT/final" "ST-CMDS final"
  if [ -s "$OUT/stcmds_final.predictions.jsonl" ] \
     && ! grep -qa "^# ST-CMDS, trained adapter\$" "$TRLOG"; then
    "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS, trained adapter" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$TRLOG" 2>&1
  fi
  infer "$OUT/eval_thchs.messages.jsonl" "$OUT/thchs_final.predictions.jsonl" "$OUT/final" "THCHS final"
  if [ -s "$OUT/thchs_final.predictions.jsonl" ] \
     && ! grep -qa "^# THCHS-30, trained adapter\$" "$TRLOG"; then
    "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/thchs_final.predictions.jsonl" \
      --label "THCHS-30, trained adapter" --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$TRLOG" 2>&1
  fi
}

read_cer() {
  "$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "ST-CMDS, trained adapter" \
    | tr ' ' '\n' | grep '^CER=' | cut -d= -f2
}

say "================ NIGHT SEQUENCER START ================"

# --------------------------------------------------------------- 1. wait + report
for i in $(seq 1 1400); do
  { [ -f "$R/TRAIN_PLAN2_DONE" ] || [ -f "$R/TRAIN_FAILED" ]; } && break
  sleep 60
done
say "training settled: DONE=$([ -f "$R/TRAIN_PLAN2_DONE" ] && echo yes || echo no) FAILED=$([ -f "$R/TRAIN_FAILED" ] && echo yes || echo no)"
"$PY" "$WS/.dsh_checks/consolidate_train.py" --log "$TRLOG" --out "$R/TRAIN_SUMMARY.txt" >> "$LOG" 2>&1
say "TRAIN_SUMMARY.txt written"
touch "$R/POST_REPORT_DONE"

if [ -f "$R/TRAIN_FAILED" ]; then
  { echo "TRAINING FAILED - see $TRLOG and train_out.log"; echo; tail -40 "$TRLOG"; } > "$R/MORNING_REPORT.txt"
  say "TRAIN_FAILED: wrote MORNING_REPORT.txt, starting no GPU arm"
  touch "$R/SEQUENCER_DONE"; exit 0
fi

# --------------------------------------------------------------- 2. ensure numbers
say "---------- ensure the transfer arms exist ----------"
if [ ! -s "$OUT/stcmds_final.predictions.jsonl" ]; then
  gpu_wait || say "gpu never freed for the repair; continuing anyway"
fi
ensure_transfer
say "ST-CMDS trained rows=$(wc -l < "$OUT/stcmds_final.predictions.jsonl" 2>/dev/null || echo 0)"

# --------------------------------------------------------------- 3. decide
VERDICT=$("$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "ST-CMDS, trained adapter")
say "trained-adapter ST-CMDS: $VERDICT"
CER=$(read_cer)
NEED_DPO=1
if [ -z "${CER:-}" ]; then
  say "STILL no trained-adapter ST-CMDS CER after repair -> treating as MISSED"
elif awk -v c="$CER" -v t="$TARGET" 'BEGIN{exit !(c<=t)}'; then
  say "TARGET MET ($CER <= $TARGET) -> planned control arms only"
  NEED_DPO=0
else
  say "TARGET MISSED ($CER > $TARGET, baseline $BASELINE) -> DPO first"
fi

# ------------------------------------------- 3b. EARLY morning report
# The full report is written at the very end, which is ~5h of control arms later (and
# later still if DPO runs).  But the trained adapter's ST-CMDS number - the entire point
# of the night - exists the moment the transfer is scored.  Publish it NOW; the full
# report overwrites this file when the remaining arms finish.
{
  echo "########################################################################"
  echo "# MORNING REPORT (EARLY)   $(date -Is)"
  echo "# target: ST-CMDS end-to-end CER <= $TARGET   (baseline $BASELINE)"
  echo "# NOTE: the FULL report overwrites this file once the remaining arms finish."
  echo "#       Remaining:${NEED_DPO:+ likelihood scorer,}${NEED_DPO:+ DPO,} controls, VD arm, old-interface arm, checkpoint selection."
  echo "########################################################################"
  echo
  echo "== HEADLINE: trained adapter on ST-CMDS =="
  V=$("$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "ST-CMDS, trained adapter" 2>&1)
  echo "  $V"
  C3=$(echo "$V" | tr ' ' '\n' | grep '^CER=' | cut -d= -f2)
  if [ -n "${C3:-}" ] && [ "$C3" != "none" ]; then
    awk -v c="$C3" -v t="$TARGET" -v b="$BASELINE" 'BEGIN{
      printf "  delta vs the 4.9356%% baseline: %+.4f pp   need %+.4f pp more to reach %.2f%%\n", c-b, c-t, t;
      print (c<=t) ? "  VERDICT: TARGET MET" : "  VERDICT: TARGET MISSED";
    }'
  else
    echo "  VERDICT: unknown - evaluation missing, check PHASE4 in train_run_v2.log"
  fi
  echo
  echo "== GATE METRICS (must hold regardless of CER) =="
  echo "  destroyed must stay 0 | recall must not drop below 95.93 (V-A) / 95.98 (V-C)"
  echo "  edit precision must stay >= 70% (baseline V-C: 78.5%); below ~54% more editing LOSES"
  echo "  parse failures must stay 0.000% (baseline: 0 across 10293 rows)"
  if [ -s "$OUT/stcmds_final.predictions.jsonl" ]; then
    "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
    echo "  --- baseline for comparison ---"
    "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$R/e2eSTCMDS_VC.predictions.jsonl" \
      --label "ST-CMDS existing adapter V-C (baseline)" 2>&1
    "$PY" "$WS/.dsh_checks/parse_failures.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
  fi
  echo
  echo "== trained vs existing adapter on ST-CMDS (paired bootstrap + THE BET / REGRESSION) =="
  if [ -s "$OUT/stcmds_final.predictions.jsonl" ] && [ -s "$R/e2eSTCMDS_VC.predictions.jsonl" ]; then
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$R/e2eSTCMDS_VC.predictions.jsonl" \
      --b "$OUT/stcmds_final.predictions.jsonl" --aligned "$S/aligned.txt" --uttid "$S/uttid" \
      --label-a "existing adapter V-C (4.9356)" --label-b "trained adapter" --draws 2000 2>&1
    echo
    echo "== where the trained adapter's gain sits (selection vs editing) =="
    "$PY" "$WS/.dsh_checks/gain_split.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
  fi
  echo
  echo "== every scored arm so far =="
  grep -aE "^# |restore\[deployable\]" "$TRLOG" 2>/dev/null | tail -40
  echo
  echo "== side reports =="
  echo "  $R/TRAIN_SUMMARY.txt   AISHELL-dev checkpoint ranking"
  echo "  $CMP                   all arms once the controls finish"
} > "$R/MORNING_REPORT.txt" 2>&1
say "EARLY morning report written (headline only; full report comes at the end)"

# --------------------------------------------------------------- 4. arms in order
if [ "$NEED_DPO" = "1" ]; then
  # Cheapest informative fallback first (~1 h, no training).  It answers a question DPO
  # cannot: does the trained adapter know WHICH candidate is right, or can it only not
  # say it?  If the scorer reaches the target, the bottleneck is decoding.  If it is no
  # better than the front end, the 9B cannot see the choice at all and DPO will not fix
  # it - worth knowing before spending 3.5 h.  Note a scorer cannot edit, so its
  # beyond-N-best imp is ~0 by construction and must not be read as lost editing ability.
  say "---------- likelihood-scoring diagnostic ----------"
  bash "$WS/.dsh_checks/run_likelihood_arm.sh" >> "$LOG" 2>&1
  rc=$?
  say "likelihood arm finished rc=$rc marker=$([ -f "$R/LIKELIHOOD_DONE" ] && echo yes || echo NO)"
  if [ -s "$OUT/stcmds_likelihood.predictions.jsonl" ]; then
    say "likelihood arm ST-CMDS: $("$PY" "$WS/.dsh_checks/get_cer.py" --log "$CMP" --label-substr "trained adapter SCORER")"
  fi

  say "---------- DPO fallback ----------"
  say "free GPU before DPO: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  # Choose the pair file from the DIAGNOSED failure mode rather than guessing.  RESULTS
  # 10.12/10.13: precision below ~54% means more editing loses, and the training prior
  # (75.4% "change") is 2.1x ST-CMDS's rate (35.7%), so over-editing is the expected
  # failure.  If the trained arm's edit precision came in low, strengthen the keep-it
  # anchors; otherwise use the standard pairs, which teach selection more directly.
  DPO_PAIRS="$R/dpo_pairs_aishell.jsonl"; DPO_WHY="standard"
  if [ -s "$OUT/stcmds_final.predictions.jsonl" ]; then
    PREC=$("$PY" "$WS/.dsh_checks/edit_precision.py" --records "$OUT/stcmds_final.predictions.jsonl" \
             --label "pre-DPO diagnosis" 2>/dev/null \
           | grep -a "EDIT PRECISION" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    RATE=$("$PY" "$WS/.dsh_checks/edit_precision.py" --records "$OUT/stcmds_final.predictions.jsonl" \
             --label "pre-DPO diagnosis" 2>/dev/null \
           | grep -a "edited rows" | grep -oE "[0-9]+\.[0-9]+%" | head -1)
    say "pre-DPO diagnosis: edit rate ${RATE:-?} precision ${PREC:-?}% (baseline 6.9% / 75.3%)"
    if [ -n "${PREC:-}" ] && awk -v p="$PREC" 'BEGIN{exit !(p<60)}'; then
      DPO_PAIRS="$R/dpo_pairs_aishell_keepit50.jsonl"
      DPO_WHY="keep-it weighted (precision ${PREC}% < 60% => over-editing; 49.5% keep-it pairs)"
    fi
  fi
  say "DPO variant: $DPO_WHY -> $(basename "$DPO_PAIRS")"
  export DPO_PAIRS
  bash "$WS/.dsh_checks/run_train_next.sh" dpo >> "$LOG" 2>&1
  rc=$?
  say "DPO finished rc=$rc marker=$([ -f "$R/NEXT_dpo_DONE" ] && echo yes || echo no) using $(basename "$DPO_PAIRS")"
  # surface the cause, so a failed fallback is diagnosable without reading raw logs.
  # train_lora_dpo_text.py holds TWO 9B copies (~36GB of weights) on a 49GB card.
  if [ "$rc" != "0" ]; then
    say "DPO failure evidence:"
    grep -aiE "out of memory|CUDA error|RuntimeError|Traceback|hipped memory" "$R/next_dpo_train.log" 2>/dev/null | tail -6 >> "$LOG"
  fi
  if [ -s "$R/next_dpo/stcmds.predictions.jsonl" ]; then
    "$PY" "$WS/.dsh_checks/gain_split.py" --records "$R/next_dpo/stcmds.predictions.jsonl" \
      --label "ST-CMDS, DPO adapter" >> "$CMP" 2>&1
    "$PY" "$WS/.dsh_checks/covo_error_anatomy.py" --records "$R/next_dpo/stcmds.predictions.jsonl" \
      --label "ST-CMDS, DPO adapter" >> "$CMP" 2>&1
  fi
else
  say "DPO skipped (target met)"
fi

run_arm() {   # script, done-marker, name
  local sc="$1" mk="$2" nm="$3"
  if [ -f "$R/$mk" ]; then say "$nm already done"; return 0; fi
  say "---------- $nm ----------"
  bash "$WS/.dsh_checks/$sc" >> "$LOG" 2>&1
  local rc=$?
  say "$nm finished rc=$rc marker=$([ -f "$R/$mk" ] && echo yes || echo NO)"
}
run_arm run_controls.sh     CONTROLS_DONE  "controls (existing adapter, same prompt files)"
run_arm run_vd_arm.sh       VD_ARM_DONE    "VD arm (trained adapter, rerank= prompt)"
run_arm run_oldiface_arm.sh OLDIFACE_DONE  "old-interface arm (shipped adapter, own template)"

# --------------------------------------------------------------- 5. checkpoint selection
say "---------- checkpoint selection ----------"
# Phase 3's own loop is broken and this repairs it.  It runs
#     ls -d "$OUT"/final/checkpoint-* | sort -t- -k2 -n | tail -2
# but $OUT is ABSOLUTE (/root/autodl-tmp/...), so the field after the FIRST hyphen is the
# "autodl-tmp" component, not the step number: every key ties, the sort is a no-op, and
# `tail -2` returns checkpoint-2000 and checkpoint-500 rather than the LAST two
# (verified on this box).  The goal selects the checkpoint by AISHELL-dev CER, so a
# selection over the wrong subset is not acceptable.  Evaluate ALL checkpoints here;
# `infer` skips existing files and the label guard keeps train_run_v2.log duplicate-free.
for ck in $(ls -d "$OUT"/final/checkpoint-* 2>/dev/null); do
  [ -s "$ck/adapter_model.safetensors" ] || continue
  tag=$(basename "$ck")
  infer "$OUT/eval_aishell.messages.jsonl" "$OUT/dev_$tag.predictions.jsonl" "$ck" "AISHELL dev $tag"
  if [ -s "$OUT/dev_$tag.predictions.jsonl" ] && ! grep -qa "^# AISHELL dev, $tag\$" "$TRLOG"; then
    "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/dev_$tag.predictions.jsonl" \
      --label "AISHELL dev, $tag" --aligned "$A/aligned.txt" --uttid-file "$A/uttid" >> "$TRLOG" 2>&1
  fi
done
"$PY" "$WS/.dsh_checks/consolidate_train.py" --log "$TRLOG" --out "$R/TRAIN_SUMMARY.txt" >> "$LOG" 2>&1
say "re-ranked every checkpoint; TRAIN_SUMMARY.txt refreshed:"
grep -aE "^  [0-9]+\. " "$R/TRAIN_SUMMARY.txt" 2>/dev/null | head -8 >> "$LOG"

BEST=$(grep -a "^BEST_CKPT=" "$R/TRAIN_SUMMARY.txt" 2>/dev/null | head -1 | cut -d= -f2)
VRD=$(grep -a "^VERDICT=" "$R/TRAIN_SUMMARY.txt" 2>/dev/null | head -1 | cut -d= -f2)
say "AISHELL-dev best ckpt: ${BEST:-none} / ${VRD:-none}"
if [ -n "${BEST:-}" ] && [ "${BEST:-}" != "AISHELL dev, final" ] && [ "${VRD:-}" = *"NOT best"* ] \
   && [ ! -s "$OUT/stcmds_best.predictions.jsonl" ]; then
  TAG=$(echo "$BEST" | sed 's/^AISHELL dev, //')
  CKB="$OUT/final/$TAG"; [ -d "$CKB" ] || CKB="$OUT/$TAG"
  if [ -s "$CKB/adapter_model.safetensors" ]; then
    say "re-running ST-CMDS with the AISHELL-dev-selected checkpoint $TAG"
    infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_best.predictions.jsonl" "$CKB" "ST-CMDS best=$TAG"
    [ -s "$OUT/stcmds_best.predictions.jsonl" ] && "$PY" "$WS/.dsh_checks/restore_eval.py" \
      --records "$OUT/stcmds_best.predictions.jsonl" --label "ST-CMDS, BEST ckpt ($TAG)" \
      --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$TRLOG" 2>&1
  else
    say "best checkpoint dir not found: $CKB"
  fi
fi

# --------------------------------------------------------------- 6. morning report
{
  echo "########################################################################"
  echo "# MORNING REPORT   $(date -Is)"
  echo "# target: ST-CMDS end-to-end CER <= $TARGET   (baseline $BASELINE)"
  echo "########################################################################"
  echo
  echo "== trained adapter, ST-CMDS: the headline number =="
  VERDICT=$("$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "ST-CMDS, trained adapter" 2>&1)
  echo "  $VERDICT"
  C2=$(echo "$VERDICT" | tr ' ' '\n' | grep '^CER=' | cut -d= -f2)
  # get_cer.py prints CER=none (rc=1) when the block is absent, and "none" is a non-empty
  # string - without the second test this prints a bogus delta like -4.9356 pp and a
  # "TARGET MISSED" verdict when the truth is that the evaluation never ran.
  if [ -n "${C2:-}" ] && [ "$C2" != "none" ]; then
    awk -v c="$C2" -v t="$TARGET" -v b="$BASELINE" 'BEGIN{
      printf "  delta vs baseline %+.4f pp   need %+.4f pp more to reach %.2f%%\n", c-b, c-t, t;
      print (c<=t) ? "  VERDICT: TARGET MET" : "  VERDICT: TARGET MISSED";
    }'
  else
    echo "  VERDICT: unknown - the trained-adapter evaluation is MISSING, not failed."
    echo "           check PHASE4 / train_out.log before drawing any conclusion."
  fi
  echo
  echo "== every scored arm recorded in train_run_v2.log =="
  grep -aE "^# |restore\[deployable\]" "$TRLOG" 2>/dev/null | tail -50
  echo
  echo "== gate metrics (must hold regardless of CER) =="
  echo "  destroyed            must stay 0"
  echo "  recall               must not drop below 95.93 (V-A) / 95.98 (V-C)"
  echo "  beyond-N-best imp/wor  was 94-98 / 22  -- the editing-ability dashboard"
  echo
  echo "== side reports =="
  echo "  $R/TRAIN_SUMMARY.txt             AISHELL-dev checkpoint ranking"
  echo "  $CMP                             existing vs trained vs DPO, all three datasets"
  echo "  $R/compare_adapter_identity.txt  which adapter produced which ST-CMDS number"
  echo
  if [ -s "$OUT/stcmds_final.predictions.jsonl" ] && [ -s "$R/e2eSTCMDS_VA.predictions.jsonl" ]; then
    echo "== trained vs existing adapter: paired bootstrap + THE BET / REGRESSION =="
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$R/e2eSTCMDS_VA.predictions.jsonl" \
      --b "$OUT/stcmds_final.predictions.jsonl" --aligned "$S/aligned.txt" --uttid "$S/uttid" \
      --label-a "existing adapter" --label-b "trained adapter" --draws 2000 2>&1
    echo
    echo "== where the trained adapter's gain sits (selection vs editing) =="
    "$PY" "$WS/.dsh_checks/gain_split.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
    echo
    echo "== behaviour fingerprint + JSON parse-failure rate (baseline: 0.000%) =="
    "$PY" "$WS/.dsh_checks/parse_failures.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
    echo
    echo "== edit precision vs coverage (THE operating-point check) =="
    echo "   baseline V-C: edit rate 6.7%, precision 78.5%, coverage 13.4%, +158 chars (policy=deployable)"
    echo "   break-even precision is ~51-59%; AISHELL dev does 28.0% edit rate at 75.2% precision"
    echo "   (positive), THCHS-30 does 36.8% at 51.6% (negative) -> safe region is edit rate <~30%."
    echo "   the training prior says 'change' on 75.4% of rows; ST-CMDS only needs 35.7%,"
    echo "   so an edit-rate rise is expected - precision is what must hold (>=70%)."
    "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$OUT/stcmds_final.predictions.jsonl" \
      --label "ST-CMDS trained adapter" 2>&1
    echo "  --- the arm it must beat ---"
    "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$R/e2eSTCMDS_VC.predictions.jsonl" \
      --label "ST-CMDS existing adapter V-C (baseline)" 2>&1
    echo "  --- the same fingerprint for the arm it must beat ---"
    "$PY" "$WS/.dsh_checks/parse_failures.py" --records "$R/e2eSTCMDS_VC.predictions.jsonl" \
      --label "ST-CMDS existing adapter V-C (baseline)" 2>&1
  fi
  if [ -s "$R/next_dpo/stcmds.predictions.jsonl" ]; then
    echo
    echo "== DPO arm vs trained arm =="
    # run_train_next.sh appends its restore_eval blocks to compare_old_vs_new.txt,
    # NOT to next_dpo_train.log (which holds only the training output)
    "$PY" "$WS/.dsh_checks/get_cer.py" --log "$CMP" --label-substr "ST-CMDS, dpo adapter" 2>&1
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$OUT/stcmds_final.predictions.jsonl" \
      --b "$R/next_dpo/stcmds.predictions.jsonl" --aligned "$S/aligned.txt" --uttid "$S/uttid" \
      --label-a "trained adapter" --label-b "DPO adapter" --draws 2000 2>&1
  else
    echo
    echo "== DPO arm DID NOT PRODUCE PREDICTIONS =="
    echo "  marker NEXT_dpo_FAILED=$([ -f "$R/NEXT_dpo_FAILED" ] && echo yes || echo no)"
    grep -aiE "out of memory|CUDA error|RuntimeError|Traceback" "$R/next_dpo_train.log" 2>/dev/null | tail -5
    echo "  (the DPO trainer holds two 9B copies ~36GB of weights; batch is 1 x accum 16)"
  fi
  if [ -s "$OUT/stcmds_likelihood.predictions.jsonl" ]; then
    echo
    echo "== likelihood-scoring arm (same trained adapter, used as a scorer) =="
    echo "  a selector cannot edit: its beyond-N-best imp is ~0 BY CONSTRUCTION."
    "$PY" "$WS/.dsh_checks/get_cer.py" --log "$CMP" --label-substr "trained adapter SCORER" 2>&1
    grep -a "selected_rank" "$LOG" 2>/dev/null | tail -4
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$OUT/stcmds_final.predictions.jsonl" \
      --b "$OUT/stcmds_likelihood.predictions.jsonl" --aligned "$S/aligned.txt" --uttid "$S/uttid" \
      --label-a "trained adapter (generator)" --label-b "trained adapter (scorer)" --draws 2000 2>&1
    # COMBINATION: keep the generator's edits, use the selector only where it abstained.
    # Targets the diagnosed failure (10.12: precise but timid - coverage 13.4%) without
    # giving up the editing that supplies 57% of the backend's value.
    if [ -s "$OUT/stcmds_final.predictions.jsonl" ]; then
      echo
      echo "== COMBINATION: generator's edits + selector where the generator abstained =="
      "$PY" "$WS/.dsh_checks/combine_arms.py" --generator "$OUT/stcmds_final.predictions.jsonl" \
        --selector "$OUT/stcmds_likelihood.predictions.jsonl" \
        --out "$OUT/stcmds_combined.predictions.jsonl" 2>&1
      if [ -s "$OUT/stcmds_combined.predictions.jsonl" ]; then
        "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$OUT/stcmds_combined.predictions.jsonl" \
          --label "ST-CMDS, trained generator + selector on abstention" \
          --aligned "$S/aligned.txt" --uttid-file "$S/uttid" 2>&1
        "$PY" "$WS/.dsh_checks/edit_precision.py" --records "$OUT/stcmds_combined.predictions.jsonl" \
          --label "ST-CMDS combined" 2>&1
        "$PY" "$WS/.dsh_checks/gain_split.py" --records "$OUT/stcmds_combined.predictions.jsonl" \
          --label "ST-CMDS combined" 2>&1
      fi
    fi
  fi
  echo
  echo "########################################################################"
  echo "# the other two datasets: trained vs existing adapter, same prompt file"
  echo "# (goal item (d) asks for all three, each against its own control)"
  echo "########################################################################"
  if [ -s "$OUT/dev_final.predictions.jsonl" ] && [ -s "$OUT/dev_OLD.predictions.jsonl" ]; then
    echo
    echo "== AISHELL dev (the tuning set) =="
    "$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "AISHELL dev, final" 2>&1
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$OUT/dev_OLD.predictions.jsonl" \
      --b "$OUT/dev_final.predictions.jsonl" --aligned "$A/aligned.txt" --uttid "$A/uttid" \
      --label-a "existing adapter" --label-b "trained adapter" --draws 2000 2>&1
  else
    echo "  (AISHELL dev arm missing: dev_final=$([ -s "$OUT/dev_final.predictions.jsonl" ] && echo yes || echo no) dev_OLD=$([ -s "$OUT/dev_OLD.predictions.jsonl" ] && echo yes || echo no))"
  fi
  if [ -s "$OUT/thchs_final.predictions.jsonl" ] && [ -s "$OUT/thchs_OLD.predictions.jsonl" ]; then
    echo
    echo "== THCHS-30 (one-shot transfer) =="
    "$PY" "$WS/.dsh_checks/get_cer.py" --log "$TRLOG" --label-substr "THCHS-30, trained adapter" 2>&1
    "$PY" "$WS/.dsh_checks/compare_arms.py" --a "$OUT/thchs_OLD.predictions.jsonl" \
      --b "$OUT/thchs_final.predictions.jsonl" --aligned "$T/aligned.txt" --uttid "$T/uttid" \
      --label-a "existing adapter" --label-b "trained adapter" --draws 2000 2>&1
    echo
    echo "  THCHS-30 behaviour fingerprint:"
    "$PY" "$WS/.dsh_checks/parse_failures.py" --records "$OUT/thchs_final.predictions.jsonl" \
      --label "THCHS-30 trained adapter" 2>&1
  else
    echo "  (THCHS arm missing: thchs_final=$([ -s "$OUT/thchs_final.predictions.jsonl" ] && echo yes || echo no) thchs_OLD=$([ -s "$OUT/thchs_OLD.predictions.jsonl" ] && echo yes || echo no))"
  fi
} > "$R/MORNING_REPORT.txt" 2>&1

say "MORNING_REPORT.txt written"
touch "$R/SEQUENCER_DONE"
say "================ NIGHT SEQUENCER DONE ================"
