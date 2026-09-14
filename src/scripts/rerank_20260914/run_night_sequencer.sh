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
  [ -s "$pred" ] && { say "SKIP infer ($label): exists"; return 0; }
  [ -s "$msg" ] || { say "MISSING messages for ($label): $msg"; return 1; }
  [ -d "$ada" ] || { say "MISSING adapter for ($label): $ada"; return 1; }
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [ -s "$pred.inprogress" ] && mv -f "$pred.inprogress" "$pred"
  cd "$WS"
  say "infer done ($label) rows=$(wc -l < "$pred" 2>/dev/null || echo 0)"
}

ensure_transfer() {
  # the two one-shot transfer arms of the trained adapter must exist and be scored
  if [ ! -s "$OUT/stcmds_final.predictions.jsonl" ]; then
    infer "$R/e2eSTCMDS_VA.messages.jsonl" "$OUT/stcmds_final.predictions.jsonl" "$OUT/final" "ST-CMDS final"
    [ -s "$OUT/stcmds_final.predictions.jsonl" ] && "$PY" "$WS/.dsh_checks/restore_eval.py" \
      --records "$OUT/stcmds_final.predictions.jsonl" --label "ST-CMDS, trained adapter" \
      --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$TRLOG" 2>&1
  fi
  if [ ! -s "$OUT/thchs_final.predictions.jsonl" ]; then
    infer "$OUT/eval_thchs.messages.jsonl" "$OUT/thchs_final.predictions.jsonl" "$OUT/final" "THCHS final"
    [ -s "$OUT/thchs_final.predictions.jsonl" ] && "$PY" "$WS/.dsh_checks/restore_eval.py" \
      --records "$OUT/thchs_final.predictions.jsonl" --label "THCHS-30, trained adapter" \
      --aligned "$T/aligned.txt" --uttid-file "$T/uttid" >> "$TRLOG" 2>&1
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
  bash "$WS/.dsh_checks/run_train_next.sh" dpo >> "$LOG" 2>&1
  rc=$?
  say "DPO finished rc=$rc marker=$([ -f "$R/NEXT_dpo_DONE" ] && echo yes || echo no)"
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

# --------------------------------------------------------------- 5. repair
say "---------- checkpoint selection ----------"
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
  if [ -n "${C2:-}" ]; then
    awk -v c="$C2" -v t="$TARGET" -v b="$BASELINE" 'BEGIN{
      printf "  delta vs baseline %+.4f pp   need %+.4f pp more to reach %.2f%%\n", c-b, c-t, t;
      print (c<=t) ? "  VERDICT: TARGET MET" : "  VERDICT: TARGET MISSED";
    }'
  else
    echo "  VERDICT: unknown (evaluation missing)"
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
