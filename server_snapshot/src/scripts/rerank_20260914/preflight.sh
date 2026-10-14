#!/usr/bin/env bash
# Pre-flight: verify every input and process the morning depends on, in one screen.
#
# The failure this guards against is an input that goes missing NOW and is only discovered
# hours later when an arm dies - by which time the head start is gone.  Safe to run at any
# time; it only reads.
set -uo pipefail
WS=/root/autodl-tmp
B="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
PY="$WS/great/bin/python"
ok=0; bad=0
chk() {  # label, path
  if [ -s "$2" ]; then printf "  OK    %-34s %s\n" "$1" "$(basename "$2")"; ok=$((ok+1))
  else printf "  MISS  %-34s %s\n" "$1" "$2"; bad=$((bad+1)); fi
}
chkd() {
  if [ -d "$2" ]; then printf "  OK    %-34s %s\n" "$1" "$(basename "$2")"; ok=$((ok+1))
  else printf "  MISS  %-34s %s\n" "$1" "$2"; bad=$((bad+1)); fi
}

echo "===== PRE-FLIGHT $(date -Is) ====="

echo "-- model and adapters --"
chk  "base model config"        "$B/models/Qwen3.5-9B/config.json"
chkd "AISHELL adapter (start)"  "$B/covo/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
chkd "ST-CMDS adapter (control)" "$B/covo/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"

echo "-- prompts the arms will use --"
chk  "ST-CMDS V-A prompts"      "$R/e2eSTCMDS_VA.messages.jsonl"
chk  "ST-CMDS V-C control preds" "$R/e2eSTCMDS_VC.predictions.jsonl"
chk  "ST-CMDS VD prompts"       "$R/e2eSTCMDS_VD.messages.jsonl"
chk  "AISHELL dev prompts"      "$OUT/eval_aishell.messages.jsonl"
chk  "THCHS-30 prompts"         "$OUT/eval_thchs.messages.jsonl"

echo "-- reference files (recall/destruction depend on these) --"
for d in "$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test" \
         "$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test" \
         "$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"; do
  chk "$(basename "$(dirname "$d")")/aligned" "$d/aligned.txt"
  chk "$(basename "$(dirname "$d")")/uttid"   "$d/uttid"
done

echo "-- training artefacts --"
chk  "SFT training file"        "$OUT/train_sft.jsonl"
chk  "training log"             "$R/train_out.log"
chk  "orchestrator log"         "$R/train_run_v2.log"
chkd "final adapter dir"        "$OUT/final"

echo "-- fallback data --"
chk  "DPO pairs (standard)"     "$R/dpo_pairs_aishell.jsonl"
chk  "DPO pairs (keep-it)"      "$R/dpo_pairs_aishell_keepit50.jsonl"
chk  "selector SFT data"        "$R/train_aishell_selector.jsonl"

echo "-- scripts the sequencer calls --"
for s in run_controls.sh run_vd_arm.sh run_oldiface_arm.sh run_likelihood_arm.sh \
         run_train_next.sh consolidate_train.py get_cer.py compare_arms.py edit_precision.py \
         gain_split.py parse_failures.py combine_arms.py covo_error_anatomy.py restore_eval.py; do
  chk "$s" "$WS/.dsh_checks/$s"
done

echo "-- syntax of the orchestrators --"
for s in run_night_sequencer.sh watch_sequencer.sh run_train2.sh run_train_next.sh; do
  if bash -n "$WS/.dsh_checks/$s" 2>/dev/null; then echo "  OK    $s"; ok=$((ok+1))
  else echo "  FAIL  $s  (bash -n)"; bad=$((bad+1)); fi
done

echo "-- processes --"
printf "  trainer        : %s\n" "$(pgrep -cf 'bash .dsh_checks/run_train2.sh')"
printf "  sequencer      : %s (must be 1)\n" "$(ps -eo args | grep -c '^bash .dsh_checks/run_night_sequencer.sh$')"
printf "  watchdog       : %s\n" "$(ps -eo args | grep -c '^bash .dsh_checks/watch_sequencer.sh$')"
printf "  gpu            : %s\n" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"

echo "-- markers --"
for m in P2_PHASE0_DONE P2_PHASE1_DONE P2_PHASE2_DONE P2_PHASE3_DONE P2_PHASE4_DONE TRAIN_PLAN2_DONE TRAIN_FAILED SEQUENCER_DONE; do
  [ -f "$R/$m" ] && echo "  SET   $m"
done
if [ -f "$R/SEQUENCER_DONE" ] && [ ! -f "$R/TRAIN_PLAN2_DONE" ]; then
  echo "  WARNING: SEQUENCER_DONE exists but training has not finished -> STALE, remove it"
  bad=$((bad+1))
fi
[ -f "$R/TRAIN_FAILED" ] && { echo "  WARNING: TRAIN_FAILED present"; bad=$((bad+1)); }

echo "-- disk --"
df -h "$WS" | tail -1
df -h /tmp | tail -1

echo
echo "===== $ok checks OK, $bad problem(s) ====="
