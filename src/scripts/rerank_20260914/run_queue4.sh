#!/usr/bin/env bash
# Queue v4: attack the interface tax on ST-CMDS (target: end-to-end CER 4.5).
#
# Three prompt-consistency variants, same learned rescorer behind all of them:
#   V-A  rank= is the position in the transmitted list, beam provenance under
#        beam_rank=                     -- removes the self-contradiction
#   V-C  V-A plus the rescorer's own score on the winner's line (rerank=)
#   V-B  the rescorer only replaces asr_top1 and leaves the beam's n-best order
#        alone                          -- the most conservative interface
# Then the two deferred items.
set -uo pipefail
WS=/root/autodl-tmp
COVO="$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WS/great/bin/python"
MODEL="$WS/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/queue4_run.log"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }
say "================ QUEUE v4 START ================"
for i in $(seq 1 20); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 120
done
infer() {
  local msg="$1" pred="$2" ada="$3" label="$4"
  if [[ -s "$pred" ]]; then say "SKIP infer ($label)"; return 0; fi
  say "infer START ($label)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$msg" --output "$pred.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ada" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  if [[ -s "$pred.inprogress" ]]; then mv -f "$pred.inprogress" "$pred"; say "infer DONE ($label) rows=$(wc -l < "$pred")"; else say "infer FAILED ($label)"; fi
  cd "$WS"
}
evaluate() {
  [[ -s "$1" ]] || { say "eval SKIP ($2)"; return 0; }
  "$PY" "$WS/.dsh_checks/restore_eval.py" --records "$1" --label "$2" --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$LOG" 2>&1
  say "eval done ($2)"
}
say "---------- V-A: self-consistent ranks ----------"
infer "$R/e2eSTCMDS_VA.messages.jsonl" "$R/e2eSTCMDS_VA.predictions.jsonl" "$ADA_ST" "ST-CMDS V-A transmitted ranks"
evaluate "$R/e2eSTCMDS_VA.predictions.jsonl" "ST-CMDS V-A (transmitted ranks) e2e"
say "---------- V-C: self-consistent ranks + rerank score ----------"
infer "$R/e2eSTCMDS_VC.messages.jsonl" "$R/e2eSTCMDS_VC.predictions.jsonl" "$ADA_ST" "ST-CMDS V-C transmitted + rerank score"
evaluate "$R/e2eSTCMDS_VC.predictions.jsonl" "ST-CMDS V-C (transmitted + rerank score) e2e"
say "---------- V-B: no n-best reorder ----------"
infer "$R/e2eSTCMDS_VB.messages.jsonl" "$R/e2eSTCMDS_VB.predictions.jsonl" "$ADA_ST" "ST-CMDS V-B no reorder"
evaluate "$R/e2eSTCMDS_VB.predictions.jsonl" "ST-CMDS V-B (no n-best reorder) e2e"
touch "$R/STCMDS_INTERFACE_VARIANTS_DONE"
say "---------- deferred: breadth split ----------"
if [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]]; then
  say "breadth settled, skipping"
else
  nohup bash "$WS/.dsh_checks/run_stcmds_breadth.sh" >> "$LOG" 2>&1 &
  for i in $(seq 1 480); do
    [[ -f "$R/STCMDS_BREADTH_DONE" || -f "$R/STCMDS_BREADTH_FAILED" ]] && break
    sleep 60
  done
  say "breadth finished"
fi
say "---------- deferred: same-code baseline ----------"
bash "$WS/.dsh_checks/run_samecode_baselines.sh" >> "$LOG" 2>&1
say "samecode done: $([[ -f "$R/SAMECODE_DONE" ]] && echo yes || echo no)"
touch "$R/QUEUE4_DONE"
say "================ QUEUE v4 DONE ================"
