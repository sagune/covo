#!/usr/bin/env bash
# Control arm: the SHIPPED ST-CMDS adapter scored through ITS OWN training interface.
#
# Why this arm exists
# -------------------
# The shipped adapter qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817 was trained with
#   system  你是一个保守的中文ASR后纠错器 ... 证据不足时保持第一候选   (100 chars)
#   user    N-best文本/N-best拼音 plain listing, tail 请输出：{"text":...}
# Every arm in this session instead feeds it a 165-char "候选选择器 ... 不要默认保守复制
# top-1" system message and a restructured evidence block (ASR top-1 / protected hotwords /
# reliability labels / stable spans / scores).  Two of those four changes invert the
# objective.  Its 93.3% no_change rate on ST-CMDS is therefore not evidence that COVO
# cannot select - it is evidence that the adapter is obeying "keep the first candidate".
#
# This arm holds the adapter and the utterances fixed and changes ONLY the interface, so
# it separates interface calibration from model capability:
#     existing adapter + old interface   (this arm)
#     existing adapter + new interface   (V-A, 4.9570 / V-C, 4.9356)
#     trained  adapter + new interface   (run_train2 phase 4)
#
# Runs last in the chain so it never delays the main line.
set -uo pipefail
WS=/root/autodl-tmp
BUNDLE="$WS/cbwhisper_covo_migration_20260609_tar_extracted"
COVO="$BUNDLE/covo"
PY="$WS/great/bin/python"
MODEL="$BUNDLE/models/Qwen3.5-9B"
ADA_ST="$COVO/outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400"
R="$WS/.dsh_checks/rerank"
OUT="$R/train_aishell_v1"
CMP="$R/compare_old_vs_new.txt"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
LOG="$R/oldiface_arm.log"
MSGS="$OUT/stcmds_oldiface.messages.jsonl"
PRED="$OUT/stcmds_oldiface.predictions.jsonl"
say() { echo "[$(date -Is)] $*" >> "$LOG"; }

say "============ OLD-INTERFACE ARM START ============"
for i in $(seq 1 1200); do
  { [ -f "$R/VD_ARM_DONE" ] || [ -f "$R/TRAIN_FAILED" ]; } && break
  sleep 60
done
for i in $(seq 1 30); do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')
  [[ -z "$busy" ]] && break
  say "gpu busy ($busy), waiting"; sleep 60
done
[[ -d "$ADA_ST" ]] || { say "shipped adapter missing: $ADA_ST"; exit 1; }

if [[ ! -s "$MSGS" ]]; then
  "$PY" "$WS/.dsh_checks/render_old_iface.py" --records "$R/e2eSTCMDS_VA.messages.jsonl" \
    --style stcmds --out "$MSGS" >> "$LOG" 2>&1 || { say "render FAILED"; exit 1; }
  say "rendered $(wc -l < "$MSGS") rows"
fi

if [[ ! -s "$PRED" ]]; then
  say "infer START (shipped adapter, old interface)"
  cd "$COVO"
  PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
    --input "$MSGS" --output "$PRED.inprogress" \
    --model-name-or-path "$MODEL" --adapter-path "$ADA_ST" \
    --batch-size 8 --max-new-tokens 96 --progress-every 256 --disable-thinking >> "$LOG" 2>&1
  [[ -s "$PRED.inprogress" ]] && mv -f "$PRED.inprogress" "$PRED"
  say "infer done rows=$(wc -l < "$PRED" 2>/dev/null || echo 0)"
  cd "$WS"
fi

{
  echo
  echo "=== ST-CMDS: shipped adapter, OLD training interface (single-variable control) ==="
} >> "$CMP"
"$PY" "$WS/.dsh_checks/restore_eval.py" --records "$PRED" \
  --label "ST-CMDS shipped adapter + old interface" \
  --aligned "$S/aligned.txt" --uttid-file "$S/uttid" >> "$CMP" 2>&1
"$PY" "$WS/.dsh_checks/covo_error_anatomy.py" --records "$PRED" \
  --label "ST-CMDS shipped adapter + old interface" >> "$CMP" 2>&1

say "============ OLD-INTERFACE ARM DONE ============"
touch "$R/OLDIFACE_DONE"
