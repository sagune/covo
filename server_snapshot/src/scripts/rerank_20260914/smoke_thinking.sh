#!/usr/bin/env bash
# Wait for the sampling probe to release the GPU, then smoke-test thinking mode.
set -uo pipefail

WORKSPACE=/root/autodl-tmp
COVO="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/covo"
PY="$WORKSPACE/great/bin/python"
MODEL="$WORKSPACE/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
DATA="$COVO/data/processed/chinesehp_aishell1"
ADAPTER="$COVO/outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022"
OUT="$WORKSPACE/.dsh_checks/thinking"
mkdir -p "$OUT"

for i in $(seq 1 40); do
  [ -f "$WORKSPACE/.dsh_checks/sampling/SAMPLING_DONE" ] && break
  sleep 30
done
echo "sampling_done=$([ -f "$WORKSPACE/.dsh_checks/sampling/SAMPLING_DONE" ] && echo yes || echo no)"
pgrep -f infer_lora_text >/dev/null && { echo "GPU still busy, aborting smoke"; exit 1; }

cd "$COVO"
start=$(date +%s)
PYTORCH_ALLOC_CONF=expandable_segments:True TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$COVO/src" "$PY" "$COVO/scripts/infer_lora_text.py" \
  --input "$DATA/test_text_rewrite_hardneg.qwen.jsonl" \
  --output "$OUT/smoke_standalone_think.jsonl" \
  --model-name-or-path "$MODEL" --adapter-path "$ADAPTER" \
  --batch-size 8 --max-new-tokens 1024 --limit 32 --progress-every 16 2>&1 | tail -5
end=$(date +%s)
echo "SMOKE_SECONDS=$((end-start)) for 32 rows"
