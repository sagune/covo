#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp
python_bin=/root/autodl-tmp/great/bin/python
covo_dir="$workspace/cbwhisper_covo_migration_20260609_tar_extracted/covo"
data_dir="$covo_dir/data/processed/stcmds_chinesehp_sensevoice"
log_dir="$workspace/src/logs"
base_adapter="$covo_dir/outputs/qwen35_stcmds_chinesehp_text_1epoch_from_dpo60_20260719"
output_dir="$covo_dir/outputs/qwen35_stcmds_chinesehp_hardbalanced_1epoch_from_full_20260720"

cd "$covo_dir"
"$python_bin" scripts/train_lora_sft.py \
  --train-file "$data_dir/train.hardbalanced.qwen.jsonl" \
  --eval-file "$data_dir/dev.qwen.jsonl" \
  --output-dir "$output_dir" \
  --model-name-or-path "$covo_dir/../models/Qwen3.5-4B" \
  --adapter-path "$base_adapter" \
  --input-format qwen-messages \
  --max-length 1024 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --lr-scheduler-type constant \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --logging-steps 20 \
  --save-steps 1000 \
  --eval-steps 1000 \
  --preprocessing-num-workers 4 \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 2 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking \
  > "$log_dir/train_covo_stcmds_chinesehp_hardbalanced_20260720_stdout.log" 2>&1

for split_name in dev test; do
  "$python_bin" scripts/infer_lora_text.py \
    --input "$data_dir/${split_name}.qwen.jsonl" \
    --output "$log_dir/stcmds_chinesehp_covo_hardbalanced_${split_name}_predictions_20260720.jsonl" \
    --model-name-or-path "$covo_dir/../models/Qwen3.5-4B" \
    --adapter-path "$output_dir" \
    --device auto \
    --batch-size 4 \
    --max-new-tokens 128 \
    --temperature 0 \
    --progress-every 200 \
    --disable-thinking \
    > "$log_dir/stcmds_chinesehp_covo_hardbalanced_${split_name}_infer_20260720_stdout.log" 2>&1
  "$python_bin" scripts/evaluate_correction_jsonl.py \
    --input "$log_dir/stcmds_chinesehp_covo_hardbalanced_${split_name}_predictions_20260720.jsonl" \
    > "$log_dir/stcmds_chinesehp_covo_hardbalanced_${split_name}_metrics_20260720.json"
done
