#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp
python_bin=/root/autodl-tmp/great/bin/python
split_dir="$workspace/datasets/stcmds/splits"
log_dir="$workspace/src/logs"
covo_dir="$workspace/covo"
processed_dir="$covo_dir/data/processed/stcmds_chinesehp_sensevoice"
output_dir="$covo_dir/outputs/qwen35_stcmds_chinesehp_text_1epoch_from_dpo60_20260719"

mkdir -p "$processed_dir" "$output_dir"

generate_split() {
  local split_name=$1
  local shard_count=$2
  local shard_index=$3
  local suffix=""
  if [[ "$shard_count" -gt 1 ]]; then
    suffix="_shard${shard_index}"
  fi
  "$python_bin" "$workspace/src/analysis/generate_sensevoice_chinesehp.py" \
    --input "$split_dir/${split_name}.jsonl" \
    --output "$log_dir/stcmds_sensevoice_chinesehp_${split_name}${suffix}.jsonl" \
    --model iic/SenseVoiceSmall \
    --device cuda:0 \
    --beam-size 32 \
    --token-topk 32 \
    --max-nbest 10 \
    --num-shards "$shard_count" \
    --shard-index "$shard_index" \
    --progress-every 500 \
    --use-itn \
    > "$log_dir/stcmds_sensevoice_chinesehp_${split_name}${suffix}_stdout.log" 2>&1
}

pids=()
for shard_index in 0 1 2 3; do
  generate_split train 4 "$shard_index" &
  pids+=("$!")
done
generate_split dev 1 0 &
pids+=("$!")
generate_split test 1 0 &
pids+=("$!")

for process_id in "${pids[@]}"; do
  wait "$process_id"
done

train_inputs=(
  "$log_dir/stcmds_sensevoice_chinesehp_train_shard0.jsonl"
  "$log_dir/stcmds_sensevoice_chinesehp_train_shard1.jsonl"
  "$log_dir/stcmds_sensevoice_chinesehp_train_shard2.jsonl"
  "$log_dir/stcmds_sensevoice_chinesehp_train_shard3.jsonl"
)

"$python_bin" "$workspace/src/analysis/build_chinesehp_text_sft.py" \
  --input "${train_inputs[@]}" \
  --output "$processed_dir/train.qwen.jsonl" \
  --max-nbest 10
"$python_bin" "$workspace/src/analysis/build_chinesehp_text_sft.py" \
  --input "$log_dir/stcmds_sensevoice_chinesehp_dev.jsonl" \
  --output "$processed_dir/dev.qwen.jsonl" \
  --max-nbest 10
"$python_bin" "$workspace/src/analysis/build_chinesehp_text_sft.py" \
  --input "$log_dir/stcmds_sensevoice_chinesehp_test.jsonl" \
  --output "$processed_dir/test.qwen.jsonl" \
  --max-nbest 10

cd "$covo_dir"
"$python_bin" scripts/train_lora_sft.py \
  --train-file "$processed_dir/train.qwen.jsonl" \
  --eval-file "$processed_dir/dev.qwen.jsonl" \
  --output-dir "$output_dir" \
  --model-name-or-path "$workspace/models/Qwen3.5-4B" \
  --adapter-path outputs/qwen35_cb_sensevoice_oracle_correction_dpo60_20260715 \
  --input-format qwen-messages \
  --max-length 1024 \
  --epochs 1 \
  --learning-rate 1e-6 \
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
  > "$log_dir/train_covo_stcmds_chinesehp_text_from_dpo60_20260719_stdout.log" 2>&1

"$python_bin" scripts/infer_lora_text.py \
  --input "$processed_dir/test.qwen.jsonl" \
  --output "$log_dir/stcmds_chinesehp_covo_from_dpo60_predictions_20260719.jsonl" \
  --model-name-or-path "$workspace/models/Qwen3.5-4B" \
  --adapter-path "$output_dir" \
  --device auto \
  --batch-size 4 \
  --max-new-tokens 128 \
  --temperature 0 \
  --progress-every 200 \
  --disable-thinking \
  > "$log_dir/stcmds_chinesehp_covo_from_dpo60_infer_20260719_stdout.log" 2>&1

"$python_bin" scripts/evaluate_correction_jsonl.py \
  --input "$log_dir/stcmds_chinesehp_covo_from_dpo60_predictions_20260719.jsonl" \
  > "$log_dir/stcmds_chinesehp_covo_from_dpo60_metrics_20260719.json"
