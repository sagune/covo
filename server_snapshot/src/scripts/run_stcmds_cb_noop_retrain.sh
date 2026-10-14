#!/usr/bin/env bash
set -euo pipefail

workspace="/root/autodl-tmp"
python="${workspace}/great/bin/python"
covo_root="${workspace}/covo"
migration_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted"
source_data="${migration_root}/covo/data/processed/stcmds_chinesehp_sensevoice"
base_adapter="${migration_root}/covo/outputs/qwen35_stcmds_chinesehp_text_1epoch_from_dpo60_20260719"
model="${migration_root}/models/Qwen3.5-4B"
output_dir="${migration_root}/covo/outputs/qwen35_stcmds_chinesehp_noop2_1epoch_20260728"
train_file="${source_data}/train.noop2.qwen.jsonl"
messages="${workspace}/src/logs/stcmds_cb_sensevoice_targeted_chinesehp_messages_20260728.jsonl"
predictions="${workspace}/src/logs/stcmds_cb_sensevoice_targeted_chinesehp_noop2_predictions_full_20260728.jsonl"
metrics="${workspace}/src/logs/stcmds_cb_sensevoice_targeted_chinesehp_noop2_metrics_full_20260728.json"

"${python}" "${workspace}/src/analysis/rebalance_chinesehp_text_sft.py" \
  --input "${source_data}/train.qwen.jsonl" \
  --output "${train_file}" \
  --recoverable-copies 1 \
  --top1-exact-copies 2 \
  --top1-exact-keep-rate 1.0 \
  --seed 20260728

cd "${covo_root}"
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${python}" scripts/train_lora_sft.py \
  --train-file "${train_file}" \
  --eval-file "${source_data}/dev.qwen.jsonl" \
  --output-dir "${output_dir}" \
  --model-name-or-path "${model}" \
  --adapter-path "${base_adapter}" \
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
  --disable-thinking

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${python}" scripts/infer_lora_text.py \
  --input "${messages}" \
  --output "${predictions}" \
  --model-name-or-path "${model}" \
  --adapter-path "${output_dir}" \
  --device auto \
  --batch-size 4 \
  --max-new-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --progress-every 100 \
  --disable-thinking

"${python}" scripts/evaluate_correction_jsonl.py \
  --input "${predictions}" \
  > "${metrics}"
cat "${metrics}"
