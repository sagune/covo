#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/root/autodl-tmp}"
python="${PYTHON_BIN:-${workspace}/great/bin/python}"
covo_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted/covo"
model="${MODEL_PATH:-${workspace}/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B}"
raw_root="${workspace}/src/logs"
data_root="${covo_root}/data/processed/magicdata_original_covo"
output_dir="${OUTPUT_DIR:-${covo_root}/outputs/qwen35_magicdata_hardneg_dropout_balanced_2epoch_20260805}"

mkdir -p "${data_root}" "${output_dir}"

prepare_split() {
  local split="$1"
  local raw="$2"
  local dropout="$3"
  local internal="${data_root}/${split}.internal.jsonl"
  local consensus="${data_root}/${split}.consensus.jsonl"
  local messages="${data_root}/${split}.hardneg.qwen.jsonl"

  "${python}" "${workspace}/src/analysis/convert_nbest_to_covo_internal.py" \
    --input "${raw}" --output "${internal}"
  PYTHONPATH="${covo_root}/src" "${python}" "${covo_root}/scripts/prepare_position_edits.py" \
    --input "${internal}" --output "${consensus}" \
    --add-nbest-consensus --consensus-threshold 0.75 --max-consensus-spans 12
  PYTHONPATH="${covo_root}/src" "${python}" "${covo_root}/scripts/prepare_text_rewrite_data.py" \
    --input "${consensus}" --output "${messages}" \
    --evidence-mode consensus --max-nbest 10 --max-pinyin 5 \
    --include-pinyin --max-hard-negatives 4 --evidence-dropout "${dropout}" --seed 13
}

echo "[$(date -Is)] prepare original-style MAGICDATA train evidence"
prepare_split train "${raw_root}/magicdata_read_train_woitn_nbest10_20260731.jsonl" 0.20
echo "[$(date -Is)] prepare untouched MAGICDATA dev evidence"
prepare_split dev "${raw_root}/magicdata_read_dev_woitn_nbest10_20260731.jsonl" 0.0

# ChineseHP's successful training split was 56.17% already-correct top1. Keep
# every real error and subsample no-op rows to reproduce that learning balance.
balanced="${data_root}/train.hardneg.balanced.qwen.jsonl"
"${python}" "${workspace}/src/analysis/rebalance_chinesehp_text_sft.py" \
  --input "${data_root}/train.hardneg.qwen.jsonl" \
  --output "${balanced}" \
  --recoverable-copies 1 --top1-exact-copies 1 --top1-exact-keep-rate 0.403

echo "[$(date -Is)] start original-style MAGICDATA COVO training"
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/train_lora_sft.py" \
  --train-file "${balanced}" \
  --eval-file "${data_root}/dev.hardneg.qwen.jsonl" \
  --output-dir "${output_dir}" \
  --model-name-or-path "${model}" \
  --input-format qwen-messages \
  --max-length 1024 \
  --epochs 2 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 7 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --logging-steps 20 \
  --save-steps 2000 \
  --eval-steps 2000 \
  --preprocessing-num-workers 16 \
  --dataloader-num-workers 8 \
  --dataloader-prefetch-factor 4 \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --bf16 --gradient-checkpointing --disable-thinking

echo "[$(date -Is)] original-style MAGICDATA COVO training complete"
