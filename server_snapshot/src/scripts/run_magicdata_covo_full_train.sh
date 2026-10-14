#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/root/autodl-tmp}"
python="${PYTHON_BIN:-${workspace}/great/bin/python}"
covo_root="${workspace}/covo"
migration_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted"
model="${MODEL_PATH:-${migration_root}/models/Qwen3.5-4B}"
base_adapter="${BASE_ADAPTER:-${migration_root}/covo/outputs/qwen35_acoustic_listwise_aishell_1epoch_20260729}"
output_dir="${OUTPUT_DIR:-${migration_root}/covo/outputs/qwen35_acoustic_sft_magicdata_full_1epoch_20260731}"
log_root="${workspace}/src/logs"
expected_rows="${EXPECTED_ROWS:-573480}"

shard_glob="${log_root}/magicdata_read_train_woitn_nbest10_shard"
merged="${log_root}/magicdata_read_train_woitn_nbest10_20260731.jsonl"
train_records="${covo_root}/data/processed/acoustic_listwise/magicdata_train_woitn_20260731.jsonl"

count_shards() {
  local total=0
  local path
  for path in "${shard_glob}"{0,1,2,3}_20260731.jsonl; do
    if [[ -f "${path}" ]]; then
      total=$((total + $(wc -l < "${path}")))
    fi
  done
  echo "${total}"
}

echo "[$(date -Is)] waiting for MAGICDATA real N-best generation"
while true; do
  completed="$(count_shards)"
  echo "[$(date -Is)] N-best rows=${completed}/${expected_rows}"
  if [[ "${completed}" -eq "${expected_rows}" ]]; then
    break
  fi
  if [[ "${completed}" -gt "${expected_rows}" ]]; then
    echo "[error] N-best row count exceeds manifest: ${completed}/${expected_rows}" >&2
    exit 1
  fi
  if ! pgrep -f "generate_sensevoice_chinesehp.py.*magicdata_read_train.jsonl" >/dev/null; then
    echo "[error] N-best generators stopped at ${completed}/${expected_rows}" >&2
    exit 1
  fi
  sleep 120
done

temporary="${merged}.tmp"
: > "${temporary}"
for shard in 0 1 2 3; do
  cat "${shard_glob}${shard}_20260731.jsonl" >> "${temporary}"
done
[[ "$(wc -l < "${temporary}")" -eq "${expected_rows}" ]]
mv "${temporary}" "${merged}"

echo "[$(date -Is)] building natural full-sentence COVO records"
mkdir -p "$(dirname "${train_records}")" "${output_dir}"
"${python}" "${workspace}/src/analysis/build_acoustic_listwise_covo.py" \
  --input "${merged}" \
  --output "${train_records}" \
  --max-nbest 10 \
  --max-hotwords 0 \
  --allow-reference-outside-nbest
[[ "$(wc -l < "${train_records}")" -eq "${expected_rows}" ]]

echo "[$(date -Is)] starting full MAGICDATA COVO SFT"
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/train_lora_sft.py" \
  --train-file "${train_records}" \
  --output-dir "${output_dir}" \
  --model-name-or-path "${model}" \
  --adapter-path "${base_adapter}" \
  --input-format qwen-messages \
  --max-length 768 \
  --epochs 1 \
  --learning-rate 3e-7 \
  --lr-scheduler-type constant \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.03 \
  --logging-steps 20 \
  --save-steps 9000 \
  --preprocessing-num-workers 4 \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 2 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking

echo "[$(date -Is)] MAGICDATA COVO full training complete"
