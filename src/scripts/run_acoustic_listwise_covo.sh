#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/root/autodl-tmp}"
src_root="${workspace}/src"
python="${PYTHON_BIN:-${workspace}/great/bin/python}"
covo_root="${workspace}/covo"
migration_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted"
model="${MODEL_PATH:-${migration_root}/models/Qwen3.5-4B}"
base_adapter="${BASE_ADAPTER:-${migration_root}/covo/outputs/qwen35_stcmds_cb_hardnegative_1epoch_20260728}"
output_dir="${OUTPUT_DIR:-${migration_root}/covo/outputs/qwen35_acoustic_listwise_aishell_1epoch_20260729}"
wait_pid="${WAIT_PID:-}"

train_evidence_prefix="${src_root}/logs/cb_sensevoice_w14_train_evidence"
train_evidence="${train_evidence_prefix}_full.jsonl"
train_records="${covo_root}/data/processed/acoustic_listwise/aishell_train.jsonl"
test_evidence="${src_root}/logs/cb_sensevoice_stcmds_standard3139_full_20260729.jsonl"
test_records="${src_root}/logs/stcmds_standard3139_acoustic_listwise_messages_20260729.jsonl"
prompt_predictions="${src_root}/logs/stcmds_standard3139_acoustic_prompt_predictions_20260729.jsonl"
prompt_metrics="${src_root}/logs/stcmds_standard3139_acoustic_prompt_metrics_20260729.json"
listwise_predictions="${src_root}/logs/stcmds_standard3139_acoustic_listwise_predictions_20260729.jsonl"
listwise_metrics="${src_root}/logs/stcmds_standard3139_acoustic_listwise_metrics_20260729.json"

mkdir -p "$(dirname "${train_records}")" "${output_dir}" "${src_root}/logs"

if [[ -n "${wait_pid}" ]]; then
  echo "[$(date -Is)] waiting for pid=${wait_pid}"
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 60
  done
fi

[[ -s "${test_evidence}" ]] || {
  echo "[error] missing completed standard-word-list evidence: ${test_evidence}" >&2
  exit 1
}

echo "[$(date -Is)] build acoustic-aware test prompts"
"${python}" "${src_root}/analysis/build_acoustic_listwise_covo.py" \
  --input "${test_evidence}" \
  --output "${test_records}" \
  --max-nbest 10 \
  --max-hotwords 8 \
  --allow-reference-outside-nbest

echo "[$(date -Is)] prompt-only baseline"
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/infer_lora_text.py" \
  --input "${test_records}" \
  --output "${prompt_predictions}" \
  --model-name-or-path "${model}" \
  --adapter-path "${base_adapter}" \
  --device auto \
  --batch-size 4 \
  --max-new-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --progress-every 100 \
  --disable-thinking
"${python}" "${covo_root}/scripts/evaluate_correction_jsonl.py" \
  --input "${prompt_predictions}" > "${prompt_metrics}"
cat "${prompt_metrics}"

if [[ ! -s "${train_evidence}" ]] || [[ "$(wc -l < "${train_evidence}")" -ne 17301 ]]; then
  echo "[$(date -Is)] generate full AISHELL CB-SenseVoice train evidence"
  (
    cd "${src_root}"
    OUTPUT_PREFIX="$(basename "${train_evidence_prefix}")" \
      ./analysis/run_aishell_sensevoice_w14_train_evidence_shards.sh
  )
fi

echo "[$(date -Is)] build acoustic listwise train records"
"${python}" "${src_root}/analysis/build_acoustic_listwise_covo.py" \
  --input "${train_evidence}" \
  --output "${train_records}" \
  --max-nbest 10 \
  --max-hotwords 8

echo "[$(date -Is)] train acoustic listwise COVO"
PYTORCH_ALLOC_CONF=expandable_segments:True \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/train_lora_listwise.py" \
  --train-file "${train_records}" \
  --output-dir "${output_dir}" \
  --model-name-or-path "${model}" \
  --adapter-path "${base_adapter}" \
  --max-length 768 \
  --max-candidates 4 \
  --epochs 1 \
  --learning-rate 5e-7 \
  --temperature 0.2 \
  --sft-weight 0.2 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --logging-steps 20 \
  --save-steps 1000 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking

echo "[$(date -Is)] full ST-CMDS inference"
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/infer_lora_text.py" \
  --input "${test_records}" \
  --output "${listwise_predictions}" \
  --model-name-or-path "${model}" \
  --adapter-path "${output_dir}" \
  --device auto \
  --batch-size 4 \
  --max-new-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --progress-every 100 \
  --disable-thinking
"${python}" "${covo_root}/scripts/evaluate_correction_jsonl.py" \
  --input "${listwise_predictions}" > "${listwise_metrics}"
cat "${listwise_metrics}"
echo "[$(date -Is)] acoustic listwise experiment done"
