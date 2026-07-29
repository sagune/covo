#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE:-/root/autodl-tmp}"
src_root="${workspace}/src"
python="${PYTHON_BIN:-${workspace}/great/bin/python}"
covo_root="${workspace}/covo"
migration_root="${workspace}/cbwhisper_covo_migration_20260609_tar_extracted"
model="${MODEL_PATH:-${migration_root}/models/Qwen3.5-4B}"
sensevoice_model="${SENSEVOICE_MODEL:-/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master}"
base_adapter="${BASE_ADAPTER:-${migration_root}/covo/outputs/qwen35_acoustic_listwise_aishell_1epoch_20260729}"
output_dir="${OUTPUT_DIR:-${migration_root}/covo/outputs/qwen35_acoustic_listwise_stcmds_1epoch_20260730}"
wait_pid="${WAIT_PID:-}"
num_shards="${NUM_SHARDS:-8}"
max_parallel="${MAX_PARALLEL:-2}"

source_nbest="${migration_root}/covo/data/processed/stcmds_chinesehp_sensevoice/train.qwen.jsonl"
manifest="${workspace}/datasets/stcmds/splits/train.jsonl"
score_prefix="${src_root}/logs/stcmds_train_forced_ctc"
merged_evidence="${score_prefix}_full.jsonl"
train_records="${covo_root}/data/processed/acoustic_listwise/stcmds_train.jsonl"
test_records="${src_root}/logs/stcmds_standard3139_acoustic_listwise_messages_20260729.jsonl"
predictions="${src_root}/logs/stcmds_standard3139_acoustic_listwise_domain_predictions_20260730.jsonl"
metrics="${src_root}/logs/stcmds_standard3139_acoustic_listwise_domain_metrics_20260730.json"

mkdir -p "$(dirname "${train_records}")" "${output_dir}" "${src_root}/logs"

if [[ -n "${wait_pid}" ]]; then
  echo "[$(date -Is)] waiting for AISHELL listwise pid=${wait_pid}"
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 60
  done
fi

[[ -s "${base_adapter}/adapter_model.safetensors" ]] || {
  echo "[error] missing AISHELL listwise adapter: ${base_adapter}" >&2
  exit 1
}
[[ -s "${source_nbest}" ]] || {
  echo "[error] missing ST-CMDS N-best source: ${source_nbest}" >&2
  exit 1
}
[[ -s "${manifest}" ]] || {
  echo "[error] missing ST-CMDS train manifest: ${manifest}" >&2
  exit 1
}

run_shard() {
  local shard="$1"
  local suffix
  suffix=$(printf "%02d" "${shard}")
  local output="${score_prefix}_shard${suffix}.jsonl"
  local log="${src_root}/logs/stcmds_train_forced_ctc_shard${suffix}.log"
  echo "[$(date -Is)] start forced-CTC shard ${shard}/${num_shards}"
  TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "${python}" "${src_root}/analysis/score_sensevoice_candidate_evidence.py" \
    --input "${source_nbest}" \
    --manifest "${manifest}" \
    --output "${output}" \
    --model "${sensevoice_model}" \
    --device cuda:0 \
    --max-nbest 10 \
    --num-shards "${num_shards}" \
    --shard-index "${shard}" \
    --progress-every 200 \
    --use-itn \
    > "${log}" 2>&1
  echo "[$(date -Is)] done forced-CTC shard ${shard}, lines=$(wc -l < "${output}")"
}

if [[ ! -s "${merged_evidence}" ]] || [[ "$(wc -l < "${merged_evidence}")" -lt 95000 ]]; then
  active=0
  pids=()
  for shard in $(seq 0 $((num_shards - 1))); do
    run_shard "${shard}" &
    pids+=("$!")
    active=$((active + 1))
    if [[ "${active}" -ge "${max_parallel}" ]]; then
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
      pids=()
      active=0
    fi
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  temporary="${merged_evidence}.tmp"
  : > "${temporary}"
  for shard in $(seq 0 $((num_shards - 1))); do
    suffix=$(printf "%02d" "${shard}")
    cat "${score_prefix}_shard${suffix}.jsonl" >> "${temporary}"
  done
  mv "${temporary}" "${merged_evidence}"
fi

evidence_lines=$(wc -l < "${merged_evidence}")
if [[ "${evidence_lines}" -lt 95000 ]]; then
  echo "[error] forced-CTC evidence is incomplete: ${evidence_lines}/95418" >&2
  exit 1
fi

echo "[$(date -Is)] build ST-CMDS acoustic listwise records"
"${python}" "${src_root}/analysis/build_acoustic_listwise_covo.py" \
  --input "${merged_evidence}" \
  --output "${train_records}" \
  --max-nbest 10 \
  --max-hotwords 0 \
  --allow-reference-outside-nbest

echo "[$(date -Is)] continue acoustic-aware full-sentence SFT on ST-CMDS"
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
  --save-steps 2000 \
  --preprocessing-num-workers 4 \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 2 \
  --bf16 \
  --gradient-checkpointing \
  --disable-thinking

echo "[$(date -Is)] evaluate ST-CMDS domain listwise adapter"
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH="${covo_root}/src" \
"${python}" "${covo_root}/scripts/infer_lora_text.py" \
  --input "${test_records}" \
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
"${python}" "${covo_root}/scripts/evaluate_correction_jsonl.py" \
  --input "${predictions}" > "${metrics}"
cat "${metrics}"
echo "[$(date -Is)] ST-CMDS acoustic listwise follow-up done"
