#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?usage: $0 PID_TO_WAIT_FOR}"
repo_root="/root/autodl-tmp"
src_root="${repo_root}/src"
python="${repo_root}/great/bin/python"
covo_model="${repo_root}/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B"
covo_adapter="${repo_root}/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7"
sensevoice_model="/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 60
done

aishell_evidence="${src_root}/logs/cb_sensevoice_aishell_error_targeted_full_20260727.jsonl"
stcmds_evidence="${src_root}/logs/cb_sensevoice_stcmds_error_targeted_full_20260727.jsonl"

[[ "$(wc -l < "${aishell_evidence}")" -eq 7176 ]]
[[ "$(wc -l < "${stcmds_evidence}")" -eq 5130 ]]

cd "${src_root}"

printf '[%s] AISHELL targeted gold full start\n' "$(date -Is)"
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
PYTHONUNBUFFERED=1 CBW_DEBUG_LOG=off \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_aishell_error_targeted_gold_full_20260727.jsonl \
CBW_METRICS_OUT=logs/test_metrics_cb_sensevoice_aishell_error_targeted_gold_full_20260727.csv \
./scripts/run_cb_sensevoice_full_test.sh aishell-targeted \
  --model.init_args.oracle=gold \
  --model.init_args.sensevoice_ckpt="${sensevoice_model}" \
  --model.init_args.oracle_nbest_diagnostic=false \
  --model.init_args.oracle_nbest_detail_path=logs/oracle_nbest_detail_cb_sensevoice_aishell_error_targeted_gold_full_20260727.csv \
  --model.init_args.oracle_nbest_summary_path=logs/oracle_nbest_summary_cb_sensevoice_aishell_error_targeted_gold_full_20260727.csv

printf '[%s] ST-CMDS targeted gold full start\n' "$(date -Is)"
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
PYTHONUNBUFFERED=1 CBW_DEBUG_LOG=off \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_stcmds_error_targeted_gold_full_20260727.jsonl \
CBW_METRICS_OUT=logs/test_metrics_cb_sensevoice_stcmds_error_targeted_gold_full_20260727.csv \
./scripts/run_cb_sensevoice_full_test.sh stcmds-targeted \
  --model.init_args.oracle=gold \
  --model.init_args.sensevoice_ckpt="${sensevoice_model}" \
  --model.init_args.oracle_nbest_diagnostic=false \
  --model.init_args.oracle_nbest_detail_path=logs/oracle_nbest_detail_cb_sensevoice_stcmds_error_targeted_gold_full_20260727.csv \
  --model.init_args.oracle_nbest_summary_path=logs/oracle_nbest_summary_cb_sensevoice_stcmds_error_targeted_gold_full_20260727.csv

run_covo() {
  local dataset="$1"
  local evidence="$2"
  local messages="logs/cb_sensevoice_${dataset}_error_targeted_covo_messages_full_20260727.jsonl"
  local predictions="logs/cb_sensevoice_${dataset}_error_targeted_covo_predictions_full_20260727.jsonl"
  local metrics="logs/cb_sensevoice_${dataset}_error_targeted_covo_metrics_full_20260727.json"

  printf '[%s] %s targeted COVO full start\n' "$(date -Is)" "${dataset}"
  "${python}" analysis/cbsensevoice_covo_bridge.py prepare \
    --input "${evidence}" \
    --output "${messages}" \
    --max-nbest 6 \
    --max-pinyin 3 \
    --max-hotwords 8 \
    --include-pinyin \
    --protect-supported-hotwords \
    --clean-nbest \
    --include-consensus-spans \
    --include-hotword-evidence \
    --prefer-same-length \
    --trust-asr-top1 \
    --preserve-anchor-digits

  PYTHONPATH="${repo_root}/covo/src" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  "${python}" "${repo_root}/covo/scripts/infer_lora_text.py" \
    --input "${messages}" \
    --output "${predictions}" \
    --model-name-or-path "${covo_model}" \
    --adapter-path "${covo_adapter}" \
    --batch-size 4 \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --top-p 1.0 \
    --device auto \
    --progress-every 100 \
    --disable-thinking

  "${python}" "${repo_root}/covo/scripts/evaluate_correction_jsonl.py" \
    --input "${predictions}" \
    --prediction-field prediction \
    --reference-field reference \
    --baseline-field input.asr_top1 | tee "${metrics}"
}

run_covo aishell "${aishell_evidence}"
run_covo stcmds "${stcmds_evidence}"

printf '[%s] all targeted full follow-ups done\n' "$(date -Is)"
