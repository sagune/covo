#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp
src_dir="$workspace/src"
python_bin="$workspace/great/bin/python"
dataset_root="$workspace/datasets/thchs30"
cb_root="$dataset_root/cb_sensevoice_error_hotwords_20260811"
log_dir="$src_dir/logs"
covo_dir="$workspace/cbwhisper_covo_migration_20260609_tar_extracted/covo"
covo_ckpt="$covo_dir/outputs/qwen35_magicdata_hardneg_dropout_balanced_2epoch_20260805/checkpoint-20000"
base_model="$workspace/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B"

mkdir -p "$log_dir"
cd "$src_dir"

"$python_bin" analysis/prepare_thchs30_error_hotwords.py \
  --messages logs/thchs30_acoustic_messages_20260730.jsonl \
  --adapter-predictions logs/thchs30_acoustic_aishell_adapter_predictions_20260730.jsonl \
  --audio-root "$dataset_root/full_audio" \
  --output-root "$cb_root" \
  --top-k 1500 \
  > "$log_dir/prepare_thchs30_error_hotwords_20260811.log" 2>&1

"$python_bin" analysis/synthesize_edge_tts_keywords.py \
  --output-dir "$cb_root/hotword/test/keywords-audios/tts" \
  --keywords "$cb_root/hotword/test/hotword.txt" \
  --concurrency 3 \
  --retries 5 \
  --timeout-seconds 60 \
  --log-every 100 \
  > "$log_dir/thchs30_error_hotwords_tts_20260811.log" 2>&1

"$python_bin" analysis/extract_sensevoice_hidden_states.py \
  --audios "$cb_root/hotword/test/keywords-audios/tts" \
  --target "$cb_root/hotword/test/keywords-hs/tts" \
  --model iic/SenseVoiceSmall \
  --device cuda:0 \
  --language zh \
  --textnorm woitn \
  --skip-existing --fail-on-error \
  > "$log_dir/thchs30_error_hotwords_keyword_hs_20260811.log" 2>&1

"$python_bin" analysis/extract_sensevoice_hidden_states.py \
  --audios "$cb_root/wav/test" \
  --target "$cb_root/hotword/test/hs" \
  --utterances "$cb_root/hotword/test/uttid" \
  --model iic/SenseVoiceSmall \
  --device cuda:0 \
  --language zh \
  --textnorm woitn \
  --skip-existing --fail-on-error \
  > "$log_dir/thchs30_error_hotwords_utt_hs_20260811.log" 2>&1

CBW_METRICS_OUT="logs/test_metrics_cb_sensevoice_thchs30_error_hotwords_20260811.csv" \
CBW_DEBUG_LOG="logs/runtime_probe_cb_sensevoice_thchs30_error_hotwords_20260811.jsonl" \
CBW_EVIDENCE_OUT="logs/cb_sensevoice_thchs30_error_hotwords_evidence_20260811.jsonl" \
CBW_EVAL_BOOTSTRAPS=0 \
"$python_bin" run_CLI.py test --config configs/cb-sensevoice-thchs30-error-hotwords.yaml \
  > "$log_dir/experiment_cb_sensevoice_thchs30_error_hotwords_20260811.log" 2>&1

"$python_bin" analysis/cbsensevoice_covo_bridge.py prepare \
  --input "$log_dir/cb_sensevoice_thchs30_error_hotwords_evidence_20260811.jsonl" \
  --output "$log_dir/thchs30_error_hotwords_cbsense_covo_messages_ckpt20000_20260811.jsonl" \
  --prompt-mode selector_spoken \
  --include-pinyin \
  --include-hotword-evidence \
  > "$log_dir/thchs30_error_hotwords_cbsense_covo_bridge_20260811.log" 2>&1

cd "$covo_dir"
"$python_bin" scripts/infer_lora_text.py \
  --model-name-or-path "$base_model" \
  --adapter-path "$covo_ckpt" \
  --input "$log_dir/thchs30_error_hotwords_cbsense_covo_messages_ckpt20000_20260811.jsonl" \
  --output "$log_dir/thchs30_error_hotwords_cbsense_covo_ckpt20000_predictions_20260811.jsonl" \
  --batch-size 8 \
  --max-new-tokens 96 \
  --disable-thinking

"$python_bin" scripts/evaluate_correction_jsonl.py \
  --input "$log_dir/thchs30_error_hotwords_cbsense_covo_ckpt20000_predictions_20260811.jsonl" \
  > "$log_dir/thchs30_error_hotwords_cbsense_covo_ckpt20000_summary_20260811.json"
