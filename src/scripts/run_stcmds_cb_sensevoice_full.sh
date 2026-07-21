#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp
python_bin="$workspace/great/bin/python"
hanlp_python="$workspace/hanlp_env/bin/python"
src_dir="$workspace/src"
dataset_dir="$workspace/datasets/stcmds"
split_dir="$dataset_dir/splits_copyne_cardinality"
cb_root="$dataset_dir/cb_sensevoice"
keyword_file="$dataset_dir/stcmds_test_hanlp_entities.txt"
keyword_audio_dir="$cb_root/hotword/test/keywords-audios/tts"
log_dir="$src_dir/logs"

mkdir -p "$split_dir" "$keyword_audio_dir" "$log_dir"

"$python_bin" "$src_dir/analysis/prepare_stcmds_splits.py" \
  --root "$dataset_dir/ST-CMDS-20170001_1-OS" \
  --output-dir "$split_dir" \
  --seed 20260721 \
  --train-count 82080 \
  --dev-count 10260 \
  > "$log_dir/stcmds_copyne_cardinality_split_20260721.json"

"$hanlp_python" "$src_dir/analysis/extract_hanlp_entity_lexicon.py" \
  --input "$split_dir/test.jsonl" \
  --output "$keyword_file" \
  --batch-size 64 \
  > "$log_dir/stcmds_test_hanlp_entities_20260721.json"

"$python_bin" "$src_dir/analysis/prepare_stcmds_cb_sensevoice.py" \
  --manifest "$split_dir/test.jsonl" \
  --keywords "$keyword_file" \
  --output-root "$cb_root" \
  --split test \
  > "$log_dir/prepare_stcmds_cb_sensevoice_20260721.json"

"$python_bin" "$src_dir/analysis/synthesize_edge_tts_keywords.py" \
  --output-dir "$keyword_audio_dir" \
  --keywords "$keyword_file" \
  --concurrency 6 \
  --retries 3 \
  --log-every 100 \
  --fail-on-error \
  > "$log_dir/stcmds_cb_sensevoice_tts_20260721.log" 2>&1

"$python_bin" "$src_dir/analysis/extract_sensevoice_hidden_states.py" \
  --audios "$keyword_audio_dir" \
  --target "$cb_root/hotword/test/keywords-hs/tts" \
  --model iic/SenseVoiceSmall \
  --device cuda:0 \
  --language zh \
  --textnorm withitn \
  --skip-existing \
  > "$log_dir/stcmds_cb_sensevoice_keyword_hs_20260721.log" 2>&1

"$python_bin" "$src_dir/analysis/extract_sensevoice_hidden_states.py" \
  --audios "$cb_root/wav/test" \
  --target "$cb_root/hotword/test/hs" \
  --utterances "$cb_root/hotword/test/uttid" \
  --model iic/SenseVoiceSmall \
  --device cuda:0 \
  --language zh \
  --textnorm withitn \
  --skip-existing \
  > "$log_dir/stcmds_cb_sensevoice_utterance_hs_20260721.log" 2>&1

cd "$src_dir"
CBW_METRICS_OUT=logs/test_metrics_cb_sensevoice_stcmds_full_20260721.csv \
CBW_DEBUG_LOG=logs/runtime_probe_cb_sensevoice_stcmds_full_20260721.jsonl \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_stcmds_full_evidence_20260721.jsonl \
"$python_bin" run_CLI.py test --config configs/cb-sensevoice-stcmds.yaml \
  > "$log_dir/experiment_cb_sensevoice_stcmds_full_20260721_stdout.log" 2>&1
