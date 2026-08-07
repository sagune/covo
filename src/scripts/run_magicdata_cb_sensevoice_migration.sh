#!/usr/bin/env bash
set -euo pipefail

workspace=/root/autodl-tmp
src_dir="$workspace/src"
python_bin="$workspace/great/bin/python"
hanlp_python="$workspace/hanlp_env/bin/python"
dataset_root="$workspace/datasets/magicdata_read"
cb_root="$dataset_root/cb_sensevoice_oraclelex600"
log_dir="$src_dir/logs"

mkdir -p "$log_dir" "$cb_root"

# Serialize behind the active original-COVO checkpoint selection and test.
while pgrep -f "finalize_magicdata_original_covo_eval.sh" >/dev/null || \
      pgrep -f "infer_lora_text.py.*magicdata_original_covo" >/dev/null; do
  sleep 60
done

for split in dev test; do
  keyword_file="$dataset_root/magicdata_${split}_hanlp_entities_top600.txt"
  keyword_audio_dir="$cb_root/hotword/$split/keywords-audios/tts"
  keyword_hs_dir="$cb_root/hotword/$split/keywords-hs/tts"
  mkdir -p "$keyword_audio_dir" "$keyword_hs_dir"

  "$hanlp_python" "$src_dir/analysis/extract_hanlp_entity_lexicon.py" \
    --input "$log_dir/magicdata_read_${split}.jsonl" \
    --output "$keyword_file" \
    --batch-size 64 \
    --min-count 1 \
    --max-keywords 600 \
    > "$log_dir/magicdata_cb_sensevoice_${split}_oracle_lexicon_20260807.json"

  "$python_bin" "$src_dir/analysis/prepare_stcmds_cb_sensevoice.py" \
    --manifest "$log_dir/magicdata_read_${split}.jsonl" \
    --keywords "$keyword_file" \
    --output-root "$cb_root" \
    --split "$split" \
    > "$log_dir/prepare_magicdata_cb_sensevoice_${split}_20260807.json"

  "$python_bin" "$src_dir/analysis/synthesize_edge_tts_keywords.py" \
    --output-dir "$keyword_audio_dir" \
    --keywords "$keyword_file" \
    --concurrency 4 \
    --retries 3 \
    --timeout-seconds 45 \
    --log-every 100 \
    --fail-on-error \
    > "$log_dir/magicdata_cb_sensevoice_${split}_tts_20260807.log" 2>&1

  "$python_bin" "$src_dir/analysis/extract_sensevoice_hidden_states.py" \
    --audios "$keyword_audio_dir" \
    --target "$keyword_hs_dir" \
    --model iic/SenseVoiceSmall \
    --device cuda:0 \
    --language zh \
    --textnorm woitn \
    --skip-existing --fail-on-error \
    > "$log_dir/magicdata_cb_sensevoice_${split}_keyword_hs_20260807.log" 2>&1

  "$python_bin" "$src_dir/analysis/extract_sensevoice_hidden_states.py" \
    --audios "$cb_root/wav/$split" \
    --target "$cb_root/hotword/$split/hs" \
    --utterances "$cb_root/hotword/$split/uttid" \
    --model iic/SenseVoiceSmall \
    --device cuda:0 \
    --language zh \
    --textnorm woitn \
    --skip-existing --fail-on-error \
    > "$log_dir/magicdata_cb_sensevoice_${split}_hs_20260807.log" 2>&1
done

cd "$src_dir"
"$python_bin" analysis/kws_topk_recall.py \
  --root "$cb_root" \
  --split dev \
  --features-size 150 750 \
  --kws-ckpt outputs/aishell_sensevoice_kws/checkpoints/f1G/f1G-epoch=16-step=102085.ckpt \
  --kws-positive-threshold 0.787 \
  --output-csv logs/kws_topk_recall_magicdata_dev_20260807.csv \
  > logs/kws_topk_recall_magicdata_dev_20260807.log 2>&1

for split in dev test; do
  CBW_METRICS_OUT="logs/test_metrics_cb_sensevoice_magicdata_${split}_20260807.csv" \
  CBW_DEBUG_LOG="logs/runtime_probe_cb_sensevoice_magicdata_${split}_20260807.jsonl" \
  CBW_EVIDENCE_OUT="logs/cb_sensevoice_magicdata_${split}_evidence_20260807.jsonl" \
  "$python_bin" run_CLI.py test --config configs/cb-sensevoice-magicdata.yaml \
    --data.init_args.test_split="$split" \
    --model.init_args.split="$split" \
    --model.init_args.oracle_nbest_detail_path="logs/oracle_nbest_detail_cb_sensevoice_magicdata_${split}_20260807.csv" \
    --model.init_args.oracle_nbest_summary_path="logs/oracle_nbest_summary_cb_sensevoice_magicdata_${split}_20260807.csv" \
    > "logs/experiment_cb_sensevoice_magicdata_${split}_20260807.log" 2>&1
done
