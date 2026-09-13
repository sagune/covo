#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/src

ROOT=/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814
TRAIN="$ROOT/mls_english_opus/train"
MODEL=/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master
PY=/root/autodl-tmp/great/bin/python
EXPECTED=$(grep -cve '^$' "$TRAIN/keywords.txt")

# Retry transient Edge-TTS failures. Existing files are skipped.
for pass in 1 2 3; do
  count=$(find "$TRAIN/keywords-audios/tts" -maxdepth 1 -type f -name '*.mp3' | wc -l)
  if [[ "$count" -ge "$EXPECTED" ]]; then
    break
  fi
  echo "[$(date -Is)] stage=tts pass=$pass count=$count/$EXPECTED"
  "$PY" analysis/synthesize_edge_tts_keywords.py \
    --keywords "$TRAIN/keywords_voice.txt" \
    --output-dir "$TRAIN/keywords-audios/tts" \
    --concurrency 8 --retries 4 --timeout-seconds 60 --log-every 100
done

count=$(find "$TRAIN/keywords-audios/tts" -maxdepth 1 -type f -name '*.mp3' | wc -l)
if [[ "$count" -lt "$EXPECTED" ]]; then
  echo "[$(date -Is)] error=incomplete_tts count=$count/$EXPECTED"
  exit 2
fi

extract_hs() {
  local name=$1
  local audio=$2
  local target=$3
  echo "[$(date -Is)] stage=extract_${name}"
  "$PY" analysis/extract_sensevoice_hidden_states.py \
    --audios "$audio" --target "$target" --model "$MODEL" \
    --device cuda:0 --language en --skip-existing --fail-on-error
}

extract_hs keyword "$TRAIN/keywords-audios/tts" "$TRAIN/keywords-hs/tts"
extract_hs train "$TRAIN/audio" "$TRAIN/hs"
extract_hs dev "$ROOT/wav/dev" "$ROOT/hotword/dev/hs"
extract_hs test "$ROOT/wav/test" "$ROOT/hotword/test/hs"

echo "[$(date -Is)] stage=prepare_dev_keyword_subset"
"$PY" scripts/prepare_mls_dev_hotword_subset_20260814.py

echo "[$(date -Is)] stage=smoke"
"$PY" scripts/smoke_mls_kws_route_20260814.py

echo "[$(date -Is)] stage=train"
"$PY" run_CLI.py fit --config configs/train-sensevoice-mls-english-10h-resnet_20260814.yaml
echo "[$(date -Is)] stage=complete"
