#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/src

ROOT=/root/autodl-tmp/datasets/mls_official_cbwhisper_selected
TRAIN="$ROOT/mls_english_opus/train"
REUSE=/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814/mls_english_opus/train
EVAL=/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814
MODEL=/root/.cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master
PY=/root/autodl-tmp/great/bin/python

while [[ ! -f "$ROOT/extraction_summary.json" ]]; do
  echo "[$(date -Is)] stage=wait_selected_audio"
  sleep 300
done

echo "[$(date -Is)] stage=reuse_existing_tts"
mkdir -p "$TRAIN/keywords-audios/tts" "$TRAIN/keywords-hs/tts"
for source in "$REUSE"/keywords-audios/tts/*.mp3; do
  [[ -e "$source" ]] || continue
  target="$TRAIN/keywords-audios/tts/$(basename "$source")"
  [[ -e "$target" ]] || ln "$source" "$target"
done
for source in "$REUSE"/keywords-hs/tts/*.bin; do
  [[ -e "$source" ]] || continue
  target="$TRAIN/keywords-hs/tts/$(basename "$source")"
  [[ -e "$target" ]] || ln "$source" "$target"
done

echo "[$(date -Is)] stage=cut_natural_keywords"
"$PY" utils.py --cut_audios \
  --audios "$TRAIN/audio" --keywords "$TRAIN/aligned.tsv" \
  --target "$TRAIN/keywords-audios/natural"

echo "[$(date -Is)] stage=complete_tts_keywords"
for pass in 1 2 3; do
  "$PY" analysis/synthesize_edge_tts_keywords.py \
    --keywords "$TRAIN/keywords_voice.txt" \
    --output-dir "$TRAIN/keywords-audios/tts" \
    --concurrency 8 --retries 4 --timeout-seconds 60 --log-every 200
  count=$(find "$TRAIN/keywords-audios/tts" -maxdepth 1 -type f -name '*.mp3' | wc -l)
  [[ "$count" -ge 12000 ]] && break
done

extract_hs() {
  local name=$1
  local audios=$2
  local target=$3
  echo "[$(date -Is)] stage=extract_${name}"
  "$PY" analysis/extract_sensevoice_hidden_states.py \
    --audios "$audios" --target "$target" --model "$MODEL" \
    --device cuda:0 --language en --skip-existing --fail-on-error
}

extract_hs train "$TRAIN/audio" "$TRAIN/hs"
extract_hs keyword_tts "$TRAIN/keywords-audios/tts" "$TRAIN/keywords-hs/tts"
extract_hs keyword_natural "$TRAIN/keywords-audios/natural" "$TRAIN/keywords-hs/natural"
extract_hs dev "$EVAL/wav/dev" "$EVAL/hotword/dev/hs"
extract_hs test "$EVAL/wav/test" "$EVAL/hotword/test/hs"

for split in dev test; do
  link="$EVAL/hotword/$split/keywords-hs"
  if [[ -L "$link" ]]; then
    rm "$link"
  elif [[ -e "$link" ]]; then
    resolved=$(readlink -f "$link")
    case "$resolved" in
      "$EVAL"/*) rm -rf -- "$resolved" ;;
      *) echo "unsafe keyword hs path: $resolved"; exit 5 ;;
    esac
  fi
  ln -s "$TRAIN/keywords-hs" "$link"
done

train_hs=$(find "$TRAIN/hs" -maxdepth 1 -type f -name '*.bin' | wc -l)
tts_audio=$(find "$TRAIN/keywords-audios/tts" -maxdepth 1 -type f -name '*.mp3' | wc -l)
tts_hs=$(find "$TRAIN/keywords-hs/tts" -maxdepth 1 -type f -name '*.bin' | wc -l)
natural_audio=$(find "$TRAIN/keywords-audios/natural" -maxdepth 1 -type f -name '*.mp3' | wc -l)
natural_hs=$(find "$TRAIN/keywords-hs/natural" -maxdepth 1 -type f -name '*.bin' | wc -l)
cat > "$ROOT/sensevoice_preparation_summary.json" <<JSON
{
  "utterance_hidden_states": $train_hs,
  "tts_keyword_audio": $tts_audio,
  "tts_keyword_hidden_states": $tts_hs,
  "natural_keyword_audio": $natural_audio,
  "natural_keyword_hidden_states": $natural_hs
}
JSON
echo "[$(date -Is)] stage=complete train_hs=$train_hs tts_hs=$tts_hs natural_hs=$natural_hs"
