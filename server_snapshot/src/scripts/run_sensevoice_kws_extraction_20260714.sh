#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/root/autodl-tmp/great/bin/python}"
ROOT="${ROOT:-/root/autodl-tmp}"
SRC="${SRC:-datasets/aishell/data_aishell}"
DST="${DST:-datasets/aishell/data_aishell_sensevoice}"

cd "$ROOT"

echo "[start] $(date -Is) SenseVoice AISHELL KWS extraction"

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/wav/train" \
  -t "$DST/kws/hs" \
  -u "$SRC/kws/positives.tsv" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/kws/keywords-audios/natural" \
  -t "$DST/kws/keywords-hs/natural" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/kws/keywords-audios/tts" \
  -t "$DST/kws/keywords-hs/tts" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/wav/dev" \
  -t "$DST/hotword/dev/hs" \
  -u "$SRC/hotword/dev/uttid" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/hotword/dev/keywords-audios/natural" \
  -t "$DST/hotword/dev/keywords-hs/natural" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/hotword/dev/keywords-audios/tts" \
  -t "$DST/hotword/dev/keywords-hs/tts" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/wav/test" \
  -t "$DST/hotword/test/hs" \
  -u "$SRC/hotword/test/uttid" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/hotword/test/keywords-audios/natural" \
  -t "$DST/hotword/test/keywords-hs/natural" \
  --skip-existing

"$PY" src/analysis/extract_sensevoice_hidden_states.py \
  -a "$SRC/hotword/test/keywords-audios/tts" \
  -t "$DST/hotword/test/keywords-hs/tts" \
  --skip-existing

echo "[done] $(date -Is) SenseVoice AISHELL KWS extraction"
