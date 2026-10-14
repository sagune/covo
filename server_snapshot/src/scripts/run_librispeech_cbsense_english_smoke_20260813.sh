#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/datasets/librispeech/cb_sensevoice_english_kws_20260813
PY=/root/autodl-tmp/great/bin/python
cd /root/autodl-tmp/src
"$PY" analysis/extract_sensevoice_hidden_states.py   -a "$ROOT/hotword/dev-clean/keywords-audios/tts"   -t "$ROOT/hotword/dev-clean/keywords-hs/tts"   --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json
"$PY" analysis/extract_sensevoice_hidden_states.py   -a "$ROOT/wav/dev-clean"   -t "$ROOT/hotword/dev-clean/hs"   --device cuda:0 --language en --textnorm woitn --limit 10 --manifest-name manifest.json
