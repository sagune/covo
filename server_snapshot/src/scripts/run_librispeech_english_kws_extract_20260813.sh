#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/datasets/librispeech/data_librispeech_kws_english_20260813
PY=/root/autodl-tmp/great/bin/python
SRC=/root/autodl-tmp/src
cd "$SRC"
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/kws/keywords-audios/tts" -t "$ROOT/kws/keywords-hs/tts" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/kws/wav" -t "$ROOT/kws/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/wav/dev" -t "$ROOT/hotword/dev/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/wav/test" -t "$ROOT/hotword/test/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
