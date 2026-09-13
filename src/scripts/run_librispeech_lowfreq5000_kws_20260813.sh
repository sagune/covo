#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/datasets/librispeech/data_librispeech_lowfreq5000_kws_20260813
PY=/root/autodl-tmp/great/bin/python
SRC=/root/autodl-tmp/src

cd "$SRC"
"$PY" scripts/build_librispeech_lowfreq5000_kws_20260813.py
"$PY" analysis/synthesize_edge_tts_keywords.py --keywords "$ROOT/kws/keywords_voice.txt" --output-dir "$ROOT/kws/keywords-audios/tts" --concurrency 12 --retries 4 --timeout-seconds 45 --log-every 100
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/kws/keywords-audios/tts" -t "$ROOT/kws/keywords-hs/tts" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/kws/wav" -t "$ROOT/kws/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/wav/dev" -t "$ROOT/hotword/dev/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
"$PY" analysis/extract_sensevoice_hidden_states.py -a "$ROOT/wav/test" -t "$ROOT/hotword/test/hs" --device cuda:0 --language en --textnorm woitn --manifest-name manifest.json --skip-existing
