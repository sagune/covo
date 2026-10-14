#!/usr/bin/env bash
set -u

pkill -f '/root/autodl-tmp/install_librispeech.sh' 2>/dev/null || true

urls=(
  'https://hf-mirror.com/datasets/k2-fsa/LibriSpeech/resolve/main/train-clean-100.tar.gz?download=true'
  'https://huggingface.co/datasets/k2-fsa/LibriSpeech/resolve/main/train-clean-100.tar.gz?download=true'
  'https://openslr.trmal.net/resources/12/train-clean-100.tar.gz'
  'https://openslr.elda.org/resources/12/train-clean-100.tar.gz'
)

for url in "${urls[@]}"; do
  echo "TEST $url"
  timeout 20 curl -L -r 0-10485759 -o /dev/null -sS \
    -w 'bytes=%{size_download} speed=%{speed_download}\n' "$url" || true
done
