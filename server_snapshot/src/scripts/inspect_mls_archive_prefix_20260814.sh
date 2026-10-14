#!/usr/bin/env bash
set -o pipefail
curl -fsSL https://dl.fbaipublicfiles.com/mls/mls_english_opus.tar.gz.partaa \
  | gzip -dc \
  | tar -tf - \
  | head -20
