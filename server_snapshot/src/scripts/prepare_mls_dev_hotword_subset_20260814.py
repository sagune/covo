#!/usr/bin/env python3
"""Build a fixed high-frequency MLS-English dev keyword subset."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path("/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814")
TRAIN = ROOT / "mls_english_opus" / "train"
DEV = ROOT / "hotword" / "dev"
SUBSET_SIZE = 1001


def main() -> None:
    vocabulary = [line.strip() for line in (TRAIN / "keywords.txt").read_text().splitlines() if line.strip()]
    old_index = {word: index for index, word in enumerate(vocabulary)}
    counts = Counter()
    for line in (TRAIN / "positives.tsv").read_text().splitlines():
        fields = line.split("\t")
        counts.update(fields[1::3])
    subset = sorted(vocabulary, key=lambda word: (-counts[word], word))[:SUBSET_SIZE]

    for name in ("hotword.txt", "r1-hotword.txt"):
        (DEV / name).write_text("\n".join(subset) + "\n", encoding="utf-8")

    hs_root = DEV / "keywords-hs"
    if hs_root.is_symlink():
        hs_root.unlink()
    elif hs_root.exists():
        shutil.rmtree(hs_root)
    target = hs_root / "tts"
    target.mkdir(parents=True)
    width = len(str(SUBSET_SIZE - 1))
    source_width = len(str(len(vocabulary) - 1))
    for new_idx, word in enumerate(subset):
        source = TRAIN / "keywords-hs" / "tts" / f"{old_index[word]:0{source_width}d}.bin"
        destination = target / f"{new_idx:0{width}d}.bin"
        destination.symlink_to(os.path.relpath(source, destination.parent))

    print({"dev_keywords": len(subset), "min_train_occurrences": min(counts[word] for word in subset)})


if __name__ == "__main__":
    main()
