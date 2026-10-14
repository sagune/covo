#!/usr/bin/env python3
"""Reuse upstream MLS keyword metadata with the locally available MLS audio."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814")
TRAIN = ROOT / "mls_english_opus" / "train"
UPSTREAM = Path("/root/autodl-tmp/datasets/mls/train/mls_english_opus")
PARQUET = Path("/root/autodl-tmp/datasets/mls_hf_parquet/train-00000-of-01416.parquet")
TRAIN_SECONDS = 10 * 3600.0


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def remap_indexed_assets(kind: str, suffix: str, old_words: list[str], new_index: dict[str, int]) -> int:
    parent = TRAIN / kind
    source = parent / "tts"
    backup = parent / "tts_custom4721_backup"
    if source.exists() and not backup.exists():
        source.rename(backup)
    source.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        return 0
    old_width = len(str(len(old_words) - 1))
    new_width = len(str(len(new_index) - 1))
    reused = 0
    for old_idx, word in enumerate(old_words):
        candidate = backup / f"{old_idx:0{old_width}d}.{suffix}"
        if word not in new_index or not candidate.exists():
            continue
        destination = source / f"{new_index[word]:0{new_width}d}.{suffix}"
        if not destination.exists():
            os.link(candidate, destination)
            reused += 1
    return reused


def main() -> None:
    old_words = [line.strip() for line in (TRAIN / "keywords.txt").read_text().splitlines() if line.strip()]
    vocabulary = [line.strip() for line in (UPSTREAM / "keywords.txt").read_text().splitlines() if line.strip()]
    word_index = {word: index for index, word in enumerate(vocabulary)}
    reverse_index = {word: index for index, word in enumerate(sorted(vocabulary, key=lambda word: word[::-1]))}

    reused_audio = remap_indexed_assets("keywords-audios", "mp3", old_words, word_index)
    reused_hs = remap_indexed_assets("keywords-hs", "bin", old_words, word_index)

    reference = TRAIN / "upstream_metadata_reference"
    reference.mkdir(parents=True, exist_ok=True)
    for name in ("aligned.tsv", "keywords.txt", "keywords_voice.txt", "positives.tsv", "uttid"):
        shutil.copy2(UPSTREAM / name, reference / name)
    shutil.copy2(UPSTREAM / "keywords.txt", TRAIN / "keywords.txt")
    shutil.copy2(UPSTREAM / "keywords_voice.txt", TRAIN / "keywords_voice.txt")

    rows = []
    duration = 0.0
    parquet = pq.ParquetFile(PARQUET)
    for batch in parquet.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            rows.append(row)
            duration += float(row.get("audio_duration") or 0.0)
            if duration >= TRAIN_SECONDS:
                break
        if duration >= TRAIN_SECONDS:
            break

    positives = []
    utterance_ids = []
    positive_pairs = 0
    frequencies = Counter()
    vocab_set = set(vocabulary)
    for row in rows:
        code = Path(row["audio"]["path"]).stem
        present = sorted(set(tokens(row["transcript"])) & vocab_set, key=word_index.__getitem__)
        if not present:
            continue
        fields = [code]
        for word in present:
            fields.extend((word, str(word_index[word]), str(reverse_index[word])))
            frequencies[word] += 1
        positives.append("\t".join(fields))
        utterance_ids.append(code)
        positive_pairs += len(present)
    (TRAIN / "positives.tsv").write_text("\n".join(positives) + "\n", encoding="utf-8")
    (TRAIN / "uttid").write_text("\n".join(utterance_ids) + "\n", encoding="utf-8")

    for split in ("dev", "test"):
        split_root = ROOT / "hotword" / split
        for name in ("hotword.txt", "r1-hotword.txt"):
            (split_root / name).write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        hs = split_root / "keywords-hs"
        if hs.is_symlink():
            hs.unlink()
        elif hs.exists():
            shutil.rmtree(hs)
        hs.symlink_to(Path("../..") / "mls_english_opus" / "train" / "keywords-hs", target_is_directory=True)

    summary = {
        "upstream_keywords": len(vocabulary),
        "local_train_utterances": len(positives),
        "local_positive_pairs": positive_pairs,
        "keywords_with_local_positives": len(frequencies),
        "reused_tts_audio": reused_audio,
        "reused_sensevoice_keyword_states": reused_hs,
        "upstream_reference": str(reference),
    }
    (ROOT / "upstream_metadata_adoption_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
