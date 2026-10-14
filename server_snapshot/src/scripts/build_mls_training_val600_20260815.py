#!/usr/bin/env python3
"""Build an MLS-English KWS validation split matching the AISHELL train-time scale."""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


TRAIN = Path("/root/autodl-tmp/datasets/mls_official_cbwhisper_selected/mls_english_opus/train")
SOURCE_DEV = Path("/root/autodl-tmp/datasets/mls_sensevoice_english_10h_20260814/hotword/dev")
OUTPUT = Path("/root/autodl-tmp/datasets/mls_sensevoice_english_trainval600_20260815")
KEYWORD_COUNT = 600
UTTERANCE_COUNT = 1334
SEED = 123


def read_nonempty(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def main() -> None:
    vocabulary = read_nonempty(TRAIN / "keywords.txt")
    old_index = {word: index for index, word in enumerate(vocabulary)}
    source_width = len(str(len(vocabulary) - 1))

    lines = read_nonempty(SOURCE_DEV / "text")
    utterance_tokens: list[set[str]] = []
    occurrences: dict[str, set[int]] = defaultdict(set)
    vocabulary_set = set(vocabulary)
    for utterance_index, line in enumerate(lines):
        fields = line.lower().split()
        tokens = set(fields[1:]) & vocabulary_set
        utterance_tokens.append(tokens)
        for word in tokens:
            occurrences[word].add(utterance_index)

    eligible = []
    for word, indices in occurrences.items():
        index = old_index[word]
        name = f"{index:0{source_width}d}.bin"
        if not (TRAIN / "keywords-hs" / "tts" / name).exists():
            continue
        if not (TRAIN / "keywords-hs" / "natural" / name).exists():
            continue
        if len(word) < 4 or not word.isalpha():
            continue
        if 2 <= len(indices) <= 6:
            eligible.append(word)

    if len(eligible) < KEYWORD_COUNT:
        raise RuntimeError(f"only {len(eligible)} eligible keywords, need {KEYWORD_COUNT}")

    rng = random.Random(SEED)
    rng.shuffle(eligible)
    covered: set[int] = set()
    selected: list[str] = []
    remaining = set(eligible)
    while len(selected) < KEYWORD_COUNT:
        # Prefer words that add three new utterances, then lower document frequency.
        word = max(
            remaining,
            key=lambda value: (
                min(3, len(occurrences[value] - covered)),
                len(occurrences[value] - covered),
                -abs(len(occurrences[value]) - 3),
                -len(value),
                value,
            ),
        )
        selected.append(word)
        covered.update(occurrences[word])
        remaining.remove(word)

    if len(covered) < UTTERANCE_COUNT:
        raise RuntimeError(f"selected keywords cover only {len(covered)} utterances")

    selected_set = set(selected)
    covered_indices = sorted(covered)
    rng.shuffle(covered_indices)
    chosen_indices = sorted(covered_indices[:UTTERANCE_COUNT])
    chosen_lines = [lines[index] for index in chosen_indices]
    positive_counts = [len(utterance_tokens[index] & selected_set) for index in chosen_indices]

    dev = OUTPUT / "hotword" / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    for name in ("hotword.txt", "r1-hotword.txt"):
        (dev / name).write_text("\n".join(selected) + "\n", encoding="utf-8")
    content = "\n".join(chosen_lines) + "\n"
    (dev / "text").write_text(content, encoding="utf-8")
    (dev / "uttid").write_text(content, encoding="utf-8")
    (dev / "aligned.txt").write_text("", encoding="utf-8")

    hs_link = dev / "hs"
    replace_path(hs_link)
    hs_link.symlink_to(SOURCE_DEV / "hs", target_is_directory=True)

    keyword_hs = dev / "keywords-hs"
    replace_path(keyword_hs)
    new_width = len(str(KEYWORD_COUNT - 1))
    for modality in ("tts", "natural"):
        target = keyword_hs / modality
        target.mkdir(parents=True, exist_ok=True)
        for new_index, word in enumerate(selected):
            old_name = f"{old_index[word]:0{source_width}d}.bin"
            destination = target / f"{new_index:0{new_width}d}.bin"
            destination.symlink_to(TRAIN / "keywords-hs" / modality / old_name)

    summary = {
        "seed": SEED,
        "keywords": len(selected),
        "utterances": len(chosen_lines),
        "covered_utterances_before_sampling": len(covered),
        "positive_pairs": sum(positive_counts),
        "average_positives_per_utterance": sum(positive_counts) / len(positive_counts),
        "zero_positive_utterances": sum(count == 0 for count in positive_counts),
        "positive_prevalence": sum(positive_counts) / (len(chosen_lines) * len(selected)),
        "keyword_dev_frequency_min": min(len(occurrences[word]) for word in selected),
        "keyword_dev_frequency_max": max(len(occurrences[word]) for word in selected),
        "source_dev": str(SOURCE_DEV),
        "train_keyword_states": str(TRAIN / "keywords-hs"),
    }
    (OUTPUT / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
