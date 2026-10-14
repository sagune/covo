#!/usr/bin/env python3
"""Build a CB-Whisper-style English KWS corpus from LibriSpeech train-clean-100."""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


SOURCE = Path("/root/autodl-tmp/datasets/librispeech/LibriSpeech/train-clean-100")
TARGET = Path("/root/autodl-tmp/datasets/librispeech/data_librispeech_lowfreq5000_kws_20260813")
VOCAB_SIZE = 5000
VOICE = "en-US-AriaNeural"


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> None:
    utterances: dict[str, tuple[Path, str, list[str]]] = {}
    frequencies: Counter[str] = Counter()
    for transcript in SOURCE.rglob("*.trans.txt"):
        for line in transcript.read_text(encoding="utf-8").splitlines():
            code, text = line.split(" ", 1)
            token_list = words(text)
            audio = transcript.parent / f"{code}.flac"
            if audio.exists():
                utterances[code] = (audio, text.lower(), token_list)
                frequencies.update(token_list)

    # Low frequency takes priority, but exclude one-off spellings: every enrolled
    # word needs at least two positive utterances in the source training split.
    vocabulary = [word for _, word in sorted((count, word) for word, count in frequencies.items() if count >= 2)[:VOCAB_SIZE]]
    vocab_set = set(vocabulary)
    word_index = {word: index for index, word in enumerate(vocabulary)}
    reverse_index = {word: sorted(vocabulary, key=lambda value: value[::-1]).index(word) for word in vocabulary}

    occurrences: dict[str, list[str]] = defaultdict(list)
    positives_by_code: dict[str, list[str]] = defaultdict(list)
    for code, (_, _, token_list) in utterances.items():
        for word in set(token_list) & vocab_set:
            occurrences[word].append(code)
            positives_by_code[code].append(word)

    # Guarantee each selected word contributes a training positive. Remaining
    # utterances are deterministically split for held-out KWS calibration.
    train_codes: set[str] = set()
    for word in vocabulary:
        train_codes.add(sorted(occurrences[word])[0])
    split: dict[str, str] = {}
    for code in positives_by_code:
        if code in train_codes:
            split[code] = "train"
        else:
            bucket = int(hashlib.sha1(code.encode("utf-8")).hexdigest()[:8], 16) % 10
            split[code] = "dev" if bucket == 0 else "test" if bucket == 1 else "train"

    for path in [
        TARGET / "kws" / "wav",
        TARGET / "kws" / "hs",
        TARGET / "kws" / "keywords-audios" / "tts",
        TARGET / "kws" / "keywords-hs" / "tts",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (TARGET / "kws" / "keywords.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    (TARGET / "kws" / "keywords_voice.txt").write_text(
        "\n".join(f"{word}\t{VOICE}" for word in vocabulary) + "\n", encoding="utf-8"
    )

    train_rows: list[str] = []
    heldout: dict[str, list[str]] = {"dev": [], "test": []}
    for code in sorted(positives_by_code):
        audio, text, _ = utterances[code]
        positive_words = sorted(set(positives_by_code[code]), key=word_index.__getitem__)
        if split[code] == "train":
            link(audio, TARGET / "kws" / "wav" / f"{code}.wav")
            entries = [code]
            for word in positive_words:
                entries.extend((word, str(word_index[word]), str(reverse_index[word])))
            train_rows.append("\t".join(entries))
        else:
            heldout[split[code]].append(code)

    (TARGET / "kws" / "positives.tsv").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    width = len(str(VOCAB_SIZE - 1))
    for split_name, codes in heldout.items():
        hotword = TARGET / "hotword" / split_name
        wav_root = TARGET / "wav" / split_name
        (hotword / "hs").mkdir(parents=True, exist_ok=True)
        wav_root.mkdir(parents=True, exist_ok=True)
        for name in ("hotword.txt", "r1-hotword.txt"):
            (hotword / name).write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        for name in ("keywords-audios", "keywords-hs"):
            destination = hotword / name
            if not destination.exists():
                destination.symlink_to(os.path.relpath(TARGET / "kws" / name, hotword), target_is_directory=True)
        records = []
        for code in sorted(codes):
            audio, text, _ = utterances[code]
            link(audio, wav_root / f"{code}.wav")
            records.append(f"{code} {text}")
        (hotword / "text").write_text("\n".join(records) + "\n", encoding="utf-8")
        (hotword / "uttid").write_text("\n".join(records) + "\n", encoding="utf-8")
        (hotword / "aligned.txt").write_text("", encoding="utf-8")

    print({
        "vocabulary": len(vocabulary),
        "frequency_range": [frequencies[vocabulary[0]], frequencies[vocabulary[-1]]],
        "positive_occurrences": sum(len(values) for values in occurrences.values()),
        "train_utterances": len(train_rows),
        "dev_utterances": len(heldout["dev"]),
        "test_utterances": len(heldout["test"]),
        "root": str(TARGET),
    })


if __name__ == "__main__":
    main()
