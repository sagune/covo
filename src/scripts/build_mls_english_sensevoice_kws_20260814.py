#!/usr/bin/env python3
"""Convert MLS-English Parquet shards to the CB-Whisper KWS layout."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


VOICES = (
    "en-US-AriaNeural",
    "en-US-AvaNeural",
    "en-US-ChristopherNeural",
    "en-US-EricNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def iter_rows(path: Path):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=128):
        yield from batch.to_pylist()


def select_rows(path: Path, max_hours: float | None) -> list[dict]:
    selected = []
    duration = 0.0
    limit = None if max_hours is None or max_hours <= 0 else max_hours * 3600.0
    for row in iter_rows(path):
        if not row.get("audio", {}).get("bytes") or not row.get("transcript"):
            continue
        selected.append(row)
        duration += float(row.get("audio_duration") or 0.0)
        if limit is not None and duration >= limit:
            break
    return selected


def code_for(row: dict) -> str:
    return Path(row["audio"]["path"]).stem


def write_audio(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(row["audio"]["bytes"])


def write_hotword_split(root: Path, split: str, rows: list[dict], vocabulary: list[str]) -> dict:
    hotword = root / "hotword" / split
    wav = root / "wav" / split
    (hotword / "hs").mkdir(parents=True, exist_ok=True)
    wav.mkdir(parents=True, exist_ok=True)
    for name in ("hotword.txt", "r1-hotword.txt"):
        (hotword / name).write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

    keyword_root = root / "mls_english_opus" / "train"
    for name in ("keywords-audios", "keywords-hs"):
        link = hotword / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(Path("../..") / "mls_english_opus" / "train" / name, target_is_directory=True)

    records = []
    duration = 0.0
    for row in rows:
        code = code_for(row)
        transcript = " ".join(tokenize(row["transcript"]))
        write_audio(row, wav / f"{code}.opus")
        records.append(f"{code} {transcript}")
        duration += float(row.get("audio_duration") or 0.0)
    content = "\n".join(records) + "\n"
    (hotword / "text").write_text(content, encoding="utf-8")
    (hotword / "uttid").write_text(content, encoding="utf-8")
    (hotword / "aligned.txt").write_text("", encoding="utf-8")
    return {"rows": len(rows), "hours": duration / 3600.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-hours", type=float, default=10.0)
    parser.add_argument("--dev-hours", type=float, default=0.0, help="0 keeps the full split")
    parser.add_argument("--test-hours", type=float, default=0.0, help="0 keeps the full split")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-keywords", type=int, default=12000)
    args = parser.parse_args()

    train_rows = select_rows(args.train_parquet, args.train_hours)
    dev_rows = select_rows(args.dev_parquet, args.dev_hours)
    test_rows = select_rows(args.test_parquet, args.test_hours)

    frequencies = Counter(word for row in train_rows for word in tokenize(row["transcript"]))
    vocabulary = sorted(word for word, count in frequencies.items() if count >= args.min_frequency)
    if args.max_keywords > 0:
        vocabulary = vocabulary[: args.max_keywords]
    vocab_set = set(vocabulary)
    word_index = {word: index for index, word in enumerate(vocabulary)}
    reverse_index = {word: index for index, word in enumerate(sorted(vocabulary, key=lambda value: value[::-1]))}

    train_root = args.output / "mls_english_opus" / "train"
    for path in (
        train_root / "audio",
        train_root / "hs",
        train_root / "keywords-audios" / "tts",
        train_root / "keywords-hs" / "tts",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (train_root / "keywords.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    (train_root / "keywords_voice.txt").write_text(
        "\n".join(f"{word}\t{VOICES[index % len(VOICES)]}" for index, word in enumerate(vocabulary)) + "\n",
        encoding="utf-8",
    )

    positives = []
    utterance_ids = []
    train_duration = 0.0
    positive_count = 0
    for row in train_rows:
        code = code_for(row)
        words = sorted(set(tokenize(row["transcript"])) & vocab_set, key=word_index.__getitem__)
        if not words:
            continue
        write_audio(row, train_root / "audio" / str(row["speaker_id"]) / str(row["book_id"]) / f"{code}.opus")
        fields = [code]
        for word in words:
            fields.extend((word, str(word_index[word]), str(reverse_index[word])))
        positives.append("\t".join(fields))
        utterance_ids.append(code)
        positive_count += len(words)
        train_duration += float(row.get("audio_duration") or 0.0)
    (train_root / "positives.tsv").write_text("\n".join(positives) + "\n", encoding="utf-8")
    (train_root / "uttid").write_text("\n".join(utterance_ids) + "\n", encoding="utf-8")

    summary = {
        "source": "parler-tts/mls_eng (MLS-English mirror)",
        "train": {"rows": len(positives), "hours": train_duration / 3600.0, "positive_pairs": positive_count},
        "dev": write_hotword_split(args.output, "dev", dev_rows, vocabulary),
        "test": write_hotword_split(args.output, "test", test_rows, vocabulary),
        "vocabulary": len(vocabulary),
        "min_frequency": args.min_frequency,
        "output": str(args.output),
    }
    (args.output / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
