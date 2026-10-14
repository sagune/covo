#!/usr/bin/env python3
"""Create a deterministic utterance-level train/dev/test split for ST-CMDS."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--train-ratio", type=float, default=0.93)
    parser.add_argument("--dev-ratio", type=float, default=0.02)
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--dev-count", type=int)
    args = parser.parse_args()
    use_counts = args.train_count is not None or args.dev_count is not None
    if use_counts and (args.train_count is None or args.dev_count is None):
        raise ValueError("--train-count and --dev-count must be provided together")
    if not use_counts and (args.train_ratio <= 0 or args.dev_ratio <= 0 or args.train_ratio + args.dev_ratio >= 1):
        raise ValueError("train-ratio and dev-ratio must be positive and leave test data")

    rows = []
    for txt in sorted(args.root.glob("*.txt")):
        wav = txt.with_suffix(".wav")
        if not wav.exists():
            continue
        reference = txt.read_text(encoding="utf-8-sig").strip()
        if not reference:
            continue
        stem = wav.stem
        marker = stem.split("P", 1)[-1]
        device = "A" if "A" in marker else "I"
        rows.append({
            "id": stem,
            "reference": reference,
            "wav": str(wav),
            "speaker": marker.split("A", 1)[0].split("I", 1)[0],
            "device": device,
            "source": "ST-CMDS",
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n = len(rows)
    if use_counts:
        train_end = int(args.train_count)
        dev_end = train_end + int(args.dev_count)
        if train_end <= 0 or dev_end >= n:
            raise ValueError("explicit counts must be positive and leave a non-empty test split")
    else:
        train_end = int(n * args.train_ratio)
        dev_end = train_end + int(n * args.dev_ratio)
    splits = {
        "train": rows[:train_end],
        "dev": rows[train_end:dev_end],
        "test": rows[dev_end:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps({**row, "split": split}, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "available": n,
        "seed": args.seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "paper_cardinality_only": bool(use_counts),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
