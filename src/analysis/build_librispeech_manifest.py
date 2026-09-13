#!/usr/bin/env python3
"""Build JSONL manifests from an extracted LibriSpeech tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split_root = args.root / args.split
    rows = []
    for transcript_path in sorted(split_root.rglob("*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            utterance_id, reference = line.split(maxsplit=1)
            wav = transcript_path.parent / f"{utterance_id}.flac"
            if not wav.is_file():
                raise FileNotFoundError(wav)
            rows.append({
                "dataset": "librispeech",
                "split": args.split,
                "id": utterance_id,
                "wav": str(wav),
                "reference": reference,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(json.dumps({"split": args.split, "rows": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
