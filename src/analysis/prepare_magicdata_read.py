#!/usr/bin/env python3
"""Build a JSONL evaluation manifest for the MAGICDATA read-speech test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transcript_path = args.test_root / "TRANS.txt"
    audio_by_name = {
        path.name: path
        for path in args.test_root.rglob("*.wav")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing = []
    with transcript_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        with args.output.open("w", encoding="utf-8", newline="\n") as target:
            for row in reader:
                filename = str(row.get("UtteranceID", "")).strip()
                reference = str(row.get("Transcription", "")).strip()
                wav_path = audio_by_name.get(filename)
                if not filename or not reference or wav_path is None:
                    if filename and wav_path is None:
                        missing.append(filename)
                    continue
                target.write(json.dumps({
                    "id": Path(filename).stem,
                    "reference": reference,
                    "wav": str(wav_path),
                    "source": "MAGICDATA-READ",
                    "split": "test",
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} transcript entries have no audio, first: {missing[:5]}"
        )
    print(json.dumps({
        "rows": written,
        "audio_files": len(audio_by_name),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
