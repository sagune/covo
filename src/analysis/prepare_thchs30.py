#!/usr/bin/env python3
"""Extract the THCHS-30 Hugging Face parquet test split for local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["audio", "sentence"])
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(table.to_pylist()):
            audio = row.get("audio") or {}
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            reference = str(row.get("sentence") or "").strip()
            if not audio_bytes or not reference:
                continue
            utt_id = f"thchs30_test_{idx:05d}"
            wav_path = args.audio_dir / f"{utt_id}.wav"
            if not wav_path.exists():
                wav_path.write_bytes(audio_bytes)
            handle.write(json.dumps({
                "id": utt_id,
                "reference": reference,
                "wav": str(wav_path),
                "source": "THCHS-30",
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": idx + 1, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
