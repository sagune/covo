#!/usr/bin/env python3
"""Extract a Speechio-Formal parquet subset into wavs and evaluation JSONL."""

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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["idx", "subset_id", "original_text", "target_text", "audio"])
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            utt_id = f"{row['subset_id']}_{int(row['idx']):06d}"
            wav_path = args.audio_dir / f"{utt_id}.wav"
            audio = row["audio"] or {}
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not audio_bytes:
                raise ValueError(f"missing embedded audio for {utt_id}")
            if not wav_path.exists():
                wav_path.write_bytes(audio_bytes)
            target = str(row.get("target_text") or "").strip()
            original = str(row.get("original_text") or "").strip()
            if not target or not original:
                continue
            handle.write(
                json.dumps(
                    {
                        "id": utt_id,
                        "subset_id": row["subset_id"],
                        "reference": target,
                        "verbatim_reference": original,
                        "wav": str(wav_path),
                        "source": "Speechio-Formal",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    print(json.dumps({"rows": len(rows), "output": str(args.output), "audio_dir": str(args.audio_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
