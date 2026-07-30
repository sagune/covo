#!/usr/bin/env python3
"""Extract WeNetSpeech evaluation parquet files into audio and JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def audio_extension(path_hint: str, payload: bytes) -> str:
    suffix = Path(path_hint).suffix.lower()
    if suffix in {".wav", ".flac", ".mp3", ".opus", ".ogg"}:
        return suffix
    if payload.startswith(b"RIFF"):
        return ".wav"
    if payload.startswith(b"fLaC"):
        return ".flac"
    if payload.startswith(b"OggS"):
        return ".opus"
    return ".audio"


def first_value(row: dict, *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for parquet_path in args.parquet:
            parquet = pq.ParquetFile(parquet_path)
            available = set(parquet.schema_arrow.names)
            columns = [
                name for name in ("sid", "utt_id", "id", "text", "sentence", "WavPath", "audio_path", "audio")
                if name in available
            ]
            for batch in parquet.iter_batches(columns=columns, batch_size=256):
                for row in batch.to_pylist():
                    audio = row.get("audio") or {}
                    payload = audio.get("bytes") if isinstance(audio, dict) else None
                    reference = first_value(row, "text", "sentence")
                    utt_id = first_value(row, "sid", "utt_id", "id")
                    path_hint = first_value(row, "WavPath", "audio_path")
                    if not payload or not reference or not utt_id:
                        skipped += 1
                        continue
                    output_path = args.audio_dir / f"{utt_id}{audio_extension(path_hint, payload)}"
                    if not output_path.exists():
                        output_path.write_bytes(payload)
                    writer.write(json.dumps({
                        "id": utt_id,
                        "reference": reference,
                        "wav": str(output_path),
                        "source": "WeNetSpeech",
                        "split": args.split,
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
    print(json.dumps({
        "rows": written,
        "skipped": skipped,
        "split": args.split,
        "output": str(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
