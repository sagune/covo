#!/usr/bin/env python
"""Build audio/reference manifest JSONL for Whisper-v3 decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Kaldi-style text file: utt_id transcript")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-dir", default="", help="Directory containing wav/flac/mp3 files named by utt_id")
    parser.add_argument("--wav-scp", default="", help="Kaldi-style wav.scp: utt_id path-or-command")
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument("--must-exist", action="store_true")
    return parser.parse_args()


def _read_text(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected 'utt_id transcript'")
            out[parts[0]] = parts[1]
    return out


def _read_wav_scp(path: str) -> Dict[str, str]:
    out = {}
    if not path:
        return out
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_no}: expected 'utt_id audio_path'")
            out[parts[0]] = parts[1]
    return out


def main() -> int:
    args = parse_args()
    refs = _read_text(args.text)
    wav_scp = _read_wav_scp(args.wav_scp)
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    records = []
    missing = []
    for utt_id, reference in refs.items():
        if utt_id in wav_scp:
            audio = wav_scp[utt_id]
        elif audio_dir is not None:
            audio = str(audio_dir / f"{utt_id}{args.audio_ext}")
        else:
            audio = ""
        if args.must_exist and audio and not Path(audio).exists():
            missing.append(audio)
            continue
        records.append({"id": utt_id, "audio": audio, "reference": reference})

    written = write_jsonl(args.output, records)
    print(json.dumps({"text": args.text, "output": args.output, "written": written, "missing": len(missing)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
