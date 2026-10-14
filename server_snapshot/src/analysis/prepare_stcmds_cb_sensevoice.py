#!/usr/bin/env python3
"""Build the generic hotword-dataset layout used by CB-SenseVoice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def read_keywords(path: Path) -> list[str]:
    keywords = []
    seen = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        keyword = line.split("\t", 1)[0].strip()
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)
    if not keywords:
        raise ValueError(f"no keywords found in {path}")
    return keywords


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--keywords", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--keyword-hs-source", type=Path)
    parser.add_argument("--keyword-audio-source", type=Path)
    parser.add_argument("--utterance-hs-source", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:max(0, args.limit)]
    keywords = read_keywords(args.keywords)
    split_dir = args.output_root / "hotword" / args.split
    wav_dir = args.output_root / "wav" / args.split
    utterance_hs = split_dir / "hs"
    keyword_hs = split_dir / "keywords-hs" / "tts"
    for directory in (split_dir, wav_dir, utterance_hs, keyword_hs):
        directory.mkdir(parents=True, exist_ok=True)

    text_lines = []
    reused_utterance_hs = 0
    missing_utterance_hs = 0
    for row in rows:
        code = str(row["id"])
        reference = str(row["reference"]).strip()
        wav = Path(row["wav"])
        if not wav.exists():
            raise FileNotFoundError(wav)
        text_lines.append(f"{code} {reference}")
        relative_symlink(wav, wav_dir / f"{code}{wav.suffix.lower()}")
        if args.utterance_hs_source is not None:
            source = args.utterance_hs_source / f"{code}.bin"
            if source.exists():
                relative_symlink(source, utterance_hs / source.name)
                reused_utterance_hs += 1
            else:
                missing_utterance_hs += 1

    (split_dir / "text").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (split_dir / "uttid").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (split_dir / "hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    (split_dir / "r1-hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    aligned = [
        (keyword, str(row["id"]))
        for row in rows
        for keyword in keywords
        if keyword in str(row["reference"])
    ]
    (split_dir / "aligned.txt").write_text(
        "\n".join(f"{keyword}\t{utterance_id}" for keyword, utterance_id in aligned)
        + ("\n" if aligned else ""),
        encoding="utf-8",
    )

    if args.keyword_hs_source is not None:
        width = len(str(len(keywords) - 1))
        for index in range(len(keywords)):
            source = args.keyword_hs_source / f"{index:0{width}d}.bin"
            if not source.exists():
                raise FileNotFoundError(source)
            relative_symlink(source, keyword_hs / source.name)

    if args.keyword_audio_source is not None:
        audio_target = split_dir / "keywords-audios" / "tts"
        audio_target.mkdir(parents=True, exist_ok=True)
        width = len(str(len(keywords) - 1))
        for index in range(len(keywords)):
            stem = f"{index:0{width}d}"
            matches = [
                args.keyword_audio_source / f"{stem}{extension}"
                for extension in (".mp3", ".wav", ".opus")
            ]
            source = next((candidate for candidate in matches if candidate.exists()), None)
            if source is None:
                raise FileNotFoundError(matches[0])
            relative_symlink(source, audio_target / source.name)

    available_utterance_hs = sum((utterance_hs / f"{row['id']}.bin").exists() for row in rows)
    print(json.dumps({
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "split": args.split,
        "utterances": len(rows),
        "keywords": len(keywords),
        "positive_utterances": len({utterance_id for _, utterance_id in aligned}),
        "keyword_mentions": len(aligned),
        "reused_utterance_hidden_states": reused_utterance_hs,
        "missing_utterance_hidden_states": missing_utterance_hs,
        "available_utterance_hidden_states": available_utterance_hs,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
