#!/usr/bin/env python3
"""Prepare full AISHELL-NER test assets for CB-SenseVoice evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


ENTITY_RE = re.compile(r"<([^<>]+)>|\[([^\[\]]+)\]|\(([^()]+)\)")


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))


def parse_annotated_transcript(
    path: Path,
    min_chars: int,
    max_chars: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], Counter[str]]:
    rows: list[tuple[str, str]] = []
    aligned: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        utterance_id, annotated = line.split(maxsplit=1)
        mentions = []
        for match in ENTITY_RE.finditer(annotated):
            mention = next(group for group in match.groups() if group is not None).strip()
            if min_chars <= len(mention) <= max_chars:
                mentions.append(mention)
                aligned.append((mention, utterance_id))
                counts[mention] += 1
        rows.append((utterance_id, ENTITY_RE.sub(lambda match: next(g for g in match.groups() if g), annotated)))
    return rows, aligned, counts


def read_keywords(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_hidden_state_cache(split_dirs: Iterable[Path]) -> dict[str, Path]:
    hidden_states: dict[str, Path] = {}
    for split_dir in split_dirs:
        keywords = read_keywords(split_dir / "hotword.txt")
        if not keywords:
            continue
        width = len(str(len(keywords) - 1))
        hs_dir = split_dir / "keywords-hs" / "tts"
        for index, keyword in enumerate(keywords):
            stem = f"{index:0{width}d}"
            hs_path = hs_dir / f"{stem}.bin"
            if hs_path.exists() and keyword not in hidden_states:
                hidden_states[keyword] = hs_path
    return hidden_states


def build_audio_cache(split_dirs: Iterable[Path]) -> dict[str, Path]:
    audios: dict[str, Path] = {}
    for split_dir in split_dirs:
        keywords = read_keywords(split_dir / "hotword.txt")
        if not keywords:
            continue
        width = len(str(len(keywords) - 1))
        audio_dir = split_dir / "keywords-audios" / "tts"
        for index, keyword in enumerate(keywords):
            stem = f"{index:0{width}d}"
            if keyword not in audios:
                for extension in (".mp3", ".wav", ".opus"):
                    audio_path = audio_dir / f"{stem}{extension}"
                    if audio_path.exists():
                        audios[keyword] = audio_path
                        break
    return audios


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--legacy-split", type=Path, required=True)
    parser.add_argument("--reuse-hs-split", type=Path, action="append", default=[])
    parser.add_argument("--reuse-audio-split", type=Path, action="append", default=[])
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=12)
    args = parser.parse_args()

    rows, aligned, entity_counts = parse_annotated_transcript(
        args.annotations,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    legacy_keywords = read_keywords(args.legacy_split / "hotword.txt")
    keywords = list(dict.fromkeys([*legacy_keywords, *entity_counts.keys()]))

    split_dir = args.output_root / "hotword" / "test"
    hs_dir = split_dir / "hs"
    keyword_hs_dir = split_dir / "keywords-hs" / "tts"
    keyword_audio_dir = split_dir / "keywords-audios" / "tts"
    for directory in (split_dir, hs_dir, keyword_hs_dir, keyword_audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    wav_link = args.output_root / "wav"
    relative_symlink(args.wav_root, wav_link)

    text = "\n".join(f"{utterance_id} {transcript}" for utterance_id, transcript in rows) + "\n"
    (split_dir / "text").write_text(text, encoding="utf-8")
    (split_dir / "uttid").write_text(text, encoding="utf-8")
    (split_dir / "hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    (split_dir / "r1-hotword.txt").write_text("\n".join(keywords) + "\n", encoding="utf-8")
    (split_dir / "aligned.txt").write_text(
        "\n".join(f"{mention}\t{utterance_id}" for mention, utterance_id in aligned) + "\n",
        encoding="utf-8",
    )

    reused_utterance_hs = 0
    legacy_hs_dir = args.legacy_split / "hs"
    for utterance_id, _ in rows:
        source = legacy_hs_dir / f"{utterance_id}.bin"
        if source.exists():
            relative_symlink(source, hs_dir / source.name)
            reused_utterance_hs += 1

    cached_hs = build_hidden_state_cache([args.legacy_split, *args.reuse_hs_split])
    cached_audio = build_audio_cache(args.reuse_audio_split)
    width = len(str(len(keywords) - 1))
    reused_keyword_hs = 0
    reused_keyword_audio = 0
    missing_keywords = []
    for index, keyword in enumerate(keywords):
        stem = f"{index:0{width}d}"
        if keyword in cached_hs:
            relative_symlink(cached_hs[keyword], keyword_hs_dir / f"{stem}.bin")
            reused_keyword_hs += 1
        else:
            missing_keywords.append(keyword)
        if keyword in cached_audio:
            source = cached_audio[keyword]
            relative_symlink(source, keyword_audio_dir / f"{stem}{source.suffix.lower()}")
            reused_keyword_audio += 1

    (split_dir / "missing_keywords.txt").write_text(
        "\n".join(missing_keywords) + ("\n" if missing_keywords else ""),
        encoding="utf-8",
    )
    summary = {
        "annotations": str(args.annotations),
        "output_root": str(args.output_root),
        "utterances": len(rows),
        "positive_utterances": len({utterance_id for _, utterance_id in aligned}),
        "entity_mentions": len(aligned),
        "unique_entities": len(entity_counts),
        "legacy_keywords": len(legacy_keywords),
        "total_keywords": len(keywords),
        "reused_utterance_hidden_states": reused_utterance_hs,
        "reused_keyword_hidden_states": reused_keyword_hs,
        "reused_keyword_audios": reused_keyword_audio,
        "missing_keyword_hidden_states": len(missing_keywords),
        "protocol": "oracle context derived from AISHELL-NER test annotations",
    }
    (split_dir / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
