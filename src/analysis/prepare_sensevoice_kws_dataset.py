#!/usr/bin/env python
"""Prepare an AISHELL-style KWS dataset root for SenseVoice hidden states.

The script preserves the original CB-Whisper dataset metadata and keyword
audio assets through symlinks, while creating fresh ``hs`` and ``keywords-hs``
directories for SenseVoice encoder states.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def rel_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    link_target = os.path.relpath(source.resolve(), start=target.parent.resolve())
    target.symlink_to(link_target, target_is_directory=source.is_dir())


def link_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if source.is_file():
            rel_symlink(source, target_dir / source.name)


def prepare_kws(source: Path, target: Path) -> None:
    source_kws = source / "kws"
    target_kws = target / "kws"
    link_files(source_kws, target_kws)
    rel_symlink(source_kws / "keywords-audios", target_kws / "keywords-audios")
    (target_kws / "hs").mkdir(parents=True, exist_ok=True)
    for kw_type in ("natural", "tts"):
        (target_kws / "keywords-hs" / kw_type).mkdir(parents=True, exist_ok=True)


def prepare_hotword_split(source: Path, target: Path, split: str) -> None:
    source_split = source / "hotword" / split
    target_split = target / "hotword" / split
    link_files(source_split, target_split)
    rel_symlink(source_split / "keywords-audios", target_split / "keywords-audios")
    (target_split / "hs").mkdir(parents=True, exist_ok=True)
    for kw_type in ("natural", "tts"):
        (target_split / "keywords-hs" / kw_type).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="datasets/aishell/data_aishell")
    parser.add_argument("--target", default="datasets/aishell/data_aishell_sensevoice")
    parser.add_argument("--splits", default="dev,test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    target = Path(args.target)
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    rel_symlink(source / "wav", target / "wav")
    if (source / "transcript").exists():
        rel_symlink(source / "transcript", target / "transcript")
    prepare_kws(source, target)
    for split in [part.strip() for part in args.splits.split(",") if part.strip()]:
        prepare_hotword_split(source, target, split)
    print(f"prepared SenseVoice KWS dataset root: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
