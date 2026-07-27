#!/usr/bin/env python
"""Materialize AISHELL train as a CB-SenseVoice hotword split.

The CB-SenseVoice evaluator expects a hotword-style split with text, uttid,
utterance hidden states, and keyword hidden states. AISHELL train already has
the needed alignments and KWS hidden states, so this script only creates the
small metadata files and symlinks the heavy feature folders.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def norm_text(text: str) -> str:
    return "".join(str(text or "").split())


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def read_transcripts(path: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            transcripts[parts[0]] = norm_text(parts[1])
    return transcripts


def read_alignments(path: Path) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, List[str]]]:
    rows: List[Tuple[str, str, str, str]] = []
    by_utt: Dict[str, List[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.strip().split()
            if len(parts) < 4:
                continue
            keyword = norm_text(parts[0])
            utt_id = parts[1].strip()
            start = parts[2].strip()
            end = parts[3].strip()
            if not keyword or not utt_id:
                continue
            rows.append((keyword, utt_id, start, end))
            if keyword not in by_utt[utt_id]:
                by_utt[utt_id].append(keyword)
    return rows, dict(by_utt)


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def select_probe(
    by_utt: Dict[str, List[str]],
    transcripts: Dict[str, str],
    hs_dir: Path,
    all_keywords: List[str],
    limit_utterances: int,
    utterance_offset: int,
    max_keywords: int,
    seed: int,
    selection: str,
) -> Tuple[List[str], List[str]]:
    usable_utts = [
        utt_id
        for utt_id in by_utt
        if utt_id in transcripts and (hs_dir / f"{utt_id}.bin").exists()
    ]
    usable_utts = sorted(usable_utts)
    if selection == "random" and limit_utterances and limit_utterances < len(usable_utts):
        rng = random.Random(seed)
        usable_utts = sorted(rng.sample(usable_utts, limit_utterances))
    elif selection == "sequential":
        start = max(0, int(utterance_offset))
        end = start + int(limit_utterances) if limit_utterances else None
        usable_utts = usable_utts[start:end]
    elif selection != "all":
        raise ValueError(f"unsupported selection `{selection}`, expected all|random|sequential")

    selected_keyword_set: Set[str] = set()
    for utt_id in usable_utts:
        selected_keyword_set.update(by_utt.get(utt_id, []))

    if max_keywords and max_keywords > len(selected_keyword_set):
        rng = random.Random(seed + 17)
        candidates = [keyword for keyword in all_keywords if keyword not in selected_keyword_set]
        fill = rng.sample(candidates, min(len(candidates), max_keywords - len(selected_keyword_set)))
        selected_keyword_set.update(fill)
    elif max_keywords and max_keywords < len(selected_keyword_set):
        selected_keyword_set = set(sorted(selected_keyword_set, key=lambda item: (-len(item), item))[:max_keywords])

    selected_keywords = [keyword for keyword in all_keywords if keyword in selected_keyword_set]
    return usable_utts, selected_keywords


def replace_with_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.relpath(target_path, link_path.parent), link_path)


def maybe_rebuild_keyword_symlinks(
    output_dir: Path,
    source_keywords: List[str],
    target_keywords: List[str],
    source_hs_root: Path,
    kw_type: str,
) -> str:
    source_dir = source_hs_root / kw_type
    target_dir = output_dir / "keywords-hs" / kw_type
    if source_keywords == target_keywords:
        replace_with_symlink(target_dir, source_dir)
        return "whole-dir"

    if target_dir.is_symlink() or target_dir.exists():
        if target_dir.is_dir() and not target_dir.is_symlink():
            shutil.rmtree(target_dir)
        else:
            target_dir.unlink()
    target_dir.mkdir(parents=True, exist_ok=True)
    source_index = {keyword: idx for idx, keyword in enumerate(source_keywords)}
    source_width = len(str(len(source_keywords) - 1))
    target_width = len(str(len(target_keywords) - 1))
    linked = 0
    for target_idx, keyword in enumerate(target_keywords):
        source_idx = source_index.get(keyword)
        if source_idx is None:
            continue
        source_file = source_dir / f"{source_idx:0{source_width}d}.bin"
        target_file = target_dir / f"{target_idx:0{target_width}d}.bin"
        if source_file.exists():
            os.symlink(os.path.relpath(source_file, target_file.parent), target_file)
            linked += 1
    manifest = source_dir / "_hs_manifest.json"
    if manifest.exists():
        os.symlink(os.path.relpath(manifest, target_dir), target_dir / "_hs_manifest.json")
    return f"per-keyword:{linked}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../datasets/aishell/data_aishell")
    parser.add_argument("--train-aligned", default="../datasets/aishell/train/aligned.txt")
    parser.add_argument("--train-keywords", default="../datasets/aishell/train/keywords.txt")
    parser.add_argument("--transcript", default="../datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt")
    parser.add_argument("--kws-root", default="../datasets/aishell/data_aishell/kws")
    parser.add_argument("--output-split", default="train")
    parser.add_argument("--kw-types", default="tts,natural")
    parser.add_argument("--limit-utterances", type=int, default=0)
    parser.add_argument("--utterance-offset", type=int, default=0)
    parser.add_argument("--max-keywords", type=int, default=0)
    parser.add_argument("--selection", choices=["all", "random", "sequential"], default="random")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    kws_root = Path(args.kws_root).resolve()
    output_dir = data_root / "hotword" / args.output_split
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to rebuild it")
    output_dir.mkdir(parents=True, exist_ok=True)

    alignments, by_utt = read_alignments(Path(args.train_aligned).resolve())
    transcripts = read_transcripts(Path(args.transcript).resolve())
    all_train_keywords = [norm_text(line) for line in read_lines(Path(args.train_keywords).resolve())]
    source_keywords = [norm_text(line) for line in read_lines(kws_root / "keywords.txt")]

    usable_utts, target_keywords = select_probe(
        by_utt=by_utt,
        transcripts=transcripts,
        hs_dir=kws_root / "hs",
        all_keywords=all_train_keywords,
        limit_utterances=args.limit_utterances,
        utterance_offset=args.utterance_offset,
        max_keywords=args.max_keywords,
        seed=args.seed,
        selection=args.selection,
    )
    usable_set = set(usable_utts)
    keyword_set = set(target_keywords)

    write_text(output_dir / "aligned.txt", ["\t".join(row) for row in alignments if row[1] in usable_set and row[0] in keyword_set])
    write_text(output_dir / "hotword.txt", target_keywords)
    write_text(output_dir / "r1-hotword.txt", target_keywords)
    write_text(output_dir / "uttid", usable_utts)
    write_text(output_dir / "text", [f"{utt_id}\t{transcripts[utt_id]}" for utt_id in usable_utts])
    replace_with_symlink(output_dir / "hs", kws_root / "hs")
    replace_with_symlink(data_root / "wav" / args.output_split, data_root / "wav" / "train")

    link_modes = {}
    for kw_type in [item.strip() for item in args.kw_types.split(",") if item.strip()]:
        link_modes[kw_type] = maybe_rebuild_keyword_symlinks(
            output_dir=output_dir,
            source_keywords=source_keywords,
            target_keywords=target_keywords,
            source_hs_root=kws_root / "keywords-hs",
            kw_type=kw_type,
        )

    print(
        json.dumps(
            {
                "output": str(output_dir),
                "alignments": len(alignments),
                "aligned_utterances": len(by_utt),
                "usable_utterances": len(usable_utts),
                "keywords": len(target_keywords),
                "selection": args.selection,
                "limit_utterances": args.limit_utterances,
                "utterance_offset": args.utterance_offset,
                "max_keywords": args.max_keywords,
                "keyword_hs_link_modes": link_modes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
