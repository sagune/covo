#!/usr/bin/env python
"""Build synthetic oral-style ASR correction data from MagicData-RAMC text."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List

from pypinyin import Style, lazy_pinyin


PUNCT_RE = re.compile(r"[\s，。？！、,.!?；;：:“”\"'（）()\[\]{}<>《》]")
SPAN_RE = re.compile(r"^\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]\t")

FILLERS = [
    "然后就是",
    "就是说",
    "就是",
    "然后",
    "那个",
    "这个",
    "的话",
    "那么",
    "咱们",
    "咱",
    "嗯",
    "呃",
    "啊",
    "呢",
    "嘛",
    "呀",
    "吧",
]

FALSE_INSERTIONS = [
    "水利工程",
    "闸门",
    "地基",
    "石方",
    "施工技术",
    "基坑",
    "开挖",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-root", required=True, help="Path to MDT2021S003/TXT")
    parser.add_argument("--raw-output", required=True, help="Internal JSONL output")
    parser.add_argument("--train-output", required=True, help="Train JSONL output")
    parser.add_argument("--dev-output", required=True, help="Dev JSONL output")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--dev-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-chars", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=48)
    return parser.parse_args()


def norm(text: str) -> str:
    text = text.replace("[+]", "").replace("[*]", "")
    return PUNCT_RE.sub("", text).strip()


def iter_segments(text_root: Path) -> Iterable[dict]:
    for txt in sorted(text_root.glob("*.txt")):
        session = txt.stem
        for line_idx, line in enumerate(txt.read_text(encoding="utf-8-sig").splitlines()):
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            if parts[3] in {"[*]", "[+]"}:
                continue
            match = SPAN_RE.match(line)
            start = float(match.group(1)) if match else 0.0
            end = float(match.group(2)) if match else 0.0
            yield {
                "id": f"{session}_{line_idx:04d}",
                "session": session,
                "speaker": parts[1],
                "speaker_desc": parts[2],
                "start": start,
                "end": end,
                "text": parts[3],
            }


def remove_one_filler(text: str, rng: random.Random) -> str:
    present = [item for item in FILLERS if item in text]
    if not present:
        return text
    filler = rng.choice(present)
    return text.replace(filler, "", 1)


def remove_fillers(text: str, rng: random.Random, max_remove: int = 3) -> str:
    output = text
    present = [item for item in FILLERS if item in output]
    rng.shuffle(present)
    removed = 0
    for filler in present:
        if filler in output and removed < max_remove:
            output = output.replace(filler, "", 1)
            removed += 1
    return output or text


def shorten(text: str, rng: random.Random) -> str:
    if len(text) <= 8:
        return text
    mode = rng.choice(["prefix", "suffix", "middle"])
    if mode == "prefix":
        cut = rng.randint(1, min(5, len(text) - 4))
        return text[cut:]
    if mode == "suffix":
        keep = rng.randint(max(4, len(text) - 8), len(text) - 1)
        return text[:keep]
    start = rng.randint(0, max(0, len(text) // 3))
    end = rng.randint(max(start + 4, len(text) // 2), len(text))
    return text[start:end]


def insert_false_word(text: str, rng: random.Random) -> str:
    word = rng.choice(FALSE_INSERTIONS)
    if not text:
        return word
    pos = rng.randint(0, len(text))
    return text[:pos] + word + text[pos:]


def duplicate_noise(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    start = rng.randint(0, len(text) - 2)
    end = rng.randint(start + 1, min(len(text), start + 4))
    return text[:end] + text[start:end] + text[end:]


def make_pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        item = norm(item)
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def build_record(seg: dict, rng: random.Random) -> dict | None:
    ref = norm(seg["text"])
    if not ref:
        return None
    variants = [
        ref,
        remove_one_filler(ref, rng),
        remove_fillers(ref, rng, max_remove=2),
        remove_fillers(ref, rng, max_remove=4),
        shorten(ref, rng),
        duplicate_noise(ref, rng),
    ]
    if rng.random() < 0.35:
        variants.append(insert_false_word(remove_fillers(ref, rng, max_remove=2), rng))
    if rng.random() < 0.25:
        variants.append(shorten(remove_fillers(ref, rng, max_remove=3), rng))
    variants = unique(variants)
    if len(variants) < 2:
        return None

    if rng.random() < 0.18:
        top1 = ref
    else:
        non_ref = [item for item in variants if item != ref]
        top1 = rng.choice(non_ref or variants)
    nbest = unique([top1] + rng.sample(variants, k=len(variants)))
    if ref not in nbest:
        insert_at = rng.randint(1, min(len(nbest), 5))
        nbest.insert(insert_at, ref)
    nbest = nbest[:10]

    return {
        "id": seg["id"],
        "source": "magicdata_ramc_synthetic_oral",
        "split": "",
        "reference": ref,
        "input": {
            "asr_top1": top1,
            "nbest": nbest,
            "nbest_pinyin": [make_pinyin(item) for item in nbest],
            "metadata": {
                "session": seg["session"],
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "original_text": seg["text"],
            },
        },
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    records = []
    for seg in iter_segments(Path(args.text_root)):
        ref = norm(seg["text"])
        if not (args.min_chars <= len(ref) <= args.max_chars):
            continue
        record = build_record(seg, rng)
        if record is not None:
            records.append(record)
    rng.shuffle(records)
    train = records[: args.train_size]
    dev = records[args.train_size : args.train_size + args.dev_size]
    for record in train:
        record["split"] = "train"
    for record in dev:
        record["split"] = "dev"
    raw = train + dev
    raw_count = write_jsonl(Path(args.raw_output), raw)
    train_count = write_jsonl(Path(args.train_output), train)
    dev_count = write_jsonl(Path(args.dev_output), dev)
    print(
        json.dumps(
            {
                "raw_output": args.raw_output,
                "train_output": args.train_output,
                "dev_output": args.dev_output,
                "available_records": len(records),
                "raw_written": raw_count,
                "train_written": train_count,
                "dev_written": dev_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
