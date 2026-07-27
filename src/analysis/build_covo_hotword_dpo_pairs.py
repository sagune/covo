#!/usr/bin/env python
"""Build DPO pairs that prefer preserving CB-SenseVoice hotword evidence."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def norm_text(value: Any) -> str:
    return "".join(str(value or "").split()).replace("，", "").replace(",", "").replace("。", "")


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def allowed_hotwords(input_block: Dict[str, Any], source: str) -> set[str]:
    source = str(source).strip().lower()
    if source == "prompt":
        rows = input_block.get("prompt_hotwords", []) or []
    elif source == "covo":
        rows = input_block.get("covo_hotwords", []) or []
    elif source == "all":
        rows = list(input_block.get("prompt_hotwords", []) or []) + list(input_block.get("hotwords", []) or [])
    else:
        raise ValueError(f"unsupported hotword source: {source}")
    return {norm_text(row.get("text", "")) for row in rows if isinstance(row, dict)}


def mentioned_hotwords(record: Dict[str, Any], source: str, min_len: int) -> List[str]:
    input_block = record.get("input", {}) or {}
    reference = norm_text(record.get("reference", ""))
    allowed = allowed_hotwords(input_block, source)
    keywords = []
    for mention in input_block.get("keyword_mentions", []) or []:
        if not isinstance(mention, dict):
            continue
        text = norm_text(mention.get("mention", ""))
        if len(text) >= int(min_len) and text in reference and text in allowed:
            keywords.append(text)
    return sorted(set(keywords), key=lambda item: (-len(item), item))


def prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [{"role": str(row.get("role", "")), "content": str(row.get("content", ""))} for row in messages]


def make_pair(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    reference = norm_text(record.get("reference", ""))
    if not reference:
        return None
    keywords = mentioned_hotwords(record, args.hotword_source, int(args.min_hotword_len))
    if not keywords:
        return None

    input_block = record.get("input", {}) or {}
    nbest = [norm_text(item) for item in input_block.get("nbest", []) or []]
    max_distance = max(int(args.min_edit_distance), int(len(reference) * float(args.max_edit_distance_ratio)))
    best_rejected = None
    best_missing: List[str] = []
    best_distance = 10**9
    for candidate in nbest[1 : max(2, int(args.max_nbest_scan))]:
        if not candidate or candidate == reference:
            continue
        missing = [keyword for keyword in keywords if keyword not in candidate]
        if not missing:
            continue
        distance = edit_distance(reference, candidate)
        if distance <= max_distance and distance < best_distance:
            best_rejected = candidate
            best_missing = missing
            best_distance = distance
    if best_rejected is None:
        return None

    return {
        "id": str(record.get("id", "")),
        "source": record.get("source", "cbwhisper"),
        "split": record.get("split", ""),
        "pair_type": "hotword_preserve_candidate_dpo",
        "missing_hotwords_in_rejected": best_missing,
        "edit_distance": best_distance,
        "prompt_messages": prompt_messages(record),
        "chosen": {"text": reference},
        "rejected": {"text": best_rejected},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hotword-source", choices=["prompt", "covo", "all"], default="prompt")
    parser.add_argument("--min-hotword-len", type=int, default=2)
    parser.add_argument("--max-nbest-scan", type=int, default=8)
    parser.add_argument("--min-edit-distance", type=int, default=2)
    parser.add_argument("--max-edit-distance-ratio", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for record in read_jsonl(args.input) if (row := make_pair(record, args)) is not None]
    random.Random(int(args.seed)).shuffle(rows)
    if args.limit:
        rows = rows[: int(args.limit)]
    written = write_jsonl(args.output, rows)
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
