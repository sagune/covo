#!/usr/bin/env python
"""Build CB-Whisper evidence SFT splits with hotword-preservation oversampling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def norm_text(text: Any) -> str:
    return "".join(str(text or "").split())


def mention_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("mention", "")).strip()
    return str(item or "").strip()


def hotword_preserved(row: Dict[str, Any], require_all: bool = False) -> bool:
    input_block = row.get("input", {}) or {}
    reference = norm_text(row.get("reference", ""))
    asr_top1 = norm_text(input_block.get("asr_top1", ""))
    mentions = [
        norm_text(mention_text(item))
        for item in input_block.get("keyword_mentions", []) or []
    ]
    mentions = [item for item in mentions if item and item in reference]
    if not mentions:
        return False
    hits = [item in asr_top1 for item in mentions]
    return all(hits) if require_all else any(hits)


def assistant_text(row: Dict[str, Any]) -> str:
    messages = row.get("messages", []) or []
    if not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return ""
    return str(last.get("content", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dev-size", type=int, default=99)
    parser.add_argument("--test-size", type=int, default=1)
    parser.add_argument("--preserve-repeat", type=int, default=2)
    parser.add_argument("--require-all-hotwords", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input)
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)

    test_size = max(0, int(args.test_size))
    dev_size = max(0, int(args.dev_size))
    test_rows = rows[:test_size]
    dev_rows = rows[test_size: test_size + dev_size]
    train_base = rows[test_size + dev_size:]

    train_rows: List[Dict[str, Any]] = []
    preserve_count = 0
    skipped = 0
    for row in train_base:
        if not assistant_text(row):
            skipped += 1
            continue
        train_rows.append(row)
        if hotword_preserved(row, require_all=bool(args.require_all_hotwords)):
            preserve_count += 1
            for _ in range(max(0, int(args.preserve_repeat))):
                train_rows.append(row)

    prefix = Path(args.output_prefix)
    train_path = prefix.with_name(prefix.name + "_train.jsonl")
    dev_path = prefix.with_name(prefix.name + "_dev.jsonl")
    test_path = prefix.with_name(prefix.name + "_test.jsonl")
    summary = {
        "input": str(args.input),
        "train_base_rows": len(train_base),
        "train_written": write_jsonl(train_path, train_rows),
        "dev_written": write_jsonl(dev_path, dev_rows),
        "test_written": write_jsonl(test_path, test_rows),
        "preserve_rows_oversampled": preserve_count,
        "preserve_repeat": int(args.preserve_repeat),
        "require_all_hotwords": bool(args.require_all_hotwords),
        "skipped_no_assistant": skipped,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "test_path": str(test_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
