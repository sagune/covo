#!/usr/bin/env python
"""Strip optional filler tokens from COVO evidence JSONL fields."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


DEFAULT_FILLERS = (
    "这一个",
    "那么",
    "的话",
    "这个",
    "那个",
    "我们呢",
    "就是",
    "呢",
    "啊",
    "呃",
    "嗯",
)


def parse_fillers(spec: str) -> List[str]:
    fillers = [item.strip() for item in re.split(r"[,，\s]+", spec) if item.strip()]
    return sorted(set(fillers), key=lambda item: (-len(item), item))


def normalize_and_strip(value: Any, fillers: List[str]) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    for filler in fillers:
        text = text.replace(filler, "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def strip_record(record: Dict[str, Any], fillers: List[str]) -> Dict[str, Any]:
    row = dict(record)
    row["reference"] = normalize_and_strip(row.get("reference", ""), fillers)
    input_block = dict(row.get("input", {}) or {})
    input_block["asr_top1"] = normalize_and_strip(input_block.get("asr_top1", ""), fillers)
    input_block["nbest"] = [normalize_and_strip(item, fillers) for item in input_block.get("nbest", []) or []]
    input_block["nbest"] = [item for item in input_block["nbest"] if item]
    row["input"] = input_block
    row["filler_stripped"] = {"fillers": fillers}
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fillers", default=",".join(DEFAULT_FILLERS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fillers = parse_fillers(args.fillers)
    written = write_jsonl(args.output, (strip_record(row, fillers) for row in read_jsonl(args.input)))
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "fillers": fillers}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
