#!/usr/bin/env python
"""Build SFT rows that teach COVO to preserve true protected hotwords."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


_PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    return _PUNCT_RE.sub("", text)


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


def has_messages(row: Dict[str, Any]) -> bool:
    messages = row.get("messages", []) or []
    return bool(messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant")


def prompt_keywords(row: Dict[str, Any]) -> List[str]:
    input_block = row.get("input", {}) or {}
    output = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if not isinstance(item, dict):
            continue
        keyword = norm(item.get("text", ""))
        if keyword:
            output.append(keyword)
    return sorted(set(output), key=lambda item: (-len(item), item))


def true_protected_keywords(row: Dict[str, Any]) -> List[str]:
    input_block = row.get("input", {}) or {}
    asr_top1 = norm(input_block.get("asr_top1", ""))
    reference = norm(row.get("reference", ""))
    return [keyword for keyword in prompt_keywords(row) if keyword in asr_top1 and keyword in reference]


def lost_true_protected_keywords(row: Dict[str, Any]) -> List[str]:
    prediction = norm(row.get("prediction", ""))
    return [keyword for keyword in true_protected_keywords(row) if keyword not in prediction]


def make_sft_row(row: Dict[str, Any], suffix: str) -> Dict[str, Any] | None:
    messages = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in row.get("messages", [])
        if isinstance(item, dict)
    ]
    if len(messages) < 2:
        return None
    target = norm(row.get("reference", ""))
    if not target:
        return None
    messages = messages[:-1] + [
        {
            "role": "assistant",
            "content": json.dumps({"text": target}, ensure_ascii=False, separators=(",", ":")),
        }
    ]
    true_keywords = true_protected_keywords(row)
    return {
        "id": f"{row.get('id', '')}:protected_positive:{suffix}",
        "source": row.get("source", "cbwhisper"),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "messages": messages,
        "protected_positive": {
            "true_protected_hotwords": true_keywords,
            "lost_by_base_model": lost_true_protected_keywords(row),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-limit", type=int, default=6000)
    parser.add_argument("--positive-repeat", type=int, default=1)
    parser.add_argument("--lost-repeat", type=int, default=30)
    parser.add_argument("--anchor-file", default="")
    parser.add_argument("--anchor-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    rows = [row for row in read_jsonl(args.input) if has_messages(row)]
    positive_rows = [row for row in rows if true_protected_keywords(row)]
    lost_rows = [row for row in positive_rows if lost_true_protected_keywords(row)]
    rng.shuffle(positive_rows)
    rng.shuffle(lost_rows)
    if int(args.positive_limit) > 0:
        positive_rows = positive_rows[: int(args.positive_limit)]

    output_rows: List[Dict[str, Any]] = []
    for row in positive_rows:
        for repeat_idx in range(max(1, int(args.positive_repeat))):
            sft_row = make_sft_row(row, f"positive:{repeat_idx}")
            if sft_row is not None:
                output_rows.append(sft_row)
    for row in lost_rows:
        for repeat_idx in range(max(0, int(args.lost_repeat))):
            sft_row = make_sft_row(row, f"lost:{repeat_idx}")
            if sft_row is not None:
                output_rows.append(sft_row)

    anchor_rows: List[Dict[str, Any]] = []
    if args.anchor_file and int(args.anchor_limit) > 0:
        anchor_rows = [row for row in read_jsonl(args.anchor_file) if has_messages(row)]
        rng.shuffle(anchor_rows)
        anchor_rows = anchor_rows[: int(args.anchor_limit)]
        output_rows.extend(anchor_rows)

    rng.shuffle(output_rows)
    written = write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "base_rows": len(rows),
                "positive_rows_available": len([row for row in rows if true_protected_keywords(row)]),
                "positive_rows_used": len(positive_rows),
                "lost_rows": len(lost_rows),
                "positive_repeat": int(args.positive_repeat),
                "lost_repeat": int(args.lost_repeat),
                "anchor_rows": len(anchor_rows),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
