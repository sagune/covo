#!/usr/bin/env python
"""Build hotword-use-focused COVO SFT data by oversampling CB-Whisper train rows."""

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


_PUNCT_RE = re.compile(r"[\s,，。.!！?？:：;；、\"“”‘’《》<>[\]()（）-]+")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    return _PUNCT_RE.sub("", text)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
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


def prompt_hotwords(row: Dict[str, Any]) -> List[str]:
    input_block = row.get("input", {}) or {}
    output = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if not isinstance(item, dict):
            continue
        text = norm(item.get("text", ""))
        if text:
            output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def target_hotwords(row: Dict[str, Any]) -> List[str]:
    reference = norm(row.get("reference", ""))
    return [keyword for keyword in prompt_hotwords(row) if keyword and keyword in reference]


def false_hotwords_in_asr(row: Dict[str, Any]) -> List[str]:
    input_block = row.get("input", {}) or {}
    reference = norm(row.get("reference", ""))
    asr_top1 = norm(input_block.get("asr_top1", ""))
    return [keyword for keyword in prompt_hotwords(row) if keyword and keyword not in reference and keyword in asr_top1]


def hotword_use_needed(row: Dict[str, Any]) -> List[str]:
    input_block = row.get("input", {}) or {}
    asr_top1 = norm(input_block.get("asr_top1", ""))
    return [keyword for keyword in target_hotwords(row) if keyword and keyword not in asr_top1]


def has_assistant(row: Dict[str, Any]) -> bool:
    messages = row.get("messages", []) or []
    return bool(messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--use-repeat", type=int, default=5)
    parser.add_argument("--false-repeat", type=int, default=1)
    parser.add_argument("--max-use-rows", type=int, default=0)
    parser.add_argument("--max-false-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [row for row in read_jsonl(args.input) if has_assistant(row)]
    use_rows = [row for row in rows if hotword_use_needed(row)]
    false_rows = [row for row in rows if false_hotwords_in_asr(row)]

    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    rng.shuffle(use_rows)
    rng.shuffle(false_rows)

    if int(args.max_use_rows) > 0:
        use_rows = use_rows[: int(args.max_use_rows)]
    if int(args.max_false_rows) > 0:
        false_rows = false_rows[: int(args.max_false_rows)]

    output_rows: List[Dict[str, Any]] = list(rows)
    for row in use_rows:
        for _ in range(max(0, int(args.use_repeat))):
            output_rows.append(row)
    for row in false_rows:
        for _ in range(max(0, int(args.false_repeat))):
            output_rows.append(row)
    rng.shuffle(output_rows)

    written = write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "base_rows": len(rows),
                "hotword_use_rows": len(use_rows),
                "false_hotword_rows": len(false_rows),
                "use_repeat": int(args.use_repeat),
                "false_repeat": int(args.false_repeat),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
