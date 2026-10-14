#!/usr/bin/env python
"""Normalize COVO final-text predictions for diagnostic CER experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


SHUILI_DOMAIN_REPLACEMENTS = (
    ("参见单位", "参建单位"),
    ("重力是码头", "重力式码头"),
    ("Volume 5", "挖运方案"),
    ("开开发", "开挖"),
    ("进行的开发", "进行开挖"),
    ("土石敌方", "土石堤防"),
    ("为海造田", "围海造田"),
)


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collapse_repeats(text: str, max_chunk: int = 12) -> str:
    changed = True
    while changed:
        changed = False
        for length in range(min(max_chunk, len(text) // 2), 1, -1):
            idx = 0
            output = []
            local_change = False
            while idx < len(text):
                chunk = text[idx : idx + length]
                if len(chunk) == length and text[idx + length : idx + 2 * length] == chunk:
                    output.append(chunk)
                    next_idx = idx + length
                    while text[next_idx : next_idx + length] == chunk:
                        next_idx += length
                    idx = next_idx
                    local_change = True
                else:
                    output.append(text[idx])
                    idx += 1
            if local_change:
                text = "".join(output)
                changed = True
                break
    return text


def build_opencc():
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:
        return None


def normalize_prediction(text: str, args: argparse.Namespace, opencc: Any) -> str:
    if args.opencc_t2s and opencc is not None:
        text = opencc.convert(text)
    if args.collapse_repeats:
        text = collapse_repeats(text, max_chunk=int(args.max_repeat_chunk))
    if args.shuili_domain_normalize:
        for src, tgt in SHUILI_DOMAIN_REPLACEMENTS:
            text = text.replace(src, tgt)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--opencc-t2s", action="store_true")
    parser.add_argument("--collapse-repeats", action="store_true")
    parser.add_argument("--max-repeat-chunk", type=int, default=12)
    parser.add_argument("--shuili-domain-normalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    opencc = build_opencc()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    changed = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in read_jsonl(args.input):
            old = str(record.get(args.prediction_field, ""))
            new = normalize_prediction(old, args, opencc)
            if new != old:
                changed += 1
            record[args.prediction_field] = new
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
    print(json.dumps({"input": args.input, "output": args.output, "rows": rows, "changed": changed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
