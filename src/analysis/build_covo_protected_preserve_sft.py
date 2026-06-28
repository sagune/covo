#!/usr/bin/env python
"""Convert protected-hotword preservation preference pairs to repeated SFT rows."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def make_sft_rows(pair: Dict[str, Any], repeats: int) -> List[Dict[str, Any]]:
    chosen = pair.get("chosen", {})
    if not isinstance(chosen, dict) or not str(chosen.get("text", "")).strip():
        return []
    messages = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))}
        for item in pair.get("prompt_messages", [])
        if isinstance(item, dict)
    ]
    if not messages:
        return []
    messages.append(
        {
            "role": "assistant",
            "content": json.dumps({"text": str(chosen.get("text", "")).strip()}, ensure_ascii=False, separators=(",", ":")),
        }
    )
    rows = []
    for copy_idx in range(max(1, int(repeats))):
        rows.append(
            {
                "id": f"{pair.get('id', '')}:protected_preserve_sft:{copy_idx}",
                "source": pair.get("source", "cbwhisper"),
                "split": pair.get("split", ""),
                "messages": messages,
                "protected_preserve_sft": {
                    "lost_hotwords": pair.get("lost_protected_hotwords", []),
                    "base_d": pair.get("base_d"),
                    "pred_d": pair.get("pred_d"),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for pair in read_jsonl(args.pairs):
        rows.extend(make_sft_rows(pair, int(args.repeats)))
    random.Random(int(args.seed)).shuffle(rows)
    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "pairs": args.pairs,
                "output": args.output,
                "repeats": int(args.repeats),
                "written": written,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
