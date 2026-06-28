#!/usr/bin/env python
"""Mix protected-hotword preservation rows with general anchor COVO rows."""

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


def has_messages(row: Dict[str, Any]) -> bool:
    messages = row.get("messages", []) or []
    return bool(messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant")


def sample_rows(path: str, limit: int, rng: random.Random) -> List[Dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if has_messages(row)]
    rng.shuffle(rows)
    if int(limit) > 0:
        rows = rows[: int(limit)]
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preserve-file", required=True)
    parser.add_argument("--anchor-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preserve-repeat", type=int, default=1)
    parser.add_argument("--anchor-limit", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    preserve_base = [row for row in read_jsonl(args.preserve_file) if has_messages(row)]
    preserve_rows: List[Dict[str, Any]] = []
    for _ in range(max(1, int(args.preserve_repeat))):
        preserve_rows.extend(preserve_base)
    anchor_rows = sample_rows(args.anchor_file, int(args.anchor_limit), rng)
    rows = preserve_rows + anchor_rows
    rng.shuffle(rows)
    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "preserve_file": args.preserve_file,
                "anchor_file": args.anchor_file,
                "output": args.output,
                "preserve_base_rows": len(preserve_base),
                "preserve_repeat": int(args.preserve_repeat),
                "preserve_rows": len(preserve_rows),
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
