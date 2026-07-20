#!/usr/bin/env python3
"""Rebalance ChineseHP-style SFT toward recoverable N-best corrections."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recoverable-copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    rows = []
    counts = {"top1_exact": 0, "recoverable_error": 0, "outside_nbest_error": 0}
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            input_block = row.get("input", {}) or {}
            top1 = str(input_block.get("asr_top1", ""))
            reference = str(row.get("reference", ""))
            nbest = {str(item) for item in input_block.get("nbest", [])}
            if top1 == reference:
                counts["top1_exact"] += 1
                copies = 1
            elif reference in nbest:
                counts["recoverable_error"] += 1
                copies = max(1, int(args.recoverable_copies))
            else:
                counts["outside_nbest_error"] += 1
                copies = 1
            rows.extend(row for _ in range(copies))

    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "source_rows": sum(counts.values()),
        "output_rows": len(rows),
        "recoverable_copies": args.recoverable_copies,
        "seed": args.seed,
        "counts": counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
