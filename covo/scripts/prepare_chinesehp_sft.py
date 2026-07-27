#!/usr/bin/env python
"""Convert ChineseHP AISHELL-1 JSONL into constrained-correction SFT JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.chinesehp import iter_jsonl, make_sft_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to chinesehp_aishell1.jsonl")
    parser.add_argument("--output", required=True, help="Output SFT JSONL path")
    parser.add_argument("--nbest-size", type=int, default=5)
    parser.add_argument("--output-mode", choices=["edits", "text"], default="edits")
    parser.add_argument("--split", default="", help="Optional split filter: train/dev/test")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of records")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as writer:
        for record in iter_jsonl(args.input):
            if args.split and str(record.get("split", "")) != args.split:
                continue
            sft_record = make_sft_record(
                record,
                nbest_size=args.nbest_size,
                output_mode=args.output_mode,
            )
            writer.write(json.dumps(sft_record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(
        json.dumps(
            {
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(args.output),
                "written": written,
                "nbest_size": args.nbest_size,
                "output_mode": args.output_mode,
                "split": args.split or "all",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
