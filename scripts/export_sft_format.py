#!/usr/bin/env python
"""Export internal SFT records to Qwen messages or prompt/completion JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.formats import convert_record_format
from covo.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format",
        choices=["internal", "qwen-messages", "prompt-completion", "covoger"],
        default="qwen-messages",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def records():
        count = 0
        for record in read_jsonl(args.input):
            yield convert_record_format(record, args.format)
            count += 1
            if args.limit and count >= args.limit:
                break

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "format": args.format, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
