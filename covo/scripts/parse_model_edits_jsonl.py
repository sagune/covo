#!/usr/bin/env python
"""Parse noisy model output text into normalized JSON edits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.model_output import parse_model_edits_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="model_output")
    parser.add_argument("--edits-field", default="predicted_edits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def records():
        for record in read_jsonl(args.input):
            parsed = parse_model_edits_json(str(record.get(args.text_field, "")))
            yield {
                **record,
                args.edits_field: {"edits": parsed["edits"]},
                "parse_warnings": parsed["parse_warnings"],
            }

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
