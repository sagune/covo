#!/usr/bin/env python
"""Apply a conservative rule-based N-best/pinyin correction baseline."""

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
from covo.rule_corrector import apply_rule_correction, make_rule_edits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-char-distance", type=int, default=4)
    parser.add_argument("--max-pinyin-distance", type=int, default=1)
    parser.add_argument("--min-candidate-support", type=int, default=2)
    parser.add_argument("--max-total-changed-chars", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def records():
        for record in read_jsonl(args.input):
            evidence = record.get("input", {}) or record
            edits = make_rule_edits(
                evidence,
                max_char_distance=args.max_char_distance,
                max_pinyin_distance=args.max_pinyin_distance,
                min_candidate_support=args.min_candidate_support,
            )
            result = apply_rule_correction(
                evidence,
                max_char_distance=args.max_char_distance,
                max_pinyin_distance=args.max_pinyin_distance,
                min_candidate_support=args.min_candidate_support,
                max_total_changed_chars=args.max_total_changed_chars,
            )
            yield {
                **record,
                "predicted_edits": {"edits": [edit.to_json() for edit in edits]},
                "correction_result": result,
                "prediction": result["text"],
            }

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
