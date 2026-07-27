#!/usr/bin/env python
"""Filter constrained-correction SFT records by edit size and baseline CER."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import parse_edits, validate_edits
from covo.io import read_jsonl, write_jsonl
from covo.metrics import cer
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-baseline-cer", type=float, default=0.8)
    parser.add_argument("--max-total-changed-chars", type=int, default=24)
    parser.add_argument("--max-edits", type=int, default=4)
    parser.add_argument("--drop-empty-edits", action="store_true")
    parser.add_argument("--require-verifier-pass", action="store_true")
    return parser.parse_args()


def _total_changed_chars(edits) -> int:
    return int(sum(max(len(normalize_chinese_text(e.from_text)), len(normalize_chinese_text(e.to_text))) for e in edits))


def _keep(record, args) -> tuple[bool, str]:
    input_block = record.get("input", {}) or {}
    output = record.get("output", {}) or {}
    asr_top1 = str(input_block.get("asr_top1", ""))
    reference = str(record.get("reference", output.get("corrected_text", "")))
    if not normalize_chinese_text(asr_top1) or not normalize_chinese_text(reference):
        return False, "empty_text"
    if cer(asr_top1, reference) > float(args.max_baseline_cer):
        return False, "baseline_cer_too_high"
    try:
        edits = parse_edits(output)
    except ValueError:
        return False, "bad_edits"
    if args.drop_empty_edits and not edits:
        return False, "empty_edits"
    if len(edits) > int(args.max_edits):
        return False, "too_many_edits"
    if _total_changed_chars(edits) > int(args.max_total_changed_chars):
        return False, "too_many_changed_chars"
    if args.require_verifier_pass:
        ok, reasons = validate_edits(
            edits,
            input_block,
            max_total_changed_chars=int(args.max_total_changed_chars),
        )
        if not ok:
            return False, "verifier:" + "|".join(reasons[:3])
    return True, "kept"


def main() -> int:
    args = parse_args()
    stats = {}

    def records():
        for record in read_jsonl(args.input):
            keep, reason = _keep(record, args)
            stats[reason] = stats.get(reason, 0) + 1
            if keep:
                yield record

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
