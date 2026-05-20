#!/usr/bin/env python
"""Apply predicted or gold JSON edits with the conservative verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import safe_apply_edits
from covo.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--edits-field",
        default="predicted_edits",
        help="Field containing edits. Falls back to output.edits when missing.",
    )
    parser.add_argument("--max-total-changed-chars", type=int, default=12)
    return parser.parse_args()


def _get_input_block(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("input", {})
    return value if isinstance(value, dict) else {}


def _get_edits_value(record: Dict[str, Any], field: str) -> Any:
    if field in record:
        return record[field]
    output = record.get("output", {})
    if isinstance(output, dict) and "edits" in output:
        return {"edits": output.get("edits", [])}
    return {"edits": []}


def convert_record(record: Dict[str, Any], edits_field: str, max_total_changed_chars: int) -> Dict[str, Any]:
    evidence = _get_input_block(record)
    asr_top1 = str(evidence.get("asr_top1", record.get("asr_top1", "")))
    edits_value = _get_edits_value(record, edits_field)
    result = safe_apply_edits(
        asr_top1,
        edits_value,
        evidence,
        max_total_changed_chars=max_total_changed_chars,
    )
    return {
        **record,
        "correction_result": result,
        "prediction": result["text"],
    }


def main() -> int:
    args = parse_args()

    def records():
        for record in read_jsonl(args.input):
            yield convert_record(record, args.edits_field, args.max_total_changed_chars)

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
