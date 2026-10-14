#!/usr/bin/env python
"""Expand edit spans so generated edits are uniquely locatable in ASR text."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-span-chars", type=int, default=12)
    return parser.parse_args()


def _single_edit_span(asr: str, ref: str) -> Tuple[int, int, int, int] | None:
    opcodes = [
        item
        for item in difflib.SequenceMatcher(a=asr, b=ref, autojunk=False).get_opcodes()
        if item[0] != "equal"
    ]
    if len(opcodes) != 1:
        return None
    _, i1, i2, j1, j2 = opcodes[0]
    return i1, i2, j1, j2


def _count_occurrences(text: str, span: str) -> int:
    if not span:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(span, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def _anchor_span(asr: str, ref: str, max_span_chars: int) -> Tuple[str, str, bool]:
    span = _single_edit_span(asr, ref)
    if span is None:
        return "", "", False
    i1, i2, j1, j2 = span
    left = 0
    right = 0
    best_from = asr[i1:i2]
    best_to = ref[j1:j2]
    max_left = min(i1, j1)
    max_right = min(len(asr) - i2, len(ref) - j2)

    while True:
        from_text = asr[i1 - left : i2 + right]
        to_text = ref[j1 - left : j2 + right]
        if from_text and from_text != to_text and _count_occurrences(asr, from_text) == 1:
            return from_text, to_text, from_text != best_from or to_text != best_to
        can_left = left < max_left
        can_right = right < max_right
        if not can_left and not can_right:
            break
        if len(from_text) >= int(max_span_chars) and len(to_text) >= int(max_span_chars):
            break
        if can_left and (left <= right or not can_right):
            left += 1
        elif can_right:
            right += 1

    from_text = asr[i1:i2]
    to_text = ref[j1:j2]
    return from_text, to_text, False


def _replace_primary_edit(record: Dict[str, Any], from_text: str, to_text: str) -> Dict[str, Any]:
    record = deepcopy(record)
    output = record.get("output", {})
    edits = list(output.get("edits", []) or []) if isinstance(output, dict) else []
    if not edits:
        return record
    edits[0] = {
        **edits[0],
        "from": from_text,
        "to": to_text,
        "reason": "anchored_span_replace",
    }
    output["edits"] = edits
    record["output"] = output

    evidence = record.get("input", {}).get("phonetic_evidence")
    if isinstance(evidence, dict):
        evidence["from_text"] = from_text
        evidence["to_text"] = to_text
        evidence["anchored_span"] = True
    return record


def main() -> int:
    args = parse_args()
    stats = {
        "records": 0,
        "edit_records": 0,
        "anchored": 0,
        "single_to_span": 0,
        "skipped_multi_diff": 0,
        "non_unique_after": 0,
    }

    def records():
        for record in read_jsonl(args.input):
            stats["records"] += 1
            output = record.get("output", {})
            edits = list(output.get("edits", []) or []) if isinstance(output, dict) else []
            if not edits:
                yield record
                continue
            stats["edit_records"] += 1
            asr = normalize_chinese_text(record.get("input", {}).get("asr_top1", ""))
            ref = normalize_chinese_text(record.get("reference", ""))
            if not asr or not ref:
                yield record
                continue
            from_text, to_text, changed = _anchor_span(asr, ref, int(args.max_span_chars))
            if not from_text:
                stats["skipped_multi_diff"] += 1
                yield record
                continue
            if _count_occurrences(asr, from_text) != 1:
                stats["non_unique_after"] += 1
                yield record
                continue
            old_from = str(edits[0].get("from", ""))
            old_to = str(edits[0].get("to", ""))
            if changed or old_from != from_text or old_to != to_text:
                stats["anchored"] += 1
                if len(old_from) == len(old_to) == 1 and (len(from_text) > 1 or len(to_text) > 1):
                    stats["single_to_span"] += 1
                yield _replace_primary_edit(record, from_text, to_text)
            else:
                yield record

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
