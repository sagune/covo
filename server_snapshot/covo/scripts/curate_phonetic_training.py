#!/usr/bin/env python
"""Build a conservative phonetic SFT subset for strong zero-shot models."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--keep-ratio", type=float, default=2.0)
    parser.add_argument("--min-to-support", type=int, default=2)
    return parser.parse_args()


def _phonetic_evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("input", {}).get("phonetic_evidence", {})
    return value if isinstance(value, dict) else {}


def _is_keep(record: Dict[str, Any]) -> bool:
    return record.get("output", {}).get("decision") == "keep"


def _is_conservative_edit(record: Dict[str, Any], min_to_support: int) -> bool:
    output = record.get("output", {})
    if output.get("decision") != "edit" or not output.get("edits"):
        return False
    evidence = _phonetic_evidence(record)
    edit_type = str(output.get("edit_type", ""))
    pinyin_distance = int(evidence.get("pinyin_distance") or 999)
    to_support = int(evidence.get("to_support_count") or 0)
    to_in_nbest = bool(evidence.get("to_in_nbest"))
    if edit_type in {"deletion", "insertion_or_expansion"}:
        return False
    return to_in_nbest and pinyin_distance <= 1 and to_support >= int(min_to_support)


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    edits: List[Dict[str, Any]] = []
    keep_traps: List[Dict[str, Any]] = []
    keep_plain: List[Dict[str, Any]] = []
    total = 0
    for record in read_jsonl(args.input):
        total += 1
        if _is_conservative_edit(record, args.min_to_support):
            edits.append(record)
        elif _is_keep(record):
            evidence = _phonetic_evidence(record)
            if int(evidence.get("same_pinyin_trap_count") or 0) > 0:
                keep_traps.append(record)
            else:
                keep_plain.append(record)

    rng.shuffle(edits)
    rng.shuffle(keep_traps)
    rng.shuffle(keep_plain)
    keep_target = int(round(len(edits) * float(args.keep_ratio)))
    trap_target = min(len(keep_traps), int(round(keep_target * 0.75)))
    plain_target = max(0, keep_target - trap_target)
    selected = edits + keep_traps[:trap_target] + keep_plain[:plain_target]
    rng.shuffle(selected)

    written = write_jsonl(args.output, selected)
    summary = {
        "input": args.input,
        "output": args.output,
        "input_rows": total,
        "written": written,
        "conservative_edits": len(edits),
        "selected_keep_traps": trap_target,
        "selected_keep_plain": min(len(keep_plain), plain_target),
        "keep_ratio": float(args.keep_ratio),
        "min_to_support": int(args.min_to_support),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
