#!/usr/bin/env python
"""Split JSONL records by field or random ratios."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["field", "random"], default="field")
    parser.add_argument("--field", default="split")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--dev-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _nested_get(record: Dict[str, Any], dotted: str, default: str = "") -> str:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return str(value)


def _random_split(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(int(args.seed))
    shuffled = list(records)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_n = int(total * float(args.train_ratio))
    dev_n = int(total * float(args.dev_ratio))
    return {
        "train": shuffled[:train_n],
        "dev": shuffled[train_n : train_n + dev_n],
        "test": shuffled[train_n + dev_n :],
    }


def _field_split(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        key = _nested_get(record, args.field, "unknown") or "unknown"
        buckets.setdefault(key, []).append(record)
    return buckets


def main() -> int:
    args = parse_args()
    records = []
    for record in read_jsonl(args.input):
        records.append(record)
        if args.limit and len(records) >= args.limit:
            break

    buckets = _field_split(records, args) if args.mode == "field" else _random_split(records, args)
    out_dir = Path(args.output_dir)
    prefix = f"{args.prefix}_" if args.prefix else ""
    counts = {}
    for name, bucket in sorted(buckets.items()):
        path = out_dir / f"{prefix}{name}.jsonl"
        counts[name] = write_jsonl(path, bucket)

    print(json.dumps({"input": args.input, "output_dir": str(out_dir), "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
