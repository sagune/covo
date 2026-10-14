#!/usr/bin/env python
"""Evaluate no-correction and oracle N-best baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.baselines import evaluate_baselines, oracle_nbest
from covo.io import read_jsonl, write_jsonl


def _top1(record):
    evidence = record.get("input", {})
    if isinstance(evidence, dict) and evidence.get("asr_top1"):
        return str(evidence["asr_top1"])
    nbest = evidence.get("nbest", []) if isinstance(evidence, dict) else record.get("nbest", [])
    return str(nbest[0]) if nbest else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--oracle-top-k", type=int, default=0, help="0 means use all N-best candidates.")
    parser.add_argument("--output", default="", help="Optional per-record prediction JSONL.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = list(read_jsonl(args.input))
    metrics = evaluate_baselines(records, oracle_top_k=args.oracle_top_k)

    if args.output:
        def output_records():
            for record in records:
                oracle_text, oracle_index, oracle_distance = oracle_nbest(record, top_k=args.oracle_top_k)
                yield {
                    **record,
                    "baseline_predictions": {
                        "no_correction": {"prediction": _top1(record)},
                        "oracle_nbest": {
                            "prediction": oracle_text,
                            "index": oracle_index,
                            "distance": oracle_distance,
                        },
                    },
                }

        written = write_jsonl(args.output, output_records())
        metrics["output"] = args.output
        metrics["written"] = written
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
