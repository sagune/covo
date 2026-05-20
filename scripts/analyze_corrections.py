#!/usr/bin/env python
"""Analyze correction outputs and optionally export improved/worsened examples."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.metrics import edit_distance
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--topk", type=int, default=20)
    return parser.parse_args()


def _distance(text: str, reference: str) -> int:
    return edit_distance(list(normalize_chinese_text(text)), list(normalize_chinese_text(reference)))


def main() -> int:
    args = parse_args()
    rows = []
    reason_counts = Counter()
    status_counts = Counter()

    for record in read_jsonl(args.input):
        input_block = record.get("input", {}) or {}
        reference = str(record.get("reference", ""))
        baseline = str(input_block.get("asr_top1", record.get("asr_top1", "")))
        prediction = str(record.get("prediction", record.get("correction_result", {}).get("text", "")))
        base_d = _distance(baseline, reference)
        pred_d = _distance(prediction, reference)
        if pred_d < base_d:
            status = "improved"
        elif pred_d > base_d:
            status = "worsened"
        else:
            status = "unchanged"
        status_counts[status] += 1
        for reason in list((record.get("correction_result", {}) or {}).get("reasons", []) or []):
            reason_counts[str(reason)] += 1
        rows.append(
            {
                "id": record.get("id", ""),
                "status": status,
                "delta_edits": pred_d - base_d,
                "baseline_edits": base_d,
                "prediction_edits": pred_d,
                "baseline": baseline,
                "prediction": prediction,
                "reference": reference,
                "correction_result": record.get("correction_result", {}),
            }
        )

    summary = {
        "samples": len(rows),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts.most_common()),
    }

    if args.output_dir:
        out_dir = Path(args.output_dir)
        for status in ("improved", "worsened", "unchanged"):
            subset = [row for row in rows if row["status"] == status]
            subset.sort(key=lambda row: row["delta_edits"])
            if status == "worsened":
                subset.sort(key=lambda row: row["delta_edits"], reverse=True)
            write_jsonl(out_dir / f"{status}.jsonl", subset[: int(args.topk)])
        write_jsonl(out_dir / "summary.jsonl", [summary])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
