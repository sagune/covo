#!/usr/bin/env python3
"""Rank evaluator JSON summaries and lock the best checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, help="Directory containing checkpoint summaries")
    parser.add_argument("--glob", default="checkpoint-*.summary.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    return parser.parse_args()


def step_from_name(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    if not match:
        raise ValueError(f"cannot parse checkpoint step from {path}")
    return int(match.group(1))


def main() -> int:
    args = parse_args()
    rows = []
    for path in args.summaries.glob(args.glob):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        step = step_from_name(path)
        rows.append(
            {
                "checkpoint": f"checkpoint-{step}",
                "step": step,
                "cer": float(metrics["cer"]),
                "worsened_samples": int(metrics["worsened_samples"]),
                "summary": str(path.resolve()),
            }
        )
    if not rows:
        raise SystemExit("no checkpoint summaries found")
    rows.sort(key=lambda row: (row["cer"], row["worsened_samples"], row["step"]))
    args.ranking.parent.mkdir(parents=True, exist_ok=True)
    args.ranking.write_text(
        "checkpoint\tcer\tworsened_samples\tstep\tsummary\n"
        + "".join(
            f"{row['checkpoint']}\t{row['cer']:.12g}\t{row['worsened_samples']}\t"
            f"{row['step']}\t{row['summary']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    selected = {
        "selection_rule": ["lowest_corpus_cer", "fewest_worsened_samples", "earlier_step"],
        "selected": rows[0],
        "top_checkpoints": [row["checkpoint"] for row in rows[: args.top_k]],
        "all_checkpoints": rows,
    }
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    args.selection.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(rows[0]["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
