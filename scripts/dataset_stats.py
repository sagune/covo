#!/usr/bin/env python
"""Print dataset statistics for internal SFT JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import parse_edits
from covo.io import read_jsonl
from covo.metrics import cer
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def _mean(values):
    return float(statistics.mean(values)) if values else 0.0


def _percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * float(p)))))
    return float(values[idx])


def main() -> int:
    args = parse_args()
    n = 0
    split_counts = Counter()
    nbest_sizes = []
    ref_lengths = []
    asr_lengths = []
    baseline_cers = []
    edit_counts = []
    changed_chars = []
    empty_edits = 0

    for record in read_jsonl(args.input):
        n += 1
        split_counts[str(record.get("split", "")) or "unknown"] += 1
        input_block = record.get("input", {}) or {}
        asr_top1 = str(input_block.get("asr_top1", ""))
        reference = str(record.get("reference", ""))
        nbest = list(input_block.get("nbest", []) or [])
        nbest_sizes.append(len(nbest))
        ref_lengths.append(len(normalize_chinese_text(reference)))
        asr_lengths.append(len(normalize_chinese_text(asr_top1)))
        baseline_cers.append(cer(asr_top1, reference))
        try:
            edits = parse_edits(record.get("output", {}))
        except ValueError:
            edits = []
        edit_counts.append(len(edits))
        empty_edits += int(len(edits) == 0)
        changed_chars.append(
            sum(
                max(len(normalize_chinese_text(edit.from_text)), len(normalize_chinese_text(edit.to_text)))
                for edit in edits
            )
        )

    stats = {
        "samples": n,
        "splits": dict(split_counts),
        "empty_edits": empty_edits,
        "empty_edit_rate": float(empty_edits / n) if n else 0.0,
        "avg_nbest_size": _mean(nbest_sizes),
        "avg_reference_chars": _mean(ref_lengths),
        "avg_asr_chars": _mean(asr_lengths),
        "avg_baseline_cer": _mean(baseline_cers),
        "p50_baseline_cer": _percentile(baseline_cers, 0.5),
        "p90_baseline_cer": _percentile(baseline_cers, 0.9),
        "avg_edit_count": _mean(edit_counts),
        "avg_changed_chars": _mean(changed_chars),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
