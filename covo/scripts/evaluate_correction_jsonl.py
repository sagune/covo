#!/usr/bin/env python
"""Evaluate character error rate for ASR correction JSONL files."""

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

from covo.io import read_jsonl
from covo.metrics import cer, edit_distance
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    return parser.parse_args()


def _nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def main() -> int:
    args = parse_args()
    n = 0
    pred_edits = 0
    pred_ref_chars = 0
    base_edits = 0
    base_ref_chars = 0
    improved = 0
    worsened = 0

    for record in read_jsonl(args.input):
        pred = str(_nested_get(record, args.prediction_field, ""))
        ref = str(_nested_get(record, args.reference_field, ""))
        base = str(_nested_get(record, args.baseline_field, ""))
        pred_norm = normalize_chinese_text(pred)
        ref_norm = normalize_chinese_text(ref)
        base_norm = normalize_chinese_text(base)
        if not ref_norm:
            continue
        n += 1
        pred_d = edit_distance(list(pred_norm), list(ref_norm))
        base_d = edit_distance(list(base_norm), list(ref_norm))
        pred_edits += pred_d
        base_edits += base_d
        pred_ref_chars += len(ref_norm)
        base_ref_chars += len(ref_norm)
        improved += int(pred_d < base_d)
        worsened += int(pred_d > base_d)

    metrics = {
        "samples": n,
        "cer": float(pred_edits / pred_ref_chars) if pred_ref_chars else 0.0,
        "baseline_cer": float(base_edits / base_ref_chars) if base_ref_chars else 0.0,
        "improved_samples": improved,
        "worsened_samples": worsened,
        "unchanged_samples": n - improved - worsened,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
