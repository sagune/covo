#!/usr/bin/env python
"""Guard final-text ASR correction predictions before CER evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

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
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    parser.add_argument("--max-distance", type=int, default=12)
    parser.add_argument("--max-distance-ratio", type=float, default=0.45)
    parser.add_argument("--min-output-length-ratio", type=float, default=0.55)
    return parser.parse_args()


def _nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _accept_prediction(prediction: str, baseline: str, args: argparse.Namespace) -> tuple[bool, str, int]:
    pred_norm = normalize_chinese_text(prediction)
    base_norm = normalize_chinese_text(baseline)
    if not pred_norm:
        return False, "empty_prediction", 0
    if not base_norm:
        return True, "accepted_empty_baseline", 0
    distance = edit_distance(list(pred_norm), list(base_norm))
    max_allowed = min(
        int(args.max_distance),
        max(1, int(round(len(base_norm) * float(args.max_distance_ratio)))),
    )
    if len(pred_norm) < int(round(len(base_norm) * float(args.min_output_length_ratio))):
        return False, "too_short", distance
    if distance > max_allowed:
        return False, "too_far_from_asr", distance
    return True, "accepted", distance


def _records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    accepted = 0
    rejected = 0
    for record in read_jsonl(args.input):
        raw_prediction = str(_nested_get(record, args.prediction_field, ""))
        baseline = str(_nested_get(record, args.baseline_field, ""))
        ok, reason, distance = _accept_prediction(raw_prediction, baseline, args)
        prediction = raw_prediction if ok else baseline
        accepted += int(ok)
        rejected += int(not ok)
        yield {
            **record,
            "raw_prediction": raw_prediction,
            "prediction": prediction,
            "guard": {
                "accepted": ok,
                "reason": reason,
                "distance_from_asr": distance,
                "max_distance": int(args.max_distance),
                "max_distance_ratio": float(args.max_distance_ratio),
                "min_output_length_ratio": float(args.min_output_length_ratio),
            },
        }
    print(json.dumps({"accepted": accepted, "rejected": rejected}, ensure_ascii=False), file=sys.stderr)


def main() -> int:
    args = parse_args()
    written = write_jsonl(args.output, _records(args))
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
