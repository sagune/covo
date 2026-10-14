#!/usr/bin/env python
"""Route contextual AISHELL rows to CB-SenseVoice and all others to SenseVoice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from analysis.cbsensevoice_covo_bridge import normalize_text
except ModuleNotFoundError:
    from cbsensevoice_covo_bridge import normalize_text


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def nested_get(record: Dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part, "")
    return value


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_idx, left_char in enumerate(left, 1):
        current = [left_idx]
        for right_idx, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_idx] + 1,
                    previous[right_idx - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def load_context_rows(
    path: Path,
    uttid_path: Path,
    prediction_field: str,
) -> Dict[str, Dict[str, str]]:
    records = list(read_jsonl(path))
    utterance_ids = [
        line.strip().split()[0]
        for line in uttid_path.open("r", encoding="utf-8-sig")
        if line.strip()
    ]
    if len(records) != len(utterance_ids):
        raise ValueError(
            f"context rows and utterance IDs differ: {len(records)} != {len(utterance_ids)}"
        )

    output: Dict[str, Dict[str, str]] = {}
    for utterance_id, record in zip(utterance_ids, records):
        output[utterance_id] = {
            "prediction": str(nested_get(record, prediction_field) or ""),
            "reference": str(record.get("reference", "") or ""),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--context-uttids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--base-prediction-field", default="prediction")
    parser.add_argument("--context-prediction-field", default="input.asr_top1")
    args = parser.parse_args()

    context = load_context_rows(
        args.context,
        args.context_uttids,
        args.context_prediction_field,
    )
    base_rows = list(read_jsonl(args.base))
    seen_ids = set()
    routed_rows = []
    edits = 0
    reference_chars = 0
    exact = 0
    context_count = 0

    for record in base_rows:
        utterance_id = str(record.get("id", "") or "")
        if not utterance_id:
            raise ValueError("base record is missing id")
        if utterance_id in seen_ids:
            raise ValueError(f"duplicate base utterance ID: {utterance_id}")
        seen_ids.add(utterance_id)

        reference = str(record.get("reference", "") or "")
        if utterance_id in context:
            context_row = context[utterance_id]
            if normalize_text(reference) != normalize_text(context_row["reference"]):
                raise ValueError(f"reference mismatch for contextual row {utterance_id}")
            prediction = context_row["prediction"]
            route = "cb_sensevoice"
            context_count += 1
        else:
            prediction = str(nested_get(record, args.base_prediction_field) or "")
            route = "sensevoice"

        reference_norm = normalize_text(reference)
        prediction_norm = normalize_text(prediction)
        distance = edit_distance(prediction_norm, reference_norm)
        edits += distance
        reference_chars += len(reference_norm)
        exact += int(distance == 0)
        routed_rows.append(
            {
                "id": utterance_id,
                "reference": reference,
                "prediction": prediction,
                "route": route,
                "edits": distance,
                "reference_chars": len(reference_norm),
            }
        )

    missing_context = sorted(set(context) - seen_ids)
    if missing_context:
        raise ValueError(f"{len(missing_context)} contextual IDs are absent from base AISHELL")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in routed_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "base": str(args.base),
        "context": str(args.context),
        "output": str(args.output),
        "base_prediction_field": args.base_prediction_field,
        "context_prediction_field": args.context_prediction_field,
        "rows": len(routed_rows),
        "context_rows": context_count,
        "base_rows": len(routed_rows) - context_count,
        "normalization": "OpenCC t2s + NFKC + punctuation removal",
        "cer": edits / reference_chars if reference_chars else 0.0,
        "edits": edits,
        "reference_chars": reference_chars,
        "exact": exact,
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
