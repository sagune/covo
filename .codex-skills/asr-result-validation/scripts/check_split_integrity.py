#!/usr/bin/env python3
"""Audit JSONL train/dev/test boundaries before ASR validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def nested_get(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def audit(path: Path, expected_split: str, args: argparse.Namespace) -> dict[str, Any]:
    ids: set[str] = set()
    duplicate_ids = 0
    missing_ids = 0
    missing_references = 0
    missing_baselines = 0
    split_mismatches = 0
    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

            sample_id = nested_get(record, args.id_field)
            if sample_id is None or str(sample_id) == "":
                missing_ids += 1
            elif str(sample_id) in ids:
                duplicate_ids += 1
            else:
                ids.add(str(sample_id))

            actual_split = nested_get(record, args.split_field)
            if actual_split is not None and str(actual_split).lower() != expected_split.lower():
                split_mismatches += 1
            if nested_get(record, args.reference_field) in (None, ""):
                missing_references += 1
            if args.baseline_field and nested_get(record, args.baseline_field) in (None, ""):
                missing_baselines += 1

    return {
        "path": str(path.resolve()),
        "expected_split": expected_split,
        "rows": rows,
        "unique_ids": len(ids),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "split_mismatches": split_mismatches,
        "missing_references": missing_references,
        "missing_baselines": missing_baselines,
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "_ids": ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = {
        name: audit(path, name, args)
        for name, path in (("train", args.train), ("dev", args.dev), ("test", args.test))
    }
    overlaps = {
        "train_dev": len(reports["train"]["_ids"] & reports["dev"]["_ids"]),
        "train_test": len(reports["train"]["_ids"] & reports["test"]["_ids"]),
        "dev_test": len(reports["dev"]["_ids"] & reports["test"]["_ids"]),
    }
    for report in reports.values():
        report.pop("_ids")

    hard_failures = sum(overlaps.values())
    for report in reports.values():
        hard_failures += (
            report["duplicate_ids"]
            + report["missing_ids"]
            + report["split_mismatches"]
            + report["missing_references"]
            + report["missing_baselines"]
        )
    payload = {"splits": reports, "id_overlaps": overlaps, "passed": hard_failures == 0}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
