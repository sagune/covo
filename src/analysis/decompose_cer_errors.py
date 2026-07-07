#!/usr/bin/env python
"""Decompose character errors into hotword and non-hotword regions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.text import normalize_chinese_text  # type: ignore  # noqa: E402


DEFAULT_FILLERS = ("呃", "呢", "啊", "嗯")


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def norm(value: Any) -> str:
    return normalize_chinese_text(str(value or ""))


def keyword_mentions(input_block: Dict[str, Any]) -> List[str]:
    terms = []
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            term = norm(item.get("mention", ""))
            if term:
                terms.append(term)
    return sorted(set(terms), key=lambda term: (-len(term), term))


def mark_occurrences(text: str, terms: Iterable[str]) -> List[bool]:
    marks = [False] * len(text)
    for term in terms:
        if not term:
            continue
        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            for pos in range(idx, idx + len(term)):
                if 0 <= pos < len(marks):
                    marks[pos] = True
            start = idx + 1
    return marks


def edit_ops(ref: str, pred: str) -> List[Tuple[str, int | None, int | None, str, str]]:
    """Return Levenshtein edit ops as (op, ref_idx, pred_idx, ref_char, pred_char)."""
    n, m = len(ref), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + int(ref[i - 1] != pred[j - 1])
            delete = dp[i - 1][j] + 1
            insert = dp[i][j - 1] + 1
            dp[i][j] = min(sub, delete, insert)

    ops: List[Tuple[str, int | None, int | None, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + int(ref[i - 1] != pred[j - 1]):
            if ref[i - 1] != pred[j - 1]:
                ops.append(("sub", i - 1, j - 1, ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", i - 1, None, ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", None, j - 1, "", pred[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def category_for_op(
    op: Tuple[str, int | None, int | None, str, str],
    ref_hot: List[bool],
    pred_hot: List[bool],
    ref_filler: List[bool],
    pred_filler: List[bool],
) -> str:
    _, ref_idx, pred_idx, _, _ = op
    hot = False
    filler = False
    if ref_idx is not None:
        hot = hot or (0 <= ref_idx < len(ref_hot) and ref_hot[ref_idx])
        filler = filler or (0 <= ref_idx < len(ref_filler) and ref_filler[ref_idx])
    if pred_idx is not None:
        hot = hot or (0 <= pred_idx < len(pred_hot) and pred_hot[pred_idx])
        filler = filler or (0 <= pred_idx < len(pred_filler) and pred_filler[pred_idx])
    if hot:
        return "true_hotword_region"
    if filler:
        return "filler_region"
    return "non_hotword_region"


def analyze_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    ref = norm(nested_get(record, args.reference_field, ""))
    pred = norm(nested_get(record, args.prediction_field, ""))
    base = norm(nested_get(record, args.baseline_field, ""))
    if not base:
        base = norm((record.get("input", {}) or {}).get("asr_top1", ""))
    terms = keyword_mentions(record.get("input", {}) or {})
    ref_hot = mark_occurrences(ref, terms)
    pred_hot = mark_occurrences(pred, terms)
    ref_filler = mark_occurrences(ref, args.fillers)
    pred_filler = mark_occurrences(pred, args.fillers)
    ops = edit_ops(ref, pred)
    cat_counts = Counter()
    op_counts = Counter()
    examples = []
    for op in ops:
        cat = category_for_op(op, ref_hot, pred_hot, ref_filler, pred_filler)
        cat_counts[cat] += 1
        op_counts[op[0]] += 1
        if len(examples) < args.max_ops_per_case:
            examples.append({"op": op[0], "cat": cat, "ref": op[3], "pred": op[4], "ref_idx": op[1], "pred_idx": op[2]})
    return {
        "id": record.get("id", ""),
        "ref": nested_get(record, args.reference_field, ""),
        "base": nested_get(record, args.baseline_field, ""),
        "pred": nested_get(record, args.prediction_field, ""),
        "ref_norm": ref,
        "pred_norm": pred,
        "ref_chars": len(ref),
        "edits": len(ops),
        "terms": " ".join(terms),
        "hotword_edits": cat_counts["true_hotword_region"],
        "filler_edits": cat_counts["filler_region"],
        "non_hotword_edits": cat_counts["non_hotword_region"],
        "sub": op_counts["sub"],
        "ins": op_counts["ins"],
        "del": op_counts["del"],
        "examples": json.dumps(examples, ensure_ascii=False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    parser.add_argument("--fillers", default=",".join(DEFAULT_FILLERS))
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--max-ops-per-case", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.fillers = [norm(item) for item in args.fillers.replace("，", ",").split(",") if norm(item)]
    rows = [analyze_record(record, args) for record in read_jsonl(args.input)]
    total_chars = sum(row["ref_chars"] for row in rows)
    total_edits = sum(row["edits"] for row in rows)
    hotword_edits = sum(row["hotword_edits"] for row in rows)
    filler_edits = sum(row["filler_edits"] for row in rows)
    non_hotword_edits = sum(row["non_hotword_edits"] for row in rows)
    op_counts = Counter()
    for row in rows:
        op_counts["sub"] += row["sub"]
        op_counts["ins"] += row["ins"]
        op_counts["del"] += row["del"]
    summary = {
        "input": args.input,
        "samples": len(rows),
        "reference_chars": total_chars,
        "edits": total_edits,
        "cer": total_edits / total_chars if total_chars else 0.0,
        "hotword_edits": hotword_edits,
        "hotword_edit_ratio": hotword_edits / total_edits if total_edits else 0.0,
        "filler_edits": filler_edits,
        "filler_edit_ratio": filler_edits / total_edits if total_edits else 0.0,
        "non_hotword_edits": non_hotword_edits,
        "non_hotword_edit_ratio": non_hotword_edits / total_edits if total_edits else 0.0,
        "op_counts": dict(op_counts),
        "rows_with_edits": sum(1 for row in rows if row["edits"] > 0),
        "rows_with_hotword_edits": sum(1 for row in rows if row["hotword_edits"] > 0),
        "rows_with_non_hotword_edits": sum(1 for row in rows if row["non_hotword_edits"] > 0),
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    case_rows = sorted([row for row in rows if row["edits"] > 0], key=lambda row: (row["non_hotword_edits"], row["edits"]), reverse=True)
    case_rows = case_rows[: int(args.max_cases)]
    with Path(args.cases_output).open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "id",
            "edits",
            "hotword_edits",
            "filler_edits",
            "non_hotword_edits",
            "sub",
            "ins",
            "del",
            "terms",
            "ref",
            "base",
            "pred",
            "examples",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in case_rows])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
