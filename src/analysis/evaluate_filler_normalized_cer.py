#!/usr/bin/env python
"""Evaluate CER after removing optional Chinese lecture filler tokens."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.metrics import edit_distance  # type: ignore  # noqa: E402

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


DEFAULT_FILLERS = (
    "这一个",
    "那么",
    "的话",
    "这个",
    "那个",
    "咱们",
    "我们呢",
    "就是",
    "呢",
    "啊",
    "呃",
    "嗯",
)


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def parse_fillers(spec: str) -> List[str]:
    if not spec.strip():
        return []
    fillers = [item.strip() for item in re.split(r"[,，\\s]+", spec) if item.strip()]
    return sorted(set(fillers), key=lambda item: (-len(item), item))


def normalize_text(value: Any, fillers: List[str]) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    chars = []
    for ch in text:
        if unicodedata.category(ch).startswith("P") or ch.isspace():
            continue
        chars.append(ch)
    text = "".join(chars)
    for filler in fillers:
        text = text.replace(filler, "")
    return text


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    fillers = parse_fillers(args.fillers)
    samples = 0
    chars = 0
    base_edits = 0
    pred_edits = 0
    improved = 0
    worsened = 0
    unchanged = 0
    base_exact = 0
    pred_exact = 0
    base_changed_by_filter = 0
    pred_changed_by_filter = 0
    ref_changed_by_filter = 0

    for record in read_jsonl(args.input):
        ref_raw = nested_get(record, args.reference_field, "")
        base_raw = nested_get(record, args.baseline_field, "")
        pred_raw = nested_get(record, args.prediction_field, "")
        ref_plain = normalize_text(ref_raw, [])
        base_plain = normalize_text(base_raw, [])
        pred_plain = normalize_text(pred_raw, [])
        ref = normalize_text(ref_raw, fillers)
        base = normalize_text(base_raw, fillers)
        pred = normalize_text(pred_raw, fillers)
        if not ref:
            continue
        samples += 1
        chars += len(ref)
        base_d = int(edit_distance(list(base), list(ref)))
        pred_d = int(edit_distance(list(pred), list(ref)))
        base_edits += base_d
        pred_edits += pred_d
        improved += int(pred_d < base_d)
        worsened += int(pred_d > base_d)
        unchanged += int(pred_d == base_d)
        base_exact += int(base_d == 0)
        pred_exact += int(pred_d == 0)
        ref_changed_by_filter += int(ref_plain != ref)
        base_changed_by_filter += int(base_plain != base)
        pred_changed_by_filter += int(pred_plain != pred)

    return {
        "input": args.input,
        "samples": samples,
        "reference_chars": chars,
        "fillers": fillers,
        "base_cer": float(base_edits / chars) if chars else 0.0,
        "prediction_cer": float(pred_edits / chars) if chars else 0.0,
        "base_edits": base_edits,
        "prediction_edits": pred_edits,
        "improved_samples": improved,
        "worsened_samples": worsened,
        "unchanged_samples": unchanged,
        "base_exact": base_exact,
        "prediction_exact": pred_exact,
        "ref_changed_by_filter": ref_changed_by_filter,
        "base_changed_by_filter": base_changed_by_filter,
        "pred_changed_by_filter": pred_changed_by_filter,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    parser.add_argument("--fillers", default=",".join(DEFAULT_FILLERS))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
