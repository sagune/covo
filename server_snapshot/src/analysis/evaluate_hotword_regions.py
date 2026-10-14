#!/usr/bin/env python
"""Evaluate corpus CER inside and outside designated hotword spans."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
COVO_SRC = ROOT / "covo" / "src"
if str(COVO_SRC) not in sys.path:
    sys.path.insert(0, str(COVO_SRC))

from covo.text import normalize_chinese_text  # noqa: E402

T2S = OpenCC("t2s")


def normalize(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    return normalize_chinese_text(T2S.convert(normalized))


def read_uttids(path: Path) -> List[str]:
    return [line.split(maxsplit=1)[0] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_designated_hotwords(path: Path) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        hotword, utterance_id = normalize(fields[0]), fields[1].strip()
        if hotword:
            result.setdefault(utterance_id, []).append(hotword)
    return result


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def nested_value(row: dict, field: str) -> object:
    value: object = row
    for key in field.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(key, "")
    return value


def load_jsonl_predictions(
    path: Path,
    uttids: Sequence[str],
    prediction_field: str,
) -> Dict[str, Tuple[str, str]]:
    rows: Dict[str, Tuple[str, str]] = {}
    for row in iter_jsonl(path):
        raw_id = str(row.get("id", "")).strip()
        if raw_id.isdigit():
            idx = int(raw_id)
            utterance_id = uttids[idx]
        else:
            utterance_id = raw_id
        input_block = row.get("input", {}) or {}
        reference = row.get("reference", "")
        if prediction_field == "auto":
            prediction = (
                row.get("normalized_prediction")
                or row.get("prediction")
                or row.get("pred")
                or row.get("text")
                or input_block.get("asr_top1")
                or ""
            )
        else:
            prediction = nested_value(row, prediction_field)
        rows[utterance_id] = (normalize(reference), normalize(prediction))
    return rows


def load_csv_predictions(path: Path, uttids: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    rows: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("is_top1", "")).lower() not in {"true", "1"}:
                continue
            idx = int(row["idx"])
            rows[uttids[idx]] = (normalize(row.get("ref", "")), normalize(row.get("pred", "")))
    return rows


def load_predictions(
    path: Path,
    uttids: Sequence[str],
    prediction_field: str,
) -> Dict[str, Tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        return load_csv_predictions(path, uttids)
    return load_jsonl_predictions(path, uttids, prediction_field)


def hotword_mask(reference: str, hotwords: Sequence[str]) -> List[bool]:
    mask = [False] * len(reference)
    for hotword in hotwords:
        start = 0
        while hotword and (idx := reference.find(hotword, start)) >= 0:
            for pos in range(idx, idx + len(hotword)):
                mask[pos] = True
            start = idx + len(hotword)
    return mask


def align(reference: str, prediction: str) -> List[Tuple[str, int, int]]:
    n, m = len(reference), len(prediction)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], trace[i][0] = i, "D"
    for j in range(1, m + 1):
        costs[0][j], trace[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1])
            delete_cost = costs[i - 1][j] + 1
            insert_cost = costs[i][j - 1] + 1
            best = min(sub_cost, delete_cost, insert_cost)
            costs[i][j] = best
            trace[i][j] = "M" if best == sub_cost else ("D" if best == delete_cost else "I")
    operations: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i or j:
        op = trace[i][j]
        if op == "M":
            operations.append(("E" if reference[i - 1] != prediction[j - 1] else "C", i - 1, j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            operations.append(("E", i - 1, -1))
            i -= 1
        else:
            operations.append(("I", i, j - 1))
            j -= 1
    operations.reverse()
    return operations


def insertion_is_hotword(mask: Sequence[bool], boundary: int) -> bool:
    left_hot = boundary > 0 and mask[boundary - 1]
    right_hot = boundary < len(mask) and mask[boundary]
    return left_hot and right_hot


def evaluate(
    predictions: Dict[str, Tuple[str, str]],
    designated: Dict[str, List[str]],
    vocabulary: Sequence[str],
) -> dict:
    counts = {
        "samples": 0,
        "reference_chars": 0,
        "total_edits": 0,
        "hotword_chars": 0,
        "hotword_edits": 0,
        "non_hotword_chars": 0,
        "non_hotword_edits": 0,
        "negative_utterances": 0,
        "negative_utterances_with_false_hotword": 0,
        "false_hotword_mentions": 0,
    }
    for utterance_id, (reference, prediction) in predictions.items():
        mask = hotword_mask(reference, designated.get(utterance_id, []))
        counts["samples"] += 1
        counts["reference_chars"] += len(reference)
        counts["hotword_chars"] += sum(mask)
        counts["non_hotword_chars"] += len(mask) - sum(mask)
        for op, ref_pos, _ in align(reference, prediction):
            if op == "C":
                continue
            counts["total_edits"] += 1
            in_hotword = insertion_is_hotword(mask, ref_pos) if op == "I" else mask[ref_pos]
            key = "hotword_edits" if in_hotword else "non_hotword_edits"
            counts[key] += 1
        if utterance_id not in designated:
            counts["negative_utterances"] += 1
            false_terms = [term for term in vocabulary if term not in reference and term in prediction]
            counts["false_hotword_mentions"] += len(false_terms)
            if false_terms:
                counts["negative_utterances_with_false_hotword"] += 1

    def ratio(num_key: str, den_key: str) -> float:
        return counts[num_key] / counts[den_key] if counts[den_key] else 0.0

    return {
        **counts,
        "cer": ratio("total_edits", "reference_chars"),
        "hotword_region_cer": ratio("hotword_edits", "hotword_chars"),
        "non_hotword_region_cer": ratio("non_hotword_edits", "non_hotword_chars"),
        "false_insertion_utterance_rate": ratio(
            "negative_utterances_with_false_hotword", "negative_utterances"
        ),
        "false_hotword_mentions_per_negative_utterance": ratio(
            "false_hotword_mentions", "negative_utterances"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--uttid", required=True, type=Path)
    parser.add_argument("--aligned", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prediction-field",
        default="auto",
        help="JSONL prediction field, including dotted paths such as input.asr_top1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uttids = read_uttids(args.uttid)
    designated = read_designated_hotwords(args.aligned)
    predictions = load_predictions(args.predictions, uttids, args.prediction_field)
    missing = [utterance_id for utterance_id in uttids if utterance_id not in predictions]
    if missing:
        raise ValueError(f"missing {len(missing)} test utterances, first={missing[0]}")
    predictions = {utterance_id: predictions[utterance_id] for utterance_id in uttids}
    result = {
        "predictions": str(args.predictions),
        "prediction_field": args.prediction_field,
        "alignment_policy": (
            "substitutions/deletions follow reference character spans; "
            "insertions count as hotword errors only when inside a hotword span"
        ),
        **evaluate(predictions, designated, sorted({term for terms in designated.values() for term in terms})),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
