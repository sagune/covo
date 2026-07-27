#!/usr/bin/env python
"""Audit CB-SenseVoice -> COVO correction errors and n-best oracle ceilings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_COVO_SRC = str(Path(__file__).resolve().parents[2] / "covo" / "src")
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.metrics import edit_distance  # type: ignore
from covo.text import normalize_chinese_text  # type: ignore


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


def char_distance(left: Any, right: Any) -> int:
    return int(edit_distance(list(norm(left)), list(norm(right))))


def cer(edits: int, chars: int) -> float:
    return float(edits / chars) if chars else 0.0


def unique_texts(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def best_candidate(reference: str, candidates: List[str]) -> Tuple[str, int]:
    if not candidates:
        return "", 10**9
    scored = [(char_distance(candidate, reference), idx, candidate) for idx, candidate in enumerate(candidates)]
    scored.sort(key=lambda row: (row[0], row[1]))
    return scored[0][2], scored[0][0]


def keyword_mentions(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            text = norm(item.get("mention", ""))
            if text:
                output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def prompt_hotwords(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if isinstance(item, dict):
            text = norm(item.get("text", ""))
            if text:
                output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def hit_count(text: str, keywords: List[str]) -> int:
    text_norm = norm(text)
    return sum(1 for keyword in keywords if keyword and keyword in text_norm)


def classify_delta(base_d: int, pred_d: int) -> str:
    if base_d == 0 and pred_d == 0:
        return "base_correct_kept"
    if base_d == 0 and pred_d > 0:
        return "base_correct_broken"
    if base_d > 0 and pred_d == 0:
        return "fixed_to_exact"
    if pred_d < base_d:
        return "improved_partial"
    if pred_d == base_d:
        return "unchanged_error"
    return "worsened_error"


def audit(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows = list(read_jsonl(args.predictions))
    total_chars = 0
    sums = {
        "base_edits": 0,
        "pred_edits": 0,
        "nbest_oracle_edits": 0,
        "pred_or_nbest_oracle_edits": 0,
    }
    counts: Dict[str, int] = {}
    hotword = {
        "total_mentions": 0,
        "base_hits": 0,
        "pred_hits": 0,
        "nbest_oracle_hits": 0,
        "base_lost_by_pred": 0,
        "pred_gained_over_base": 0,
        "prompt_false_in_base": 0,
        "prompt_false_in_pred": 0,
    }
    reach = {
        "nbest_exact": 0,
        "pred_exact": 0,
        "base_exact": 0,
        "nbest_better_than_pred": 0,
        "nbest_better_than_base": 0,
        "pred_better_than_nbest": 0,
    }
    error_rows: List[Dict[str, Any]] = []

    for idx, record in enumerate(rows):
        reference = str(nested_get(record, args.reference_field, ""))
        prediction = str(nested_get(record, args.prediction_field, ""))
        input_block = dict(record.get("input", {}) or {})
        base = str(nested_get(record, args.baseline_field, ""))
        if not base:
            base = str(input_block.get("asr_top1", ""))
        ref_norm = norm(reference)
        if not ref_norm:
            continue

        nbest = unique_texts(list(input_block.get("nbest", []) or []))
        if base:
            nbest = unique_texts([base] + nbest)
        nbest_best, nbest_d = best_candidate(reference, nbest)
        pred_or_nbest_best, pred_or_nbest_d = best_candidate(reference, unique_texts([prediction] + nbest))

        base_d = char_distance(base, reference)
        pred_d = char_distance(prediction, reference)
        total_chars += len(ref_norm)
        sums["base_edits"] += base_d
        sums["pred_edits"] += pred_d
        sums["nbest_oracle_edits"] += nbest_d
        sums["pred_or_nbest_oracle_edits"] += pred_or_nbest_d

        label = classify_delta(base_d, pred_d)
        counts[label] = counts.get(label, 0) + 1
        reach["base_exact"] += int(base_d == 0)
        reach["pred_exact"] += int(pred_d == 0)
        reach["nbest_exact"] += int(nbest_d == 0)
        reach["nbest_better_than_pred"] += int(nbest_d < pred_d)
        reach["nbest_better_than_base"] += int(nbest_d < base_d)
        reach["pred_better_than_nbest"] += int(pred_d < nbest_d)

        mentions = keyword_mentions(input_block)
        prompts = prompt_hotwords(input_block)
        false_prompts = [kw for kw in prompts if kw and kw not in ref_norm]
        base_hits = hit_count(base, mentions)
        pred_hits = hit_count(prediction, mentions)
        nbest_hits = hit_count(nbest_best, mentions)
        hotword["total_mentions"] += len(mentions)
        hotword["base_hits"] += base_hits
        hotword["pred_hits"] += pred_hits
        hotword["nbest_oracle_hits"] += nbest_hits
        hotword["base_lost_by_pred"] += sum(1 for kw in mentions if kw in norm(base) and kw not in norm(prediction))
        hotword["pred_gained_over_base"] += sum(1 for kw in mentions if kw not in norm(base) and kw in norm(prediction))
        hotword["prompt_false_in_base"] += sum(1 for kw in false_prompts if kw in norm(base))
        hotword["prompt_false_in_pred"] += sum(1 for kw in false_prompts if kw in norm(prediction))

        if pred_d or base_d or nbest_d < pred_d:
            error_rows.append(
                {
                    "idx": idx,
                    "id": record.get("id", ""),
                    "label": label,
                    "base_d": base_d,
                    "pred_d": pred_d,
                    "nbest_oracle_d": nbest_d,
                    "pred_gain": base_d - pred_d,
                    "nbest_possible_gain_vs_pred": pred_d - nbest_d,
                    "ref": reference,
                    "base": base,
                    "pred": prediction,
                    "nbest_oracle": nbest_best,
                    "keyword_mentions": " ".join(mentions),
                    "prompt_hotwords": " ".join(prompts),
                    "false_prompt_hotwords": " ".join(false_prompts),
                }
            )

    for key in ("base_hits", "pred_hits", "nbest_oracle_hits"):
        hotword[key.replace("hits", "recall")] = cer(hotword[key], hotword["total_mentions"])

    summary = {
        "predictions": str(args.predictions),
        "samples": sum(counts.values()),
        "reference_chars": total_chars,
        "cer": {
            "base": cer(sums["base_edits"], total_chars),
            "prediction": cer(sums["pred_edits"], total_chars),
            "nbest_oracle": cer(sums["nbest_oracle_edits"], total_chars),
            "prediction_or_nbest_oracle": cer(sums["pred_or_nbest_oracle_edits"], total_chars),
        },
        "edit_totals": sums,
        "delta_counts": counts,
        "oracle_reachability": reach,
        "hotword": hotword,
    }
    error_rows.sort(key=lambda row: (int(row["nbest_possible_gain_vs_pred"]), int(row["pred_d"])), reverse=True)
    return summary, error_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--baseline-field", default="input.asr_top1")
    parser.add_argument("--max-cases", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, rows = audit(args)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_rows = rows[: int(args.max_cases)]
    with Path(args.cases_output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()) if out_rows else ["idx"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
