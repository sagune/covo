#!/usr/bin/env python
"""Build explicitly test-leaked domain-term DPO pairs for diagnostics only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from analysis.evaluate_filler_normalized_cer import edit_distance, normalize_text


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(value: Any) -> str:
    return normalize_text(value, [], numbers=True)


def distance(value: Any, reference: Any) -> int:
    return int(edit_distance(list(norm(value)), list(norm(reference))))


def unique_candidates(record: Dict[str, Any], limit: int) -> List[str]:
    block = record.get("input", {}) or {}
    values = [block.get("asr_top1", "")] + list(block.get("nbest", []) or [])
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
        if len(output) >= limit:
            break
    return output


def prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def load_terms(path: str | Path, min_length: int) -> List[str]:
    terms = {normalize_text(line.strip(), []) for line in Path(path).read_text(encoding="utf-8").splitlines()}
    return sorted((term for term in terms if len(term) >= min_length), key=lambda term: (-len(term), term))


def make_pair(record: Dict[str, Any], chosen: str, rejected: str, pair_type: str, terms: List[str]) -> Dict[str, Any]:
    reference = str(record.get("reference", ""))
    return {
        "id": str(record.get("id", "")),
        "source": "shuili_video_test_leak_diagnostic",
        "split": "test_leak_train",
        "pair_type": pair_type,
        "leakage_warning": "Uses test references and test-derived domain terms; never report as an untuned test result.",
        "domain_terms": terms,
        "prompt_messages": prompt_messages(record),
        "chosen": {"text": normalize_text(chosen, [])},
        "rejected": {"text": normalize_text(rejected, [])},
        "chosen_edit_distance": distance(chosen, reference),
        "rejected_edit_distance": distance(rejected, reference),
    }


def build(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    terms = load_terms(args.terms, int(args.min_term_length))
    correction: List[Dict[str, Any]] = []
    preservation: List[Dict[str, Any]] = []
    term_rows = 0
    for record in read_jsonl(args.input):
        reference = str(record.get("reference", ""))
        ref_norm = normalize_text(reference, [])
        matched = [term for term in terms if term in ref_norm]
        if not matched:
            continue
        term_rows += 1
        candidates = unique_candidates(record, int(args.max_candidates))
        if len(candidates) < 2:
            continue
        scores = [distance(candidate, reference) for candidate in candidates]
        best_idx = min(range(len(candidates)), key=lambda idx: (scores[idx], idx))
        chosen = candidates[best_idx]
        chosen_norm = normalize_text(chosen, [])
        supported = [term for term in matched if term in chosen_norm]

        if best_idx != 0 and scores[best_idx] < scores[0] and supported:
            rejected_indices = [0]
            rejected_indices.extend(
                idx
                for idx in range(1, len(candidates))
                if idx != best_idx
                and scores[idx] > scores[best_idx]
                and any(term not in normalize_text(candidates[idx], []) for term in supported)
            )
            for rejected_idx in list(dict.fromkeys(rejected_indices))[: int(args.max_rejected_per_row)]:
                pair = make_pair(record, chosen, candidates[rejected_idx], "test_leak_domain_correction", supported)
                correction.extend([pair] * int(args.correction_repeat))
        elif best_idx == 0 and scores[0] == 0 and supported:
            rejected = [
                idx
                for idx in range(1, len(candidates))
                if scores[idx] > 0 and any(term not in normalize_text(candidates[idx], []) for term in supported)
            ]
            if rejected:
                preservation.append(
                    make_pair(record, chosen, candidates[rejected[0]], "test_leak_domain_preservation", supported)
                )

    rng = random.Random(int(args.seed))
    rng.shuffle(correction)
    rng.shuffle(preservation)
    if int(args.max_preservation_pairs) >= 0:
        preservation = preservation[: int(args.max_preservation_pairs)]
    rows = correction + preservation
    rng.shuffle(rows)
    stats = {
        "test_derived_terms": len(terms),
        "rows_containing_terms": term_rows,
        "correction_pairs_after_repeat": len(correction),
        "preservation_pairs": len(preservation),
        "total_pairs": len(rows),
    }
    return rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--terms", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-term-length", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-rejected-per-row", type=int, default=2)
    parser.add_argument("--correction-repeat", type=int, default=3)
    parser.add_argument("--max-preservation-pairs", type=int, default=144)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, stats = build(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(output), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
