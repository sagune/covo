#!/usr/bin/env python3
"""Build balanced English COVO SFT data from SenseVoice N-best evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


SYSTEM = (
    "You are a conservative English ASR post-correction model. Use the ranked "
    "SenseVoice hypotheses and acoustic evidence to recover the spoken words. "
    "Make only necessary corrections. Never paraphrase, summarize, add facts, "
    "or improve writing style. Preserve a reliable first hypothesis unchanged. "
    "Output exactly one JSON object with a single text field."
)


def distance(reference: str, hypothesis: str) -> int:
    left = reference.split()
    right = hypothesis.split()
    previous = list(range(len(right) + 1))
    for i, left_word in enumerate(left, 1):
        current = [i]
        for j, right_word in enumerate(right, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + int(left_word != right_word),
            ))
        previous = current
    return previous[-1]


def stable_bucket(value: str, modulus: int = 10_000) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16) % modulus


def classify(reference: str, hypotheses: list[str]) -> tuple[str, int, int]:
    top1_distance = distance(reference, hypotheses[0])
    distances = [distance(reference, item) for item in hypotheses]
    oracle_distance = min(distances)
    if top1_distance == 0:
        category = "preserve"
    elif reference in hypotheses[1:]:
        category = "exact_in_nbest"
    elif oracle_distance < top1_distance:
        category = "recoverable"
    else:
        category = "hard_generate"
    return category, top1_distance, oracle_distance


def make_record(row: dict, category: str, hypotheses: list[str], scores: list[float], view: str) -> dict:
    best_score = scores[0] if scores else 0.0
    ranked = []
    for index, hypothesis in enumerate(hypotheses):
        score = scores[index] if index < len(scores) else best_score
        ranked.append(f"{index + 1}. [relative_acoustic_score={score - best_score:.3f}] {hypothesis}")
    user = (
        f"Utterance: {row['id']}\n"
        f"SenseVoice top-1: {hypotheses[0]}\n"
        "Ranked SenseVoice hypotheses:\n"
        + "\n".join(ranked)
        + "\nReturn JSON only."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps({"text": row["reference"]}, ensure_ascii=True)},
        ],
        "metadata": {
            "id": row["id"],
            "split": row.get("split", ""),
            "category": category,
            "view": view,
        },
    }


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            hypotheses = [str(item).strip() for item in row.get("nbest", []) if str(item).strip()]
            if not row.get("id") or not row.get("reference") or not hypotheses:
                raise ValueError(f"{path}:{line_number}: incomplete evidence")
            row["nbest"] = hypotheses
            row["candidate_scores"] = [float(item) for item in row.get("candidate_scores", [])]
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--preserve-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    rows = load_rows(args.input)
    records = []
    preserve_rows = []
    categories = Counter()
    total_reference_words = 0
    total_top1_errors = 0
    total_oracle_errors = 0

    for row in rows:
        hypotheses = row["nbest"]
        scores = row["candidate_scores"]
        category, top1_errors, oracle_errors = classify(row["reference"], hypotheses)
        categories[category] += 1
        total_reference_words += len(row["reference"].split())
        total_top1_errors += top1_errors
        total_oracle_errors += oracle_errors
        base = make_record(row, category, hypotheses, scores, "base")
        records.append(base)
        if category == "preserve":
            preserve_rows.append((row, category))

        if args.train:
            bucket = stable_bucket(row["id"])
            augment = (
                category in {"exact_in_nbest", "recoverable"} and bucket < 5000
            ) or (category == "hard_generate" and bucket < 1500)
            if augment and len(hypotheses) > 5:
                keep = 6 + (bucket % min(3, len(hypotheses) - 5))
                records.append(make_record(row, category, hypotheses[:keep], scores[:keep], "candidate_dropout"))

    preserve_added = 0
    if args.train and preserve_rows and 0.0 < args.preserve_ratio < 1.0:
        non_preserve = sum(1 for record in records if record["metadata"]["category"] != "preserve")
        current_preserve = len(records) - non_preserve
        required_preserve = math.ceil(args.preserve_ratio * non_preserve / (1.0 - args.preserve_ratio))
        rng = random.Random(args.seed)
        while current_preserve < required_preserve:
            row, category = preserve_rows[preserve_added % len(preserve_rows)]
            hypotheses = row["nbest"]
            scores = row["candidate_scores"]
            keep = max(3, len(hypotheses) - 2 - (preserve_added % 3))
            record = make_record(row, category, hypotheses[:keep], scores[:keep], "preserve_anchor")
            record["metadata"]["replica"] = preserve_added + 1
            records.append(record)
            current_preserve += 1
            preserve_added += 1
        rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "evidence_rows": len(rows),
        "sft_rows": len(records),
        "categories": dict(categories),
        "preserve_anchor_rows_added": preserve_added,
        "top1_wer": total_top1_errors / max(total_reference_words, 1),
        "nbest_oracle_wer": total_oracle_errors / max(total_reference_words, 1),
        "reference_words": total_reference_words,
        "train_augmentation": bool(args.train),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
