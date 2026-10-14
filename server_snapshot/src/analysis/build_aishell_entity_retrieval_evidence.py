#!/usr/bin/env python3
"""Build phonetic entity-retrieval candidates for AISHELL COVO evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pypinyin import lazy_pinyin

try:
    from analysis.evaluate_aishell_ner import load_annotations, normalize_text
except ModuleNotFoundError:
    from evaluate_aishell_ner import load_annotations, normalize_text


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def entity_counts(paths: list[Path], min_length: int, max_length: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        annotations = load_annotations(path)
        counts.update(
            entity
            for _, _, entities in annotations.values()
            for entity in entities
            if min_length <= len(entity) <= max_length
        )
    return counts


class PhoneticEntityIndex:
    def __init__(self, counts: Counter[str]) -> None:
        self.counts = counts
        self.exact: dict[tuple[str, ...], list[str]] = defaultdict(list)
        self.wildcard: dict[tuple[int, int, tuple[str, ...]], list[str]] = defaultdict(list)
        for entity in counts:
            pinyin = tuple(lazy_pinyin(entity))
            self.exact[pinyin].append(entity)
            for position in range(len(pinyin)):
                key = (len(pinyin), position, pinyin[:position] + pinyin[position + 1 :])
                self.wildcard[key].append(entity)

    def retrieve(self, text: str, max_distance: int, max_candidates: int) -> list[dict[str, Any]]:
        text = normalize_text(text)
        text_pinyin = tuple(lazy_pinyin(text))
        best: dict[str, dict[str, Any]] = {}
        lengths = sorted({len(key) for key in self.exact})
        for length in lengths:
            if length > len(text):
                continue
            for start in range(len(text) - length + 1):
                window = text_pinyin[start : start + length]
                candidates = set(self.exact.get(window, []))
                if max_distance >= 1:
                    for position in range(length):
                        key = (length, position, window[:position] + window[position + 1 :])
                        candidates.update(self.wildcard.get(key, []))
                surface = text[start : start + length]
                for entity in candidates:
                    if entity == surface:
                        continue
                    entity_pinyin = tuple(lazy_pinyin(entity))
                    distance = sum(left != right for left, right in zip(window, entity_pinyin))
                    if distance > max_distance:
                        continue
                    row = {
                        "entity": entity,
                        "surface": surface,
                        "start": start,
                        "end": start + length,
                        "phonetic_distance": distance,
                        "frequency": self.counts[entity],
                    }
                    previous = best.get(entity)
                    rank = (distance, -length, -self.counts[entity], start)
                    if previous is None:
                        best[entity] = row
                    else:
                        previous_rank = (
                            previous["phonetic_distance"],
                            -(previous["end"] - previous["start"]),
                            -previous["frequency"],
                            previous["start"],
                        )
                        if rank < previous_rank:
                            best[entity] = row

        rows = sorted(
            best.values(),
            key=lambda row: (
                row["phonetic_distance"],
                -(row["end"] - row["start"]),
                -row["frequency"],
                row["start"],
                row["entity"],
            ),
        )
        return rows[:max_candidates]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-annotations", type=Path, action="append", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--max-distance", type=int, choices=(0, 1), default=1)
    parser.add_argument("--min-entity-length", type=int, default=2)
    parser.add_argument("--max-entity-length", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    counts = entity_counts(args.entity_annotations, args.min_entity_length, args.max_entity_length)
    index = PhoneticEntityIndex(counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = triggered = 0
    with args.output.open("w", encoding="utf-8") as output:
        for record in read_jsonl(args.input):
            if args.limit and written >= args.limit:
                break
            top1 = str(record.get(args.prediction_field, "")).strip()
            retrieval = index.retrieve(top1, args.max_distance, args.max_candidates)
            variants = []
            seen = {normalize_text(top1)}
            for item in retrieval:
                normalized = normalize_text(top1)
                variant = normalized[: item["start"]] + item["entity"] + normalized[item["end"] :]
                key = normalize_text(variant)
                if key and key not in seen:
                    variants.append(variant)
                    seen.add(key)
                    item["variant"] = variant
            triggered += bool(variants)
            evidence = {
                "id": str(record.get("id", "")),
                "source": "sensevoice_entity_retrieval",
                "dataset": "aishell1_ner",
                "split": "test",
                "reference": record.get("reference", ""),
                "input": {
                    "asr_top1": top1,
                    "nbest": [top1, *variants],
                    "prompt_hotwords": [
                        {
                            "text": item["entity"],
                            "weight": 1.0 - item["phonetic_distance"] / max(len(item["entity"]), 1),
                            "source": "phonetic_entity_retrieval",
                        }
                        for item in retrieval
                    ],
                    "entity_retrieval": retrieval,
                },
            }
            output.write(json.dumps(evidence, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
    print(json.dumps({"written": written, "triggered": triggered, "entities": len(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
