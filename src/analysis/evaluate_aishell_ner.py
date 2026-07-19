#!/usr/bin/env python3
"""Evaluate ASR JSONL predictions with AISHELL-NER entity-aware metrics."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


PUNCT_RE = re.compile(r"[\s\u3000，。！？、；：,.!?;:'\"“”‘’（）()【】\[\]《》<>·—…-]+")
ENTITY_MARKERS = {"<": ">", "[": "]", "(": ")"}


def normalize_text(value: Any) -> str:
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", str(value or ""))).lower()


def parse_annotated_text(value: str) -> tuple[str, list[bool], list[str]]:
    text: list[str] = []
    labels: list[bool] = []
    entities: list[str] = []
    index = 0
    while index < len(value):
        char = value[index]
        closing = ENTITY_MARKERS.get(char)
        if closing is not None:
            end = value.find(closing, index + 1)
            if end < 0:
                raise ValueError(f"unclosed entity marker in: {value}")
            entity = normalize_text(value[index + 1 : end])
            text.extend(entity)
            labels.extend([True] * len(entity))
            if entity:
                entities.append(entity)
            index = end + 1
            continue
        normalized = normalize_text(char)
        text.extend(normalized)
        labels.extend([False] * len(normalized))
        index += 1
    return "".join(text), labels, entities


def load_annotations(path: Path) -> dict[str, tuple[str, list[bool], list[str]]]:
    rows: dict[str, tuple[str, list[bool], list[str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, annotated = line.split(maxsplit=1)
        rows[utt_id] = parse_annotated_text(annotated)
    return rows


def load_predictions(path: Path, field: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            utt_id = str(row.get("id", ""))
            if utt_id:
                rows[utt_id] = normalize_text(row.get(field, ""))
    return rows


def align_counts(reference: str, hypothesis: str, entity_labels: list[bool]) -> dict[str, int]:
    n, m = len(reference), len(hypothesis)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    steps = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0], steps[i][0] = i, "D"
    for j in range(1, m + 1):
        costs[0][j], steps[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            choices = [(sub_cost, "M"), (costs[i - 1][j] + 1, "D"), (costs[i][j - 1] + 1, "I")]
            costs[i][j], steps[i][j] = min(choices, key=lambda item: item[0])

    counts = {"ne_edits": 0, "nne_edits": 0, "ne_chars": sum(entity_labels), "nne_chars": n - sum(entity_labels)}
    i, j = n, m
    while i or j:
        step = steps[i][j]
        if step == "M":
            if reference[i - 1] != hypothesis[j - 1]:
                counts["ne_edits" if entity_labels[i - 1] else "nne_edits"] += 1
            i -= 1
            j -= 1
        elif step == "D":
            counts["ne_edits" if entity_labels[i - 1] else "nne_edits"] += 1
            i -= 1
        else:
            # Attribute an insertion to an entity only when it lies inside an entity span.
            inside_entity = i > 0 and i < n and entity_labels[i - 1] and entity_labels[i]
            counts["ne_edits" if inside_entity else "nne_edits"] += 1
            j -= 1
    return counts


def evaluate(annotations: dict[str, tuple[str, list[bool], list[str]]], predictions: dict[str, str]) -> dict[str, Any]:
    totals = {"ne_edits": 0, "nne_edits": 0, "ne_chars": 0, "nne_chars": 0}
    entity_total = entity_hits = missing = 0
    for utt_id, (reference, labels, entities) in annotations.items():
        if utt_id not in predictions:
            missing += 1
            continue
        hypothesis = predictions[utt_id]
        counts = align_counts(reference, hypothesis, labels)
        for key, value in counts.items():
            totals[key] += value
        entity_total += len(entities)
        entity_hits += sum(entity in hypothesis for entity in entities)

    all_edits = totals["ne_edits"] + totals["nne_edits"]
    all_chars = totals["ne_chars"] + totals["nne_chars"]
    return {
        "utterances": len(annotations) - missing,
        "missing_predictions": missing,
        "cer": all_edits / all_chars if all_chars else 0.0,
        "nne_cer": totals["nne_edits"] / totals["nne_chars"] if totals["nne_chars"] else 0.0,
        "ne_cer": totals["ne_edits"] / totals["ne_chars"] if totals["ne_chars"] else 0.0,
        "ne_recall": entity_hits / entity_total if entity_total else 0.0,
        "entity_hits": entity_hits,
        "entity_total": entity_total,
        **totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(load_annotations(args.annotations), load_predictions(args.predictions, args.prediction_field))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
