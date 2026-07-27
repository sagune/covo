#!/usr/bin/env python
"""Evaluate a lightweight adaptive-evidence selector for ASR correction outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl
from covo.metrics import edit_distance
from covo.text import normalize_chinese_text


MODES = ["asr_only", "consensus", "hardneg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-asr-only", required=True)
    parser.add_argument("--dev-consensus", required=True)
    parser.add_argument("--dev-hardneg", required=True)
    parser.add_argument("--test-asr-only", required=True)
    parser.add_argument("--test-consensus", required=True)
    parser.add_argument("--test-hardneg", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--fallback", choices=["asr_only", "consensus"], default="asr_only")
    parser.add_argument("--target", choices=["bad", "noimprove"], default="bad")
    return parser.parse_args()


def _distance(a: str, b: str) -> int:
    return edit_distance(list(normalize_chinese_text(a)), list(normalize_chinese_text(b)))


def _load_triplet(paths: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    return {mode: list(read_jsonl(path)) for mode, path in paths.items()}


def _features(record: Dict[str, Any]) -> List[float]:
    input_block = record.get("input", {}) or {}
    top1 = normalize_chinese_text(input_block.get("asr_top1", ""))
    nbest = [normalize_chinese_text(item) for item in input_block.get("nbest", []) if str(item).strip()]
    if not nbest:
        nbest = [top1]

    sims = [SequenceMatcher(a=top1, b=item).ratio() for item in nbest]
    distances = [_distance(top1, item) for item in nbest]
    nonzero_distances = [item for item in distances if item > 0]

    consensus = input_block.get("nbest_consensus") or {}
    uncertain = list(consensus.get("uncertain_spans") or []) if isinstance(consensus, dict) else []
    stable = list(consensus.get("stable_spans") or []) if isinstance(consensus, dict) else []
    supports = [float(item.get("support", 1.0)) for item in uncertain]
    variant_counts = [len(item.get("variants") or []) for item in uncertain]

    return [
        float(len(top1)),
        float(len(nbest)),
        float(len(set(nbest)) / max(1, len(nbest))),
        float(sum(item == top1 for item in nbest) / max(1, len(nbest))),
        float(sum(sims) / max(1, len(sims))),
        float(min(sims) if sims else 1.0),
        float(max(distances) if distances else 0.0),
        float(sum(distances) / max(1, len(distances))),
        float(min(nonzero_distances) if nonzero_distances else 0.0),
        float(len(uncertain)),
        float(len(stable)),
        float(min(supports) if supports else 1.0),
        float(sum(supports) / len(supports) if supports else 1.0),
        float(max(variant_counts) if variant_counts else 0.0),
        float(sum(variant_counts)),
    ]


def _prepare(rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    n = min(len(rows[mode]) for mode in MODES)
    features: List[List[float]] = []
    items: List[Dict[str, Any]] = []
    best_labels: List[int] = []
    bad_labels: List[int] = []
    noimprove_labels: List[int] = []

    for idx in range(n):
        reference = str(rows["consensus"][idx]["reference"])
        base = str((rows["consensus"][idx].get("input", {}) or {}).get("asr_top1", ""))
        base_distance = _distance(base, reference)
        distances = {mode: _distance(str(rows[mode][idx]["prediction"]), reference) for mode in MODES}
        best_mode = min(MODES, key=lambda mode: (distances[mode], MODES.index(mode)))
        features.append(_features(rows["consensus"][idx]))
        items.append(
            {
                "base_distance": base_distance,
                "distances": distances,
                "chars": len(normalize_chinese_text(reference)),
            }
        )
        best_labels.append(MODES.index(best_mode))
        bad_labels.append(int(distances["hardneg"] > base_distance))
        noimprove_labels.append(int(distances["hardneg"] >= base_distance))

    return {
        "features": features,
        "items": items,
        "best_labels": best_labels,
        "bad_labels": bad_labels,
        "noimprove_labels": noimprove_labels,
    }


def _evaluate(items: List[Dict[str, Any]], choices: List[str]) -> Dict[str, Any]:
    edits = 0
    chars = 0
    improved = 0
    worsened = 0
    unchanged = 0
    counts: Counter[str] = Counter()

    for item, choice in zip(items, choices):
        distance = int(item["distances"][choice])
        base_distance = int(item["base_distance"])
        edits += distance
        chars += int(item["chars"])
        improved += int(distance < base_distance)
        worsened += int(distance > base_distance)
        unchanged += int(distance == base_distance)
        counts[choice] += 1

    return {
        "cer": float(edits / chars) if chars else 0.0,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "counts": dict(counts),
    }


def _fixed(items: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    return _evaluate(items, [mode] * len(items))


def _oracle(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    choices = [
        min(MODES, key=lambda mode: (item["distances"][mode], MODES.index(mode)))
        for item in items
    ]
    return _evaluate(items, choices)


def _train_risk_gate(dev: Dict[str, Any], test: Dict[str, Any], target: str, fallback: str) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier

    y_key = "bad_labels" if target == "bad" else "noimprove_labels"
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight="balanced",
        random_state=13,
        min_samples_leaf=4,
    )
    clf.fit(dev["features"], dev[y_key])

    dev_prob = clf.predict_proba(dev["features"])[:, 1]
    best = None
    for threshold in [idx / 100 for idx in range(5, 96, 5)]:
        choices = [fallback if prob > threshold else "hardneg" for prob in dev_prob]
        metrics = _evaluate(dev["items"], choices)
        key = (metrics["cer"], metrics["worsened"])
        if best is None or key < best[0]:
            best = (key, threshold, metrics)

    assert best is not None
    threshold = float(best[1])
    test_prob = clf.predict_proba(test["features"])[:, 1]
    test_choices = [fallback if prob > threshold else "hardneg" for prob in test_prob]

    return {
        "model": "RandomForestClassifier",
        "target": target,
        "fallback": fallback,
        "threshold": threshold,
        "dev": best[2],
        "test": _evaluate(test["items"], test_choices),
    }


def main() -> int:
    args = parse_args()
    dev = _prepare(
        _load_triplet(
            {
                "asr_only": args.dev_asr_only,
                "consensus": args.dev_consensus,
                "hardneg": args.dev_hardneg,
            }
        )
    )
    test = _prepare(
        _load_triplet(
            {
                "asr_only": args.test_asr_only,
                "consensus": args.test_consensus,
                "hardneg": args.test_hardneg,
            }
        )
    )

    result = {
        "dev_samples": len(dev["items"]),
        "test_samples": len(test["items"]),
        "dev_oracle_labels": dict(Counter(MODES[idx] for idx in dev["best_labels"])),
        "test_oracle_labels": dict(Counter(MODES[idx] for idx in test["best_labels"])),
        "fixed_dev": {mode: _fixed(dev["items"], mode) for mode in MODES},
        "fixed_test": {mode: _fixed(test["items"], mode) for mode in MODES},
        "oracle_dev": _oracle(dev["items"]),
        "oracle_test": _oracle(test["items"]),
        "risk_gate": _train_risk_gate(dev, test, target=str(args.target), fallback=str(args.fallback)),
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
