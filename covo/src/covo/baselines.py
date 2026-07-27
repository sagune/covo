"""Baseline metrics for ASR post-correction records."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from .metrics import edit_distance
from .text import normalize_chinese_text


def _nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _evidence(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("input", {})
    return value if isinstance(value, dict) else record


def _nbest(record: Dict[str, Any]) -> List[str]:
    evidence = _evidence(record)
    candidates = evidence.get("nbest", record.get("nbest", []))
    return [str(item) for item in candidates if str(item).strip()]


def _top1(record: Dict[str, Any]) -> str:
    evidence = _evidence(record)
    value = evidence.get("asr_top1", "")
    if value:
        return str(value)
    candidates = _nbest(record)
    return candidates[0] if candidates else ""


def oracle_nbest(record: Dict[str, Any], *, top_k: int = 0) -> Tuple[str, int, int]:
    """Return the best N-best candidate under reference CER.

    The tuple is ``(candidate, index, edit_distance)``. This is an oracle upper
    bound, not a deployable baseline, because it uses the reference.
    """

    reference = normalize_chinese_text(str(_nested_get(record, "reference", "")))
    candidates = _nbest(record)
    if top_k:
        candidates = candidates[: max(1, int(top_k))]
    if not candidates:
        candidates = [_top1(record)]
    best_text = candidates[0]
    best_index = 0
    best_distance = edit_distance(list(normalize_chinese_text(best_text)), list(reference))
    for index, candidate in enumerate(candidates[1:], 1):
        distance = edit_distance(list(normalize_chinese_text(candidate)), list(reference))
        if distance < best_distance:
            best_text = candidate
            best_index = index
            best_distance = distance
    return best_text, best_index, int(best_distance)


def evaluate_baselines(records: Iterable[Dict[str, Any]], *, oracle_top_k: int = 0) -> Dict[str, Any]:
    samples = 0
    ref_chars = 0
    top1_edits = 0
    oracle_edits = 0
    oracle_exact = 0
    oracle_changed = 0
    oracle_index_counts: Counter[str] = Counter()

    for record in records:
        reference = normalize_chinese_text(str(_nested_get(record, "reference", "")))
        if not reference:
            continue
        samples += 1
        ref_chars += len(reference)
        top1 = _top1(record)
        top1_distance = edit_distance(list(normalize_chinese_text(top1)), list(reference))
        oracle_text, oracle_index, oracle_distance = oracle_nbest(record, top_k=oracle_top_k)
        top1_edits += int(top1_distance)
        oracle_edits += int(oracle_distance)
        oracle_exact += int(oracle_distance == 0)
        oracle_changed += int(normalize_chinese_text(oracle_text) != normalize_chinese_text(top1))
        oracle_index_counts[str(oracle_index)] += 1

    return {
        "samples": samples,
        "reference_chars": ref_chars,
        "oracle_top_k": int(oracle_top_k),
        "no_correction": {
            "cer": float(top1_edits / ref_chars) if ref_chars else 0.0,
            "total_edits": top1_edits,
        },
        "oracle_nbest": {
            "cer": float(oracle_edits / ref_chars) if ref_chars else 0.0,
            "total_edits": oracle_edits,
            "exact_samples": oracle_exact,
            "exact_rate": float(oracle_exact / samples) if samples else 0.0,
            "changed_from_top1_samples": oracle_changed,
            "index_counts": dict(sorted(oracle_index_counts.items(), key=lambda item: int(item[0]))),
        },
    }
