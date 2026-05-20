"""Conservative rule baseline for N-best/pinyin ASR correction."""

from __future__ import annotations

from typing import Any, Dict, List
from collections import Counter

from .chinesehp import make_minimal_edits
from .edits import Edit, safe_apply_edits
from .metrics import edit_distance
from .text import normalize_chinese_text, to_pinyin_units


def _char_distance(left: str, right: str) -> int:
    return edit_distance(list(normalize_chinese_text(left)), list(normalize_chinese_text(right)))


def _pinyin_distance(left: str, right: str) -> int:
    return edit_distance(to_pinyin_units(left), to_pinyin_units(right))


def choose_rule_candidate(
    evidence: Dict[str, Any],
    *,
    max_char_distance: int = 4,
    max_pinyin_distance: int = 1,
    min_candidate_support: int = 2,
) -> str:
    """Choose a conservative N-best alternative close to ASR top1.

    This is not an oracle. It only switches to an N-best candidate when the
    alternative is very close to top1 and no farther phonetically. The goal is a
    simple baseline and pipeline check before learned correction.
    """
    asr_top1 = str(evidence.get("asr_top1", ""))
    nbest = [str(item) for item in evidence.get("nbest", []) or [] if str(item).strip()]
    if not asr_top1 or len(nbest) <= 1:
        return asr_top1

    normalized_counts = Counter(normalize_chinese_text(item) for item in nbest if normalize_chinese_text(item))
    best = asr_top1
    best_key = (10**9, 10**9)
    for candidate in nbest[1:]:
        if normalized_counts[normalize_chinese_text(candidate)] < int(min_candidate_support):
            continue
        char_dist = _char_distance(asr_top1, candidate)
        if char_dist <= 0 or char_dist > int(max_char_distance):
            continue
        py_dist = _pinyin_distance(asr_top1, candidate)
        if py_dist > int(max_pinyin_distance):
            continue
        key = (py_dist, char_dist)
        if key < best_key:
            best = candidate
            best_key = key
    return best


def make_rule_edits(
    evidence: Dict[str, Any],
    *,
    max_char_distance: int = 4,
    max_pinyin_distance: int = 1,
    min_candidate_support: int = 2,
) -> List[Edit]:
    asr_top1 = str(evidence.get("asr_top1", ""))
    candidate = choose_rule_candidate(
        evidence,
        max_char_distance=max_char_distance,
        max_pinyin_distance=max_pinyin_distance,
        min_candidate_support=min_candidate_support,
    )
    if normalize_chinese_text(candidate) == normalize_chinese_text(asr_top1):
        return []
    edits = make_minimal_edits(asr_top1, candidate)
    return [
        Edit(from_text=edit.from_text, to_text=edit.to_text, reason="rule_nbest_same_pinyin")
        for edit in edits
    ]


def apply_rule_correction(
    evidence: Dict[str, Any],
    *,
    max_char_distance: int = 4,
    max_pinyin_distance: int = 1,
    min_candidate_support: int = 2,
    max_total_changed_chars: int = 12,
) -> Dict[str, Any]:
    edits = make_rule_edits(
        evidence,
        max_char_distance=max_char_distance,
        max_pinyin_distance=max_pinyin_distance,
        min_candidate_support=min_candidate_support,
    )
    return safe_apply_edits(
        str(evidence.get("asr_top1", "")),
        {"edits": [edit.to_json() for edit in edits]},
        evidence,
        max_total_changed_chars=max_total_changed_chars,
    )
