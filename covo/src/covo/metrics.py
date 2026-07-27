"""Lightweight text metrics for ASR correction experiments."""

from __future__ import annotations

from typing import Sequence

from .text import normalize_chinese_text


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i]
        for j, right_item in enumerate(right, 1):
            cost = 0 if left_item == right_item else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return int(previous[-1])


def cer(prediction: str, reference: str, *, normalize: bool = True) -> float:
    pred = normalize_chinese_text(prediction) if normalize else str(prediction)
    ref = normalize_chinese_text(reference) if normalize else str(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    return float(edit_distance(list(pred), list(ref)) / len(ref))
