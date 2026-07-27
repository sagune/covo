"""Text normalization and pinyin helpers for Chinese ASR correction."""

from __future__ import annotations

import re
import unicodedata
from typing import List


def normalize_chinese_text(text: str) -> str:
    """Normalize Chinese ASR text for matching and verifier checks."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", "", text)
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P"):
            continue
        chars.append(ch)
    return "".join(chars).strip()


def to_pinyin_units(text: str) -> List[str]:
    """Return pinyin units when pypinyin is available, otherwise characters."""
    text = normalize_chinese_text(text)
    if not text:
        return []
    try:
        from pypinyin import lazy_pinyin

        units = [str(unit).strip() for unit in lazy_pinyin(text, errors="default")]
        units = [unit for unit in units if unit]
        if units:
            return units
    except Exception:
        pass
    return list(text)


def joined_pinyin(text: str) -> str:
    return " ".join(to_pinyin_units(text))


def same_pinyin(left: str, right: str) -> bool:
    left_units = to_pinyin_units(left)
    right_units = to_pinyin_units(right)
    return bool(left_units and left_units == right_units)
