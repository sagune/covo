"""Robust parsing for model-generated JSON edits."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .edits import Edit, parse_edits


_FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _candidate_json_strings(text: str) -> List[str]:
    text = str(text).strip()
    candidates: List[str] = []
    for match in _FENCED_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text)

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
    return list(dict.fromkeys(c for c in candidates if c))


def parse_model_edits(text: str) -> Tuple[List[Edit], List[str]]:
    """Extract edits from noisy model text.

    Returns parsed edits and warnings. When parsing fails, returns an empty edit
    list so callers can safely fall back to the original ASR output.
    """
    warnings: List[str] = []
    for candidate in _candidate_json_strings(text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            edits = parse_edits(value)
            return edits, warnings
        except ValueError as exc:
            warnings.append(str(exc))
            continue
    warnings.append("no_valid_json_edits")
    return [], warnings


def parse_model_edits_json(text: str) -> Dict[str, Any]:
    edits, warnings = parse_model_edits(text)
    return {
        "edits": [edit.to_json() for edit in edits],
        "parse_warnings": warnings,
    }
