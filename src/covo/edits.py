"""JSON edit validation and application for constrained ASR correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from .text import normalize_chinese_text, same_pinyin


@dataclass(frozen=True)
class Edit:
    from_text: str
    to_text: str
    reason: str = ""

    def to_json(self) -> Dict[str, str]:
        return {
            "from": self.from_text,
            "to": self.to_text,
            "reason": self.reason,
        }


def parse_edits(value: Any) -> List[Edit]:
    """Parse model output shaped as either a list or {"edits": [...]}."""
    if isinstance(value, dict):
        value = value.get("edits", [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("edits must be a list or an object with an edits list")

    edits: List[Edit] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each edit must be a JSON object")
        from_text = str(item.get("from", "")).strip()
        to_text = str(item.get("to", "")).strip()
        reason = str(item.get("reason", "")).strip()
        edits.append(Edit(from_text=from_text, to_text=to_text, reason=reason))
    return edits


def apply_edits(text: str, edits: Iterable[Edit]) -> Tuple[str, List[Edit], List[str]]:
    """Apply edits once each, left to right, returning rejected reasons too."""
    current = str(text)
    applied: List[Edit] = []
    rejected: List[str] = []
    for edit in edits:
        if not edit.from_text:
            rejected.append("empty_from")
            continue
        if edit.from_text not in current:
            rejected.append(f"missing_from:{edit.from_text}")
            continue
        current = current.replace(edit.from_text, edit.to_text, 1)
        applied.append(edit)
    return current, applied, rejected


def _support_texts(evidence: Dict[str, Any]) -> List[str]:
    texts = []
    for key in ("asr_top1", "reference", "corrected_text"):
        if evidence.get(key):
            texts.append(str(evidence[key]))
    for item in evidence.get("nbest", []) or []:
        texts.append(str(item))
    for item in evidence.get("hotwords", []) or []:
        if isinstance(item, dict):
            texts.append(str(item.get("text", "")))
        else:
            texts.append(str(item))
    return [normalize_chinese_text(text) for text in texts if str(text).strip()]


def validate_edits(
    edits: Iterable[Edit],
    evidence: Dict[str, Any],
    *,
    max_total_changed_chars: int = 12,
    require_support: bool = True,
) -> Tuple[bool, List[str]]:
    """Conservative verifier for correction edits.

    A replacement is accepted only if the source span is supported by the ASR
    side and the target span is supported by N-best/hotwords or same-pinyin
    evidence. This keeps the first version useful before CB-Whisper integration.
    """
    asr_top1 = str(evidence.get("asr_top1", ""))
    nbest = [str(item) for item in evidence.get("nbest", []) or []]
    source_texts = [normalize_chinese_text(asr_top1)] + [normalize_chinese_text(x) for x in nbest]
    support_texts = _support_texts(evidence)

    reasons: List[str] = []
    changed = 0
    for edit in edits:
        from_norm = normalize_chinese_text(edit.from_text)
        to_norm = normalize_chinese_text(edit.to_text)
        if not from_norm:
            reasons.append("empty_span")
            continue
        if from_norm == to_norm:
            reasons.append(f"identity_edit:{from_norm}")
            continue
        if not any(from_norm in text for text in source_texts):
            reasons.append(f"from_not_supported:{from_norm}")
            continue
        changed += max(len(from_norm), len(to_norm))
        if changed > max_total_changed_chars:
            reasons.append("too_many_changed_chars")
            continue

        target_supported = bool(to_norm) and any(to_norm in text for text in support_texts)
        phonetic_supported = bool(to_norm) and same_pinyin(from_norm, to_norm)
        deletion_edit = not bool(to_norm)
        if require_support and not (target_supported or phonetic_supported or deletion_edit):
            reasons.append(f"to_not_supported:{to_norm}")

    return len(reasons) == 0, reasons


def safe_apply_edits(
    text: str,
    edits_value: Any,
    evidence: Dict[str, Any],
    *,
    max_total_changed_chars: int = 12,
) -> Dict[str, Any]:
    edits = parse_edits(edits_value)
    ok, reasons = validate_edits(
        edits,
        evidence,
        max_total_changed_chars=max_total_changed_chars,
    )
    if not ok:
        return {
            "accepted": False,
            "text": str(text),
            "reasons": reasons,
            "applied_edits": [],
        }
    corrected, applied, rejected = apply_edits(str(text), edits)
    accepted = len(rejected) == 0
    return {
        "accepted": accepted,
        "text": corrected if accepted else str(text),
        "reasons": rejected,
        "applied_edits": [edit.to_json() for edit in applied] if accepted else [],
    }
