#!/usr/bin/env python
"""Annotate ASR correction SFT data with phonetic edit evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import parse_edits
from covo.io import read_jsonl, write_jsonl
from covo.metrics import edit_distance
from covo.text import joined_pinyin, normalize_chinese_text, same_pinyin, to_pinyin_units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trap-limit", type=int, default=3)
    return parser.parse_args()


def _pinyin_distance(left: str, right: str) -> int:
    return edit_distance(to_pinyin_units(left), to_pinyin_units(right))


def _support_count(span: str, texts: List[str]) -> int:
    span = normalize_chinese_text(span)
    if not span:
        return 0
    return sum(1 for text in texts if span in normalize_chinese_text(text))


def _edit_type(from_text: str, to_text: str) -> str:
    from_norm = normalize_chinese_text(from_text)
    to_norm = normalize_chinese_text(to_text)
    if not to_norm:
        return "deletion"
    if len(to_norm) > len(from_norm):
        return "insertion_or_expansion"
    if len(from_norm) == len(to_norm) == 1:
        return "same_pinyin_substitution" if same_pinyin(from_norm, to_norm) else "single_char_substitution"
    return "span_rewrite"


def _same_pinyin_traps(asr_top1: str, nbest: List[str], limit: int) -> List[Dict[str, Any]]:
    asr_norm = normalize_chinese_text(asr_top1)
    asr_py = joined_pinyin(asr_top1)
    traps: List[Dict[str, Any]] = []
    seen = {asr_norm}
    for rank, hyp in enumerate(nbest[1:], 2):
        hyp_norm = normalize_chinese_text(hyp)
        if not hyp_norm or hyp_norm in seen:
            continue
        seen.add(hyp_norm)
        if joined_pinyin(hyp_norm) != asr_py:
            continue
        traps.append({"rank": rank, "candidate": hyp_norm})
        if len(traps) >= limit:
            break
    return traps


def convert_record(record: Dict[str, Any], *, trap_limit: int) -> Dict[str, Any]:
    input_block = dict(record.get("input", {}) or {})
    output = dict(record.get("output", {}) or {})
    asr_top1 = str(input_block.get("asr_top1", ""))
    nbest = [str(item) for item in input_block.get("nbest", []) or []]
    support_texts = [asr_top1, *nbest]
    edits = parse_edits(output)

    evidence: Dict[str, Any] = {
        "asr_top1_pinyin": input_block.get("asr_top1_pinyin", joined_pinyin(asr_top1)),
        "nbest_size": len(nbest),
    }
    if edits:
        primary = edits[0]
        from_norm = normalize_chinese_text(primary.from_text)
        to_norm = normalize_chinese_text(primary.to_text)
        evidence.update(
            {
                "decision_hint": "edit",
                "edit_type": _edit_type(from_norm, to_norm),
                "from_text": from_norm,
                "to_text": to_norm,
                "from_pinyin": joined_pinyin(from_norm),
                "to_pinyin": joined_pinyin(to_norm),
                "same_pinyin": same_pinyin(from_norm, to_norm),
                "pinyin_distance": _pinyin_distance(from_norm, to_norm),
                "from_support_count": _support_count(from_norm, support_texts),
                "to_support_count": _support_count(to_norm, support_texts),
                "to_in_nbest": _support_count(to_norm, nbest) > 0,
            }
        )
        output = {
            "decision": "edit",
            "edit_type": evidence["edit_type"],
            "evidence": {
                "same_pinyin": evidence["same_pinyin"],
                "pinyin_distance": evidence["pinyin_distance"],
                "to_in_nbest": evidence["to_in_nbest"],
                "to_support_count": evidence["to_support_count"],
            },
            "edits": [edit.to_json() for edit in edits],
        }
    else:
        traps = _same_pinyin_traps(asr_top1, nbest, trap_limit)
        evidence.update(
            {
                "decision_hint": "keep",
                "edit_type": "none",
                "same_pinyin_trap_count": len(traps),
                "same_pinyin_traps": traps,
            }
        )
        output = {
            "decision": "keep",
            "edit_type": "none",
            "evidence": {
                "same_pinyin_trap_count": len(traps),
            },
            "edits": [],
        }

    input_block["phonetic_evidence"] = evidence
    return {
        **record,
        "instruction": (
            "根据 ASR 输出、N-best 候选、拼音序列和 phonetic evidence 进行中文 ASR 后纠错。"
            "只做有声学/拼音/候选证据支持的最小修改；遇到同音但证据不足的候选时保持原文。"
        ),
        "input": input_block,
        "output": output,
    }


def main() -> int:
    args = parse_args()
    stats = {"records": 0, "edit": 0, "keep": 0, "same_pinyin_traps": 0}

    def records():
        for record in read_jsonl(args.input):
            converted = convert_record(record, trap_limit=int(args.trap_limit))
            stats["records"] += 1
            if converted.get("output", {}).get("decision") == "edit":
                stats["edit"] += 1
            else:
                stats["keep"] += 1
                stats["same_pinyin_traps"] += int(
                    converted.get("input", {})
                    .get("phonetic_evidence", {})
                    .get("same_pinyin_trap_count", 0)
                )
            yield converted

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
