#!/usr/bin/env python
"""Convert edit targets to position-aware compact edits."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-span-chars", type=int, default=16)
    return parser.parse_args()


def _single_edit_span(asr: str, ref: str) -> Tuple[int, int, int, int] | None:
    opcodes = [
        item
        for item in difflib.SequenceMatcher(a=asr, b=ref, autojunk=False).get_opcodes()
        if item[0] != "equal"
    ]
    if len(opcodes) != 1:
        return None
    _, i1, i2, j1, j2 = opcodes[0]
    return i1, i2, j1, j2


def _enclosing_edit_span(asr: str, ref: str) -> Tuple[int, int, int, int] | None:
    opcodes = [
        item
        for item in difflib.SequenceMatcher(a=asr, b=ref, autojunk=False).get_opcodes()
        if item[0] != "equal"
    ]
    if not opcodes:
        return None
    i1 = opcodes[0][1]
    i2 = opcodes[-1][2]
    j1 = opcodes[0][3]
    j2 = opcodes[-1][4]

    # Pure insertions have an empty ASR span. Anchor them with a neighboring
    # character so inference can still verify from == ASR[start:end].
    if i1 == i2:
        if i1 > 0 and j1 > 0:
            i1 -= 1
            j1 -= 1
        elif i2 < len(asr) and j2 < len(ref):
            i2 += 1
            j2 += 1
        else:
            return None
    return i1, i2, j1, j2


def _count_occurrences(text: str, span: str) -> int:
    if not span:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(span, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1


def _position_span(asr: str, ref: str, max_span_chars: int) -> Tuple[int, int, str, str] | None:
    span = _single_edit_span(asr, ref) or _enclosing_edit_span(asr, ref)
    if span is None:
        return None
    i1, i2, j1, j2 = span
    left = 0
    right = 0
    max_left = min(i1, j1)
    max_right = min(len(asr) - i2, len(ref) - j2)

    while True:
        start = i1 - left
        end = i2 + right
        from_text = asr[start:end]
        to_text = ref[j1 - left : j2 + right]
        if from_text and from_text != to_text and _count_occurrences(asr, from_text) == 1:
            return start, end, from_text, to_text
        can_left = left < max_left
        can_right = right < max_right
        if not can_left and not can_right:
            break
        if max(len(from_text), len(to_text)) >= int(max_span_chars):
            break
        if can_left and (left <= right or not can_right):
            left += 1
        elif can_right:
            right += 1

    start, end = i1, i2
    from_text = asr[start:end]
    to_text = ref[j1:j2]
    if not from_text or from_text == to_text:
        return None
    return start, end, from_text, to_text


def _indexed_asr(asr: str) -> str:
    return " ".join(f"{idx}:{char}" for idx, char in enumerate(asr))


def convert_record(record: Dict[str, Any], max_span_chars: int) -> Dict[str, Any]:
    record = deepcopy(record)
    input_block = dict(record.get("input", {}) or {})
    asr = normalize_chinese_text(input_block.get("asr_top1", ""))
    ref = normalize_chinese_text(record.get("reference", ""))
    output = dict(record.get("output", {}) or {})
    edits = list(output.get("edits", []) or [])

    input_block["indexed_asr"] = _indexed_asr(asr)
    input_block["position_edit_schema"] = {
        "start": "inclusive character index in ASR",
        "end": "exclusive character index in ASR",
    }
    record["input"] = input_block
    record["instruction"] = (
        "根据 ASR 输出、Indexed ASR、N-best 候选、拼音序列和 phonetic evidence 进行中文 ASR 后纠错。"
        "只输出位置感知 JSON edits；每个 edit 必须包含 start、end、from、to。"
        "start/end 指 ASR 字符位置，且 from 必须等于 ASR[start:end]；无需修改时输出空列表。"
    )

    if not edits:
        record["output"] = {"edits": []}
        return record

    positioned = _position_span(asr, ref, max_span_chars=max_span_chars)
    if positioned is None:
        record["output"] = {"edits": []}
        input_block["position_edit_skipped"] = True
        return record
    start, end, from_text, to_text = positioned
    record["output"] = {
        "edits": [
            {
                "start": start,
                "end": end,
                "from": from_text,
                "to": to_text,
                "reason": "position_span_replace",
            }
        ]
    }
    evidence = input_block.get("phonetic_evidence")
    if isinstance(evidence, dict):
        evidence["position_aware"] = True
        evidence["from_text"] = from_text
        evidence["to_text"] = to_text
    return record


def main() -> int:
    args = parse_args()
    stats = {
        "records": 0,
        "position_edits": 0,
        "keep": 0,
        "skipped_edits": 0,
        "single_to_span": 0,
    }

    def records():
        for record in read_jsonl(args.input):
            stats["records"] += 1
            old_edits = list(record.get("output", {}).get("edits", []) or [])
            converted = convert_record(record, int(args.max_span_chars))
            new_edits = list(converted.get("output", {}).get("edits", []) or [])
            if new_edits:
                stats["position_edits"] += 1
                old = old_edits[0] if old_edits else {}
                new = new_edits[0]
                if len(str(old.get("from", ""))) == len(str(old.get("to", ""))) == 1 and (
                    len(str(new.get("from", ""))) > 1 or len(str(new.get("to", ""))) > 1
                ):
                    stats["single_to_span"] += 1
            elif old_edits:
                stats["skipped_edits"] += 1
            else:
                stats["keep"] += 1
            yield converted

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
