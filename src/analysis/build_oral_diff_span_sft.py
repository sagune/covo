#!/usr/bin/env python
"""Build focused oral local-edit SFT data from ASR/N-best/reference rows.

This is for conversational corpora without explicit hotwords.  It extracts
candidate-vs-reference local edits:

- missing reference spans that should be inserted;
- redundant candidate spans that should be deleted;
- short replacement spans.

Each output row focuses on one local edit but keeps the original N-best context,
so it can be mixed with phonetic/hotword evidence data.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if COVO_SRC not in sys.path:
    sys.path.insert(0, COVO_SRC)

from covo.text import normalize_chinese_text  # type: ignore  # noqa: E402

from cbwhisper_covo_bridge import CONTENT_SELECTOR_SYSTEM_MESSAGE  # noqa: E402


PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def norm(text: Any) -> str:
    return normalize_chinese_text(PUNCT_RE.sub("", str(text or "")))


def text_list(value: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def collect_edits(record: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    reference = norm(record.get("reference", ""))
    input_block = dict(record.get("input", {}) or {})
    nbest = text_list(input_block.get("nbest"))
    if not nbest and str(input_block.get("asr_top1", "") or "").strip():
        nbest = [str(input_block.get("asr_top1", "") or "")]
    edits: List[Dict[str, Any]] = []
    seen = set()
    for rank, candidate in enumerate(nbest[: int(args.max_candidates)], start=1):
        cand = norm(candidate)
        if not reference or not cand or cand == reference:
            continue
        ratio = len(cand) / max(len(reference), 1)
        if ratio < float(args.min_candidate_ref_ratio) or ratio > float(args.max_candidate_ref_ratio):
            continue
        matcher = difflib.SequenceMatcher(a=reference, b=cand, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            ref_span = reference[i1:i2]
            cand_span = cand[j1:j2]
            if tag == "delete":
                edit_type = "insert_missing"
            elif tag == "insert":
                edit_type = "delete_redundant"
            elif tag == "replace":
                edit_type = "replace_local"
            else:
                continue
            if len(ref_span) > int(args.max_ref_span_len) or len(cand_span) > int(args.max_candidate_span_len):
                continue
            if tag == "replace" and (len(ref_span) < int(args.min_replace_len) or len(cand_span) < int(args.min_replace_len)):
                continue
            if tag == "delete" and len(ref_span) < int(args.min_insert_len):
                continue
            if tag == "insert" and len(cand_span) < int(args.min_delete_len):
                continue
            key = (edit_type, ref_span, cand_span)
            if key in seen:
                continue
            seen.add(key)
            edits.append(
                {
                    "edit_type": edit_type,
                    "rank": rank,
                    "candidate_span": cand_span,
                    "reference_span": ref_span,
                    "candidate_range": [j1, j2],
                    "reference_range": [i1, i2],
                }
            )
    edits.sort(
        key=lambda item: (
            {"replace_local": 0, "insert_missing": 1, "delete_redundant": 2}.get(item["edit_type"], 9),
            item["rank"],
            -(len(item["reference_span"]) + len(item["candidate_span"])),
        )
    )
    return edits[: int(args.max_edits_per_record)]


def format_evidence(edit: Dict[str, Any]) -> str:
    if edit["edit_type"] == "insert_missing":
        action = "candidate 缺少 reference_span，应在对应位置补入该片段。"
    elif edit["edit_type"] == "delete_redundant":
        action = "candidate 多出 candidate_span，应删除该冗余片段。"
    else:
        action = "candidate_span 与 reference_span 局部不一致，应替换为 reference_span。"
    return "\n".join(
        [
            "Oral local edit evidence:",
            "下面给出一个口语 ASR 局部错误证据。只在该局部证据明确时修改；不要为了流畅而重写整句。",
            f"- type={edit['edit_type']} cand#{edit['rank']} {action}",
            f"  candidate_span={edit['candidate_span'] or '<empty>'}",
            f"  reference_span={edit['reference_span'] or '<empty>'}",
            f"  candidate_range={edit['candidate_range']} reference_range={edit['reference_range']}",
        ]
    )


def make_messages(record: Dict[str, Any], edit: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, str]]:
    reference = str(record.get("reference", "") or "")
    input_block = dict(record.get("input", {}) or {})
    nbest = text_list(input_block.get("nbest"))
    if not nbest and str(input_block.get("asr_top1", "") or "").strip():
        nbest = [str(input_block.get("asr_top1", "") or "")]
    lines = [
        "任务：根据 ASR top-1、N-best 候选和口语局部编辑证据，输出完整中文转写。",
        "规则：口语填充词、重复词、漏词和短局部替换可以修改；但不要无证据地改写主体内容。优先选择 N-best 中被局部证据支持的完整句子，必要时只做小幅局部修正。",
        '必须只输出 JSON：{"text":"纠错后的完整句子"}',
        f"ASR top-1: {str(input_block.get('asr_top1', '') or '')}",
        "N-best candidates:",
    ]
    for idx, cand in enumerate(nbest[: int(args.max_candidates)], start=1):
        lines.append(f"{idx}. {cand}")
    lines.append(format_evidence(edit))
    return [
        {"role": "system", "content": CONTENT_SELECTOR_SYSTEM_MESSAGE},
        {"role": "user", "content": "\n".join(lines)},
        {"role": "assistant", "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":"))},
    ]


def build_rows(records: Iterable[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    for idx, record in enumerate(records, start=1):
        if args.limit and idx > int(args.limit):
            break
        reference = str(record.get("reference", "") or "").strip()
        if not reference:
            continue
        for edit_idx, edit in enumerate(collect_edits(record, args), start=1):
            yield {
                **record,
                "id": f"{record.get('id', '')}#oral{edit_idx}",
                "oral_local_edit_evidence": edit,
                "messages": make_messages(record, edit, args),
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--train-output", default="")
    parser.add_argument("--dev-output", default="")
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=706)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--min-candidate-ref-ratio", type=float, default=0.65)
    parser.add_argument("--max-candidate-ref-ratio", type=float, default=1.35)
    parser.add_argument("--max-edits-per-record", type=int, default=3)
    parser.add_argument("--max-ref-span-len", type=int, default=8)
    parser.add_argument("--max-candidate-span-len", type=int, default=8)
    parser.add_argument("--min-replace-len", type=int, default=2)
    parser.add_argument("--min-insert-len", type=int, default=1)
    parser.add_argument("--min-delete-len", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = list(build_rows(read_jsonl(args.input), args))
    if args.train_output and args.dev_output:
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
        dev_size = max(0, min(int(args.dev_size), len(rows)))
        dev = rows[:dev_size]
        train = rows[dev_size:]
        train_written = write_jsonl(args.train_output, train)
        dev_written = write_jsonl(args.dev_output, dev)
        result = {
            "input": args.input,
            "train_output": args.train_output,
            "dev_output": args.dev_output,
            "train_written": train_written,
            "dev_written": dev_written,
            "written": train_written + dev_written,
        }
    elif args.output:
        written = write_jsonl(args.output, rows)
        result = {"input": args.input, "output": args.output, "written": written}
    else:
        raise ValueError("provide --output or both --train-output/--dev-output")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
