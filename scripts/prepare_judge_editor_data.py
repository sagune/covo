#!/usr/bin/env python
"""Build Qwen-message data for candidate-edit judging.

The input is an editor prediction JSONL. Candidate edits are shown to the
model; the target contains only candidates that improve the ASR top1 against
the reference. This keeps inference one-pass while turning judgement into an
explicit supervised objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import safe_apply_position_edits
from covo.io import read_jsonl, write_jsonl
from covo.metrics import edit_distance
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--consensus", default="", help="Optional position-consensus JSONL to merge by id")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-empty-candidates", action="store_true")
    return parser.parse_args()


def _distance(text: str, reference: str) -> int:
    return edit_distance(list(normalize_chinese_text(text)), list(normalize_chinese_text(reference)))


def _load_consensus(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for record in read_jsonl(path):
        input_block = record.get("input", {}) or {}
        rows[str(record.get("id", ""))] = input_block
    return rows


def _candidate_lines(candidates: List[Dict[str, Any]]) -> List[str]:
    lines = ["Candidate edits:"]
    if not candidates:
        lines.append("- none")
        return lines
    for idx, edit in enumerate(candidates, 1):
        lines.append(
            "- "
            + json.dumps(
                {
                    "id": idx,
                    "start": edit.get("start"),
                    "end": edit.get("end"),
                    "from": edit.get("from", ""),
                    "to": edit.get("to", ""),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    return lines


def _format_user(input_block: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "请判断候选 ASR 位置编辑是否应被接受。只输出合法 JSON：{\"edits\":[...]}。",
        "输出的 edits 只能来自 Candidate edits；应拒绝边界错误、无证据替换和会让句子变差的候选。",
        "",
        f"ASR: {input_block.get('asr_top1', '')}",
    ]
    indexed = str(input_block.get("indexed_asr", "")).strip()
    if indexed:
        lines.append(f"Indexed ASR: {indexed}")
    nbest = list(input_block.get("nbest", []) or [])
    if nbest:
        lines.append("N-best:")
        for idx, hyp in enumerate(nbest, 1):
            lines.append(f"{idx}. {hyp}")
    nbest_consensus = input_block.get("nbest_consensus")
    if isinstance(nbest_consensus, dict):
        uncertain = list(nbest_consensus.get("uncertain_spans", []) or [])
        stable = list(nbest_consensus.get("stable_spans", []) or [])
        if stable:
            lines.append("Stable spans:")
            for item in stable:
                lines.append(
                    f"- {int(item.get('start', 0))}:{int(item.get('end', 0))} "
                    f"{item.get('text', '')} support={float(item.get('support', 0.0)):.3f}"
                )
        if uncertain:
            lines.append("Uncertain spans:")
            for item in uncertain:
                variants = item.get("variants", [])
                variants_text = ""
                if isinstance(variants, list) and variants:
                    variants_text = " variants=" + json.dumps(variants, ensure_ascii=False, separators=(",", ":"))
                lines.append(
                    f"- {int(item.get('start', 0))}:{int(item.get('end', 0))} "
                    f"{item.get('text', '')} support={float(item.get('support', 0.0)):.3f}{variants_text}"
                )
    lines.extend(_candidate_lines(candidates))
    return "\n".join(lines).strip()


def _accepted_candidates(record: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    input_block = record.get("input", {}) or {}
    asr = normalize_chinese_text(input_block.get("asr_top1", ""))
    reference = normalize_chinese_text(record.get("reference", ""))
    baseline_distance = _distance(asr, reference)
    accepted: List[Dict[str, Any]] = []
    for edit in candidates:
        result = safe_apply_position_edits(asr, {"edits": [edit]}, max_total_changed_chars=64)
        prediction = result["text"] if result.get("accepted") else asr
        if _distance(prediction, reference) < baseline_distance:
            accepted.append(
                {
                    "start": int(edit.get("start")),
                    "end": int(edit.get("end")),
                    "from": str(edit.get("from", "")),
                    "to": str(edit.get("to", "")),
                    "reason": "judge_accept",
                }
            )
    return accepted


def _convert(record: Dict[str, Any], consensus: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    input_block = dict(record.get("input", {}) or {})
    merged = consensus.get(str(record.get("id", "")))
    if merged and isinstance(merged.get("nbest_consensus"), dict):
        input_block["nbest_consensus"] = merged["nbest_consensus"]
    candidates = list((record.get("predicted_edits", {}) or {}).get("edits", []) or [])
    accepted = _accepted_candidates({**record, "input": input_block}, candidates)
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个中文 ASR 候选编辑审查器。必须只输出 JSON 对象，"
                "格式为 {\"edits\":[{\"start\":0,\"end\":1,\"from\":\"...\",\"to\":\"...\"}]}。"
            ),
        },
        {"role": "user", "content": _format_user(input_block, candidates)},
        {"role": "assistant", "content": json.dumps({"edits": accepted}, ensure_ascii=False, separators=(",", ":"))},
    ]
    return {
        "id": record.get("id", ""),
        "source": record.get("source", ""),
        "split": record.get("split", ""),
        "input": input_block,
        "reference": record.get("reference", ""),
        "candidate_edits": {"edits": candidates},
        "output": {"edits": accepted},
        "messages": messages,
    }


def main() -> int:
    args = parse_args()
    consensus = _load_consensus(args.consensus)
    stats = {
        "records": 0,
        "written": 0,
        "candidate_records": 0,
        "candidate_edits": 0,
        "accepted_records": 0,
        "accepted_edits": 0,
    }

    def records() -> Iterable[Dict[str, Any]]:
        for record in read_jsonl(args.predictions):
            stats["records"] += 1
            candidates = list((record.get("predicted_edits", {}) or {}).get("edits", []) or [])
            if candidates:
                stats["candidate_records"] += 1
                stats["candidate_edits"] += len(candidates)
            if not candidates and not args.include_empty_candidates:
                continue
            converted = _convert(record, consensus)
            if converted is None:
                continue
            accepted = list((converted.get("output", {}) or {}).get("edits", []) or [])
            if accepted:
                stats["accepted_records"] += 1
                stats["accepted_edits"] += len(accepted)
            stats["written"] += 1
            yield converted

    written = write_jsonl(args.output, records())
    print(json.dumps({"output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
