#!/usr/bin/env python3
"""Build acoustic-aware listwise COVO records from SenseVoice evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM = (
    "你是一个保守的中文ASR后纠错器。候选声学分数越高，表示音频对该候选的支持越强；"
    "搜索分数包含上下文热词偏置，不能单独作为修改依据。综合声学证据、候选共识、拼音和热词，"
    "选择或生成纠错后的完整句子。证据不足时保持第一候选。"
    "必须只输出JSON对象，格式为{\"text\":\"完整句子\"}。"
)
_PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return _PUNCT_RE.sub("", text)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _candidate_rows(record: Dict[str, Any], max_nbest: int) -> List[Dict[str, Any]]:
    input_block = record.get("input", {}) or {}
    nbest = list(record.get("nbest", []) or input_block.get("nbest", []) or [])
    pinyin = list(record.get("nbest_pinyin", []) or input_block.get("nbest_pinyin", []) or [])
    cb = input_block.get("cbwhisper", {}) or {}
    cb_candidates = cb.get("candidates", []) or []
    by_text: Dict[str, Dict[str, Any]] = {}
    for item in cb_candidates:
        key = normalize_text(item.get("text", ""))
        if key and key not in by_text:
            by_text[key] = dict(item)

    aligned_scores = list(record.get("candidate_scores", []) or input_block.get("candidate_scores", []) or [])
    rows = []
    seen = set()
    for source_rank, value in enumerate(nbest, 1):
        text = str(value or "").strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        item = dict(by_text.get(key, {}))
        token_count = len(item.get("token_ids", []) or []) or max(len(key), 1)
        acoustic = _finite_float(
            item.get("asr_score", aligned_scores[source_rank - 1] if source_rank <= len(aligned_scores) else 0.0)
        )
        search = _finite_float(item.get("search_score", acoustic))
        rows.append(
            {
                "rank": len(rows) + 1,
                "text": text,
                "pinyin": str(pinyin[source_rank - 1]).strip() if source_rank <= len(pinyin) else "",
                "source": str(item.get("source", "sensevoice_ctc")),
                "token_count": int(token_count),
                "acoustic_score": acoustic,
                "acoustic_score_norm": acoustic / max(token_count, 1),
                "search_score": search,
                "search_score_norm": search / max(token_count, 1),
                "ctc_hotword_score": _finite_float(item.get("ctc_hotword_score", 0.0)),
                "kws_exact_score": _finite_float(item.get("exact_score", 0.0)),
                "kws_phonetic_score": _finite_float(item.get("phonetic_score", 0.0)),
                "consensus_support": int(_finite_float(item.get("consensus_support", 0))),
            }
        )
        if len(rows) >= max(1, int(max_nbest)):
            break

    acoustic_order = sorted(
        range(len(rows)),
        key=lambda idx: (rows[idx]["acoustic_score_norm"], -rows[idx]["rank"]),
        reverse=True,
    )
    for acoustic_rank, idx in enumerate(acoustic_order, 1):
        rows[idx]["acoustic_rank"] = acoustic_rank
    if rows:
        best = max(row["acoustic_score_norm"] for row in rows)
        for row in rows:
            row["acoustic_margin"] = row["acoustic_score_norm"] - best
    return rows


def _uncertain_spans(rows: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    if len(rows) < 2:
        return []
    base = normalize_text(rows[0]["text"])
    variants: Dict[tuple[int, int, str], Dict[str, Any]] = {}
    for row in rows[1:]:
        candidate = normalize_text(row["text"])
        matcher = SequenceMatcher(a=base, b=candidate, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            base_span = base[i1:i2]
            variant = candidate[j1:j2]
            key = (i1, i2, base_span)
            entry = variants.setdefault(
                key,
                {"start": i1, "end": i2, "top1": base_span, "variants": Counter()},
            )
            entry["variants"][variant] += 1

    output = []
    for entry in variants.values():
        values = [
            {"text": text, "support": support}
            for text, support in entry["variants"].most_common()
        ]
        output.append({**entry, "variants": values})
    output.sort(
        key=lambda item: (
            -sum(value["support"] for value in item["variants"]),
            item["start"],
            item["end"],
        )
    )
    return output[: max(0, int(limit))]


def _hotwords(input_block: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    values = []
    for item in input_block.get("hotwords", []) or []:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            score = _finite_float(item.get("score", 0.0))
        else:
            text = str(item).strip()
            score = 0.0
        if text:
            values.append({"text": text, "score": score})
    values.sort(key=lambda item: (-item["score"], -len(item["text"]), item["text"]))
    return values[: max(0, int(limit))]


def _target_index(reference: str, rows: List[Dict[str, Any]]) -> int:
    key = normalize_text(reference)
    for idx, row in enumerate(rows):
        if normalize_text(row["text"]) == key:
            return idx
    return -1


def build_record(
    record: Dict[str, Any],
    max_nbest: int,
    max_hotwords: int,
    require_reference_candidate: bool,
) -> Dict[str, Any] | None:
    reference = str(record.get("reference", "")).strip()
    rows = _candidate_rows(record, max_nbest)
    if not reference or not rows:
        return None
    target_index = _target_index(reference, rows)
    if require_reference_candidate and target_index < 0:
        return None

    input_block = record.get("input", {}) or {}
    hotwords = _hotwords(input_block, max_hotwords)
    uncertain = _uncertain_spans(rows)
    user_lines = ["候选及证据："]
    for row in rows:
        user_lines.append(
            f'{row["rank"]}. {row["text"]} | '
            f'声学排名={row["acoustic_rank"]}/{len(rows)} '
            f'声学分数={row["acoustic_score_norm"]:.4f} '
            f'相对最优={row["acoustic_margin"]:.4f} '
            f'搜索分数={row["search_score_norm"]:.4f} '
            f'热词偏置={row["ctc_hotword_score"]:.4f} '
            f'来源={row["source"]}'
        )
        if row["pinyin"]:
            user_lines.append(f'   拼音：{row["pinyin"]}')
    if uncertain:
        user_lines.append("候选分歧片段：")
        for item in uncertain:
            variants = " / ".join(
                f'{value["text"] or "∅"}({value["support"]})' for value in item["variants"]
            )
            user_lines.append(
                f'位置[{item["start"]}:{item["end"]}] {item["top1"] or "∅"} -> {variants}'
            )
    if hotwords:
        user_lines.append(
            "KWS热词（仅作上下文证据）："
            + "，".join(f'{item["text"]}({item["score"]:.3f})' for item in hotwords)
        )
    user_lines.append('请输出：{"text":"纠错后的完整句子"}')

    candidates = [row["text"] for row in rows]
    return {
        "id": str(record.get("id", "")),
        "source": str(record.get("source", "cb_sensevoice")),
        "dataset": str(record.get("dataset", "")),
        "split": str(record.get("split", "")),
        "reference": reference,
        "asr_top1": candidates[0],
        "target_index": target_index,
        "candidates": candidates,
        "candidate_evidence": rows,
        "uncertain_spans": uncertain,
        "input": {
            "asr_top1": candidates[0],
            "nbest": candidates,
            "nbest_pinyin": [row["pinyin"] for row in rows],
            "candidate_evidence": rows,
            "hotwords": hotwords,
        },
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(user_lines)},
            {
                "role": "assistant",
                "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def read_jsonl(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--max-hotwords", type=int, default=8)
    parser.add_argument("--allow-reference-outside-nbest", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = written = skipped = reference_in_nbest = 0
    with args.output.open("w", encoding="utf-8") as writer:
        for source in read_jsonl(args.input):
            seen += 1
            record = build_record(
                source,
                max_nbest=args.max_nbest,
                max_hotwords=args.max_hotwords,
                require_reference_candidate=not args.allow_reference_outside_nbest,
            )
            if record is None:
                skipped += 1
                continue
            reference_in_nbest += int(record["target_index"] >= 0)
            writer.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            if args.limit and written >= int(args.limit):
                break
    print(
        json.dumps(
            {
                "input": [str(path) for path in args.input],
                "output": str(args.output),
                "seen": seen,
                "written": written,
                "skipped": skipped,
                "reference_in_nbest": reference_in_nbest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
