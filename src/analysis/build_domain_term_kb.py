#!/usr/bin/env python
"""Build a deployable domain-term knowledge base from CB-Whisper evidence.

By default this script avoids label leakage: terms are collected only from
KWS/prompt/candidate evidence fields, not from references.  References can be
used only for diagnostic confusable mining with --include-reference.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pypinyin import Style, lazy_pinyin


DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.text import normalize_chinese_text  # type: ignore  # noqa: E402


PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return normalize_chinese_text(PUNCT_RE.sub("", text)).strip()


def pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))


def pinyin_initials(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.FIRST_LETTER, errors="ignore"))


def nested_get(record: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def iter_texts(record: Dict[str, Any]) -> Iterable[str]:
    input_block = record.get("input", {}) or {}
    yield str(input_block.get("asr_top1", "") or "")
    for item in input_block.get("nbest", []) or []:
        yield str(item or "")
    for cand in nested_get(input_block, "cbwhisper.candidates", []) or []:
        if isinstance(cand, dict):
            yield str(cand.get("text", "") or "")
    if "prediction" in record:
        yield str(record.get("prediction", "") or "")


def add_term(stats: Dict[str, Any], term: str, source: str, score: float | None = None) -> None:
    term = norm(term)
    if not term:
        return
    stats["sources"][source] += 1
    stats["support"] += 1
    if score is not None:
        stats["score_sum"] += float(score)
        stats["score_count"] += 1


def collect_terms(record: Dict[str, Any], kb: Dict[str, Dict[str, Any]], include_reference: bool) -> None:
    input_block = record.get("input", {}) or {}
    for item in input_block.get("hotwords", []) or []:
        if isinstance(item, dict):
            term = norm(item.get("text", ""))
            if term:
                add_term(kb[term], term, "kws_hotwords", item.get("score"))
    for item in input_block.get("prompt_hotwords", []) or []:
        if isinstance(item, dict):
            term = norm(item.get("text", ""))
            if term:
                add_term(kb[term], term, "prompt_hotwords", item.get("weight"))
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            term = norm(item.get("mention", item.get("text", "")))
            if term:
                add_term(kb[term], term, "keyword_mentions")
    for item in input_block.get("oracle_hotwords", []) or []:
        if isinstance(item, dict):
            term = norm(item.get("text", item.get("mention", "")))
        else:
            term = norm(item)
        if term:
            add_term(kb[term], term, "oracle_hotwords")
    for cand in nested_get(input_block, "cbwhisper.candidates", []) or []:
        if not isinstance(cand, dict):
            continue
        for field in ("consensus_keywords",):
            for term in cand.get(field, []) or []:
                add_term(kb[norm(term)], str(term), field)
        for term in nested_get(cand, "exact_stats.matched_keywords", []) or []:
            add_term(kb[norm(term)], str(term), "candidate_exact_match")
    if include_reference:
        # Diagnostic only: reference-derived terms leak labels for held-out eval.
        for item in input_block.get("hotwords", []) or []:
            if isinstance(item, dict):
                term = norm(item.get("text", ""))
                if term and term in norm(record.get("reference", "")):
                    add_term(kb[term], term, "reference_hit")


def mine_confusables(record: Dict[str, Any], terms: Iterable[str], max_span_len: int) -> Dict[str, Counter[str]]:
    ref = norm(record.get("reference", ""))
    conf: Dict[str, Counter[str]] = defaultdict(Counter)
    if not ref:
        return conf
    relevant = [term for term in terms if term and term in ref]
    if not relevant:
        return conf
    for text in iter_texts(record):
        cand = norm(text)
        if not cand or cand == ref:
            continue
        matcher = difflib.SequenceMatcher(a=ref, b=cand, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            ref_span = ref[i1:i2]
            cand_span = cand[j1:j2]
            if not ref_span or not cand_span:
                continue
            if len(ref_span) > max_span_len or len(cand_span) > max_span_len:
                continue
            for term in relevant:
                if ref_span in term or term in ref_span:
                    conf[term][cand_span] += 1
    return conf


def build_kb(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    kb: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "support": 0,
            "score_sum": 0.0,
            "score_count": 0,
            "sources": Counter(),
            "hit_in_asr_top1": 0,
            "hit_in_nbest": 0,
            "hit_in_prediction": 0,
            "confusables": Counter(),
        }
    )
    rows = 0
    records: List[Dict[str, Any]] = []
    for path in args.input:
        for record in read_jsonl(path):
            rows += 1
            records.append(record)
            collect_terms(record, kb, include_reference=bool(args.include_reference))

    terms = [term for term in kb if len(term) >= int(args.min_term_len)]
    for record in records:
        input_block = record.get("input", {}) or {}
        asr_top1 = norm(input_block.get("asr_top1", ""))
        nbest_text = norm("".join(str(x or "") for x in input_block.get("nbest", []) or []))
        pred = norm(record.get("prediction", ""))
        for term in terms:
            if term in asr_top1:
                kb[term]["hit_in_asr_top1"] += 1
            if term in nbest_text:
                kb[term]["hit_in_nbest"] += 1
            if pred and term in pred:
                kb[term]["hit_in_prediction"] += 1
        if args.include_reference:
            conf = mine_confusables(record, terms, max_span_len=int(args.max_confusable_span_len))
            for term, counter in conf.items():
                kb[term]["confusables"].update(counter)

    entries: List[Dict[str, Any]] = []
    for term in terms:
        item = kb[term]
        source_counts = dict(sorted(item["sources"].items(), key=lambda x: (-x[1], x[0])))
        score_count = int(item["score_count"])
        entries.append(
            {
                "term": term,
                "pinyin": pinyin(term),
                "pinyin_initials": pinyin_initials(term),
                "length": len(term),
                "support": int(item["support"]),
                "sources": source_counts,
                "avg_score_or_weight": float(item["score_sum"] / score_count) if score_count else None,
                "hit_in_asr_top1": int(item["hit_in_asr_top1"]),
                "hit_in_nbest": int(item["hit_in_nbest"]),
                "hit_in_prediction": int(item["hit_in_prediction"]),
                "confusables": [
                    {
                        "text": text,
                        "pinyin": pinyin(text),
                        "count": count,
                    }
                    for text, count in item["confusables"].most_common(int(args.max_confusables))
                ],
            }
        )
    entries.sort(key=lambda row: (-row["support"], -row["length"], row["term"]))
    summary = {
        "inputs": args.input,
        "rows": rows,
        "terms": len(entries),
        "include_reference": bool(args.include_reference),
        "top_terms": entries[: min(30, len(entries))],
    }
    return entries, summary


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--min-term-len", type=int, default=2)
    parser.add_argument("--max-confusable-span-len", type=int, default=6)
    parser.add_argument("--max-confusables", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries, summary = build_kb(args)
    written = write_jsonl(args.output, entries)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(
        json.dumps({**summary, "written": written}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "output": args.output, "summary_output": args.summary_output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
