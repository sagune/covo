#!/usr/bin/env python
"""Inject compact domain-term KB evidence into COVO Qwen messages.

This is an inference-time prompt ablation.  It does not use references and does
not change labels; it only adds non-leak domain-term evidence mined from
CB-Whisper/KWS/candidate signals.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
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


def pinyin_distance(left: str, right: str) -> int:
    a = lazy_pinyin(left, style=Style.NORMAL, errors="ignore")
    b = lazy_pinyin(right, style=Style.NORMAL, errors="ignore")
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(x != y)))
        prev = cur
    return prev[-1]


def nested_get(record: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def load_kb(path: str | Path) -> Dict[str, Dict[str, Any]]:
    kb = {}
    for row in read_jsonl(path):
        term = norm(row.get("term", ""))
        if term:
            kb[term] = row
    return kb


def candidate_texts(record: Dict[str, Any]) -> List[str]:
    input_block = record.get("input", {}) or {}
    texts = [str(input_block.get("asr_top1", "") or "")]
    texts.extend(str(item or "") for item in input_block.get("nbest", []) or [])
    for cand in nested_get(input_block, "cbwhisper.candidates", []) or []:
        if isinstance(cand, dict):
            texts.append(str(cand.get("text", "") or ""))
    out = []
    seen = set()
    for text in texts:
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def iter_hotwords(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    input_block = record.get("input", {}) or {}
    rank = 0
    for item in input_block.get("hotwords", []) or []:
        if isinstance(item, dict):
            rank += 1
            yield {
                "term": norm(item.get("text", "")),
                "score": float(item.get("score", 0.0) or 0.0),
                "source": "kws",
                "rank": rank,
            }
    for item in input_block.get("prompt_hotwords", []) or []:
        if isinstance(item, dict):
            yield {
                "term": norm(item.get("text", "")),
                "score": float(item.get("weight", 0.0) or 0.0),
                "source": "prompt",
                "rank": 0,
            }
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            yield {
                "term": norm(item.get("mention", item.get("text", ""))),
                "score": 1.0,
                "source": "candidate_mention",
                "rank": 0,
            }


def exact_candidate_terms(record: Dict[str, Any]) -> set[str]:
    terms = set()
    for cand in nested_get(record.get("input", {}) or {}, "cbwhisper.candidates", []) or []:
        if not isinstance(cand, dict):
            continue
        for term in cand.get("consensus_keywords", []) or []:
            term_norm = norm(term)
            if term_norm:
                terms.add(term_norm)
        for term in nested_get(cand, "exact_stats.matched_keywords", []) or []:
            term_norm = norm(term)
            if term_norm:
                terms.add(term_norm)
    return terms


def retrieve_terms(record: Dict[str, Any], kb: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    texts = candidate_texts(record)
    text_blob = norm("".join(texts))
    asr_top1 = norm(nested_get(record, "input.asr_top1", ""))
    exact_terms = exact_candidate_terms(record)
    hotword_best: Dict[str, Dict[str, Any]] = {}
    for item in iter_hotwords(record):
        term = item["term"]
        if not term or term not in kb:
            continue
        old = hotword_best.get(term)
        if old is None or (item["source"] == "prompt", item["score"]) > (old["source"] == "prompt", old["score"]):
            hotword_best[term] = item

    rows = []
    for term, meta in kb.items():
        in_asr = bool(term and term in asr_top1)
        in_nbest = bool(term and term in text_blob)
        in_exact = term in exact_terms
        hot = hotword_best.get(term)
        hot_score = float(hot["score"]) if hot else 0.0
        is_prompt = bool(hot and hot["source"] == "prompt")
        if not (in_nbest or in_exact or is_prompt or hot_score >= float(args.kws_score_threshold)):
            continue
        if not args.include_absent_hotwords and not (in_nbest or in_exact or in_asr):
            continue
        if not in_nbest and not is_prompt and hot_score < float(args.absent_kws_score_threshold):
            continue

        best_py_distance = min((pinyin_distance(term, norm(text)) for text in texts if norm(text)), default=99)
        evidence = []
        if in_asr:
            evidence.append("in_asr_top1")
        if in_nbest:
            evidence.append("in_nbest")
        if in_exact:
            evidence.append("candidate_exact_or_consensus")
        if is_prompt:
            evidence.append("prompt_hotword")
        elif hot:
            evidence.append(f"kws_rank={hot['rank']}")
        if best_py_distance <= int(args.max_pinyin_distance):
            evidence.append(f"near_pinyin_dist={best_py_distance}")

        source_counts = meta.get("sources", {}) or {}
        support = int(meta.get("support", 0) or 0)
        priority = (
            int(in_exact) * 6
            + int(in_nbest) * 5
            + int(in_asr) * 3
            + int(is_prompt) * 2
            + min(hot_score, 1.0)
            + min(support / 1000.0, 2.0)
            - (0 if in_nbest else 1.5)
        )
        rows.append(
            {
                "term": term,
                "pinyin": str(meta.get("pinyin") or pinyin(term)),
                "support": support,
                "sources": source_counts,
                "hot_score": hot_score,
                "in_nbest": in_nbest,
                "in_asr": in_asr,
                "weak_absent": not in_nbest,
                "evidence": evidence,
                "priority": priority,
            }
        )
    rows.sort(key=lambda row: (-float(row["priority"]), -int(row["support"]), -len(row["term"]), row["term"]))
    return rows[: int(args.max_terms)]


def format_kb_evidence(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return (
            "Domain KB evidence (non-reference, retrieved from KWS/prompt/candidates):\n"
            "- none\n"
            "Rule: do not insert a domain term unless the utterance context or N-best candidates support it."
        )
    lines = [
        "Domain KB evidence (non-reference, retrieved from KWS/prompt/candidates):",
        "Use this as soft evidence for domain-term preservation/correction; do not force weak absent terms into the output.",
    ]
    for idx, row in enumerate(rows, start=1):
        warning = " weak_absent_do_not_force" if row["weak_absent"] else ""
        lines.append(
            f"{idx}. {row['term']} | pinyin={row['pinyin']} | support={row['support']} "
            f"| evidence={','.join(row['evidence'])}{warning}"
        )
    return "\n".join(lines)


def inject_record(record: Dict[str, Any], kb: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    messages = [dict(item) for item in record.get("messages", []) if isinstance(item, dict)]
    if not messages:
        return record
    user_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            user_idx = idx
            break
    if user_idx is None:
        return record
    rows = retrieve_terms(record, kb, args)
    evidence = format_kb_evidence(rows)
    content = str(messages[user_idx].get("content", ""))
    marker = "\n请输出 JSON："
    if marker in content:
        content = content.replace(marker, "\n" + evidence + marker, 1)
    else:
        content = content.rstrip() + "\n" + evidence
    messages[user_idx]["content"] = content
    return {
        **record,
        "input": {
            **(record.get("input", {}) or {}),
            "domain_kb_evidence": [
                {key: value for key, value in row.items() if key != "priority"}
                for row in rows
            ],
        },
        "messages": messages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--kws-score-threshold", type=float, default=0.78)
    parser.add_argument("--absent-kws-score-threshold", type=float, default=0.92)
    parser.add_argument("--max-pinyin-distance", type=int, default=1)
    parser.add_argument(
        "--include-absent-hotwords",
        action="store_true",
        help="Also include prompt/KWS terms that are absent from ASR/N-best/candidate text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kb = load_kb(args.kb)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    evidence_rows = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in read_jsonl(args.input):
            out = inject_record(record, kb, args)
            evidence_rows += len(nested_get(out, "input.domain_kb_evidence", []) or [])
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"input": args.input, "kb": args.kb, "output": args.output, "rows": count, "evidence_rows": evidence_rows}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
