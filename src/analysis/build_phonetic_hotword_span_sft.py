#!/usr/bin/env python
"""Build phonetic hotword span SFT data for local ASR correction.

This converts CB-Whisper/COVO evidence rows into Qwen-message SFT rows with
explicit pinyin-based term localization:

- protected/domain terms and their pinyin;
- candidate spans with similar pinyin that may be ASR confusions;
- candidate ranks that exactly keep or drop each term.

The goal is to teach the model to copy protected professional terms only where
the local phonetic evidence supports the replacement, instead of freely rewriting
the whole sentence.
"""

from __future__ import annotations

import argparse
import difflib
import functools
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if COVO_SRC not in sys.path:
    sys.path.insert(0, COVO_SRC)

from covo.metrics import edit_distance  # type: ignore  # noqa: E402
from covo.text import joined_pinyin, normalize_chinese_text, to_pinyin_units  # type: ignore  # noqa: E402

from cbwhisper_covo_bridge import CONTENT_SELECTOR_SYSTEM_MESSAGE, build_user_prompt  # noqa: E402


DEFAULT_DOMAIN_TERMS = [
    "水利工程",
    "水利工程施工",
    "水资源",
    "地基处理",
    "地基",
    "水闸",
    "闸门",
    "灌浆",
    "机电设备",
    "挡水建筑物",
    "挡水建筑",
    "建筑物",
    "施工组织设计",
    "施工组织",
    "组织施工",
    "施工管理",
    "施工图",
    "施工",
    "土石坝",
    "土石方",
    "土石",
    "石方",
    "开挖",
    "填筑",
    "基坑",
    "围堰",
    "导流",
    "坝体",
    "大坝",
    "水库",
    "构筑物",
    "浇筑",
    "运输",
]

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


def stable_key(record: Dict[str, Any]) -> str:
    return f"{record.get('split','')}:{record.get('id','')}:{record.get('reference','')}"


def stable_rng(record: Dict[str, Any], seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stable_key(record)}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def norm(text: Any) -> str:
    return normalize_chinese_text(PUNCT_RE.sub("", str(text or "")))


def as_text_list(value: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, str):
            text = norm(item)
        elif isinstance(item, dict):
            text = norm(item.get("text") or item.get("mention") or item.get("keyword"))
        else:
            text = ""
        if text and text not in out:
            out.append(text)
    return out


def collect_terms(input_block: Dict[str, Any], reference: str, nbest: List[str], domain_terms: List[str], max_terms: int) -> List[str]:
    ref_norm = norm(reference)
    terms: List[str] = []

    for source in (
        as_text_list(input_block.get("protected_hotwords")),
        as_text_list(input_block.get("prompt_hotwords")),
        as_text_list(input_block.get("keyword_mentions")),
        as_text_list(input_block.get("hotwords")),
        as_text_list(input_block.get("kws_hotwords")),
    ):
        for term in source:
            if len(term) >= 2 and term in ref_norm and term not in terms:
                terms.append(term)

    for term in domain_terms:
        term = norm(term)
        if len(term) >= 2 and term in ref_norm and term not in terms:
            terms.append(term)

    return sorted(terms, key=lambda item: (-len(item), item))[:max_terms]


def reference_ngram_terms(reference: str, min_len: int, max_len: int, max_terms: int) -> List[str]:
    ref_norm = norm(reference)
    terms: List[str] = []
    if not ref_norm:
        return terms
    min_len = max(2, int(min_len))
    max_len = max(min_len, int(max_len))
    for size in range(max_len, min_len - 1, -1):
        if size > len(ref_norm):
            continue
        for start in range(0, len(ref_norm) - size + 1):
            term = ref_norm[start : start + size]
            if term and term not in terms:
                terms.append(term)
            if len(terms) >= max_terms:
                return terms
    return terms


def pinyin_distance(left: str, right: str) -> int:
    return _pinyin_distance_cached(norm(left), norm(right))


@functools.lru_cache(maxsize=300000)
def _pinyin_units_cached(text: str) -> Tuple[str, ...]:
    return tuple(to_pinyin_units(text))


@functools.lru_cache(maxsize=500000)
def _pinyin_distance_cached(left: str, right: str) -> int:
    return int(edit_distance(list(_pinyin_units_cached(left)), list(_pinyin_units_cached(right))))


def pinyin_initials(text: str) -> str:
    return "".join(unit[:1] for unit in _pinyin_units_cached(norm(text)) if unit)


def best_phonetic_spans(
    term: str,
    candidate: str,
    max_distance: int,
    max_spans: int,
    min_span_len: int = 2,
    max_len_gap: int = 1,
    allow_same_pinyin: bool = True,
) -> List[Dict[str, Any]]:
    cand = norm(candidate)
    term_norm = norm(term)
    if not cand or not term_norm or term_norm in cand:
        return []
    term_len = len(term_norm)
    term_initials = pinyin_initials(term_norm)
    ranked: List[Tuple[int, int, str, int, int]] = []
    for win_len in range(max(1, term_len - 1), term_len + 2):
        if win_len < int(min_span_len):
            continue
        if abs(win_len - term_len) > int(max_len_gap):
            continue
        if win_len > len(cand):
            continue
        for start in range(0, len(cand) - win_len + 1):
            span = cand[start : start + win_len]
            if span == term_norm:
                continue
            # Cheap prefilter before full pinyin edit distance.
            span_initials = pinyin_initials(span)
            if term_initials and span_initials and edit_distance(list(term_initials), list(span_initials)) > max_distance:
                continue
            dist = pinyin_distance(term_norm, span)
            if dist == 0 and not bool(allow_same_pinyin):
                continue
            char_gap = abs(len(span) - term_len)
            if dist <= max_distance:
                ranked.append((dist, char_gap, span, start, start + win_len))
    ranked.sort(key=lambda item: (item[0], item[1], item[3]))
    out: List[Dict[str, Any]] = []
    seen = set()
    for dist, _gap, span, start, end in ranked:
        key = (span, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "span": span,
                "span_pinyin": joined_pinyin(span),
                "start": start,
                "end": end,
                "pinyin_distance": int(dist),
                "suggested_replacement": term_norm,
            }
        )
        if len(out) >= max_spans:
            break
    return out


def build_phonetic_evidence(
    input_block: Dict[str, Any],
    reference: str,
    domain_terms: List[str],
    max_terms: int,
    max_candidates: int,
    max_spans_per_term: int,
    max_distance: int,
    auto_reference_terms: bool = False,
    auto_max_terms: int = 64,
    min_term_len: int = 2,
    max_term_len: int = 4,
    min_span_len: int = 2,
    max_len_gap: int = 1,
    allow_same_pinyin: bool = True,
    require_confusion: bool = False,
) -> List[Dict[str, Any]]:
    nbest = [str(item or "") for item in list(input_block.get("nbest", []) or []) if str(item or "").strip()]
    if not nbest:
        top1 = str(input_block.get("asr_top1", "") or "")
        if top1:
            nbest = [top1]
    terms = collect_terms(input_block, reference, nbest, domain_terms, max_terms=max_terms)
    if auto_reference_terms:
        for term in reference_ngram_terms(
            reference,
            min_len=int(min_term_len),
            max_len=int(max_term_len),
            max_terms=int(auto_max_terms),
        ):
            if term not in terms:
                terms.append(term)
    evidence: List[Dict[str, Any]] = []
    for term in terms:
        term_norm = norm(term)
        item: Dict[str, Any] = {
            "term": term_norm,
            "term_pinyin": joined_pinyin(term_norm),
            "in_reference": term_norm in norm(reference),
            "candidate_support": [],
            "phonetic_confusions": [],
        }
        for rank, cand in enumerate(nbest[:max_candidates], start=1):
            cand_norm = norm(cand)
            keeps = term_norm in cand_norm
            support = {"rank": rank, "keeps_term": keeps}
            if keeps:
                support["span"] = term_norm
            item["candidate_support"].append(support)
            spans = best_phonetic_spans(
                term_norm,
                cand_norm,
                max_distance=max_distance,
                max_spans=max_spans_per_term,
                min_span_len=int(min_span_len),
                max_len_gap=int(max_len_gap),
                allow_same_pinyin=bool(allow_same_pinyin),
            )
            for span in spans:
                item["phonetic_confusions"].append({"rank": rank, **span})
        if require_confusion and not item["phonetic_confusions"]:
            continue
        if item["in_reference"] or item["phonetic_confusions"] or any(s["keeps_term"] for s in item["candidate_support"]):
            evidence.append(item)
        if len(evidence) >= int(max_terms):
            break
    return evidence


def build_diff_phonetic_evidence(
    input_block: Dict[str, Any],
    reference: str,
    max_terms: int,
    max_candidates: int,
    max_spans_per_term: int,
    max_distance: int,
    min_term_len: int = 2,
    max_term_len: int = 4,
    min_span_len: int = 2,
    max_len_gap: int = 1,
    allow_same_pinyin: bool = True,
) -> List[Dict[str, Any]]:
    ref_norm = norm(reference)
    nbest = [str(item or "") for item in list(input_block.get("nbest", []) or []) if str(item or "").strip()]
    if not nbest and str(input_block.get("asr_top1", "") or "").strip():
        nbest = [str(input_block.get("asr_top1", "") or "")]
    by_term: Dict[str, Dict[str, Any]] = {}

    for rank, candidate in enumerate(nbest[: int(max_candidates)], start=1):
        cand_norm = norm(candidate)
        if not ref_norm or not cand_norm or cand_norm == ref_norm:
            continue
        matcher = difflib.SequenceMatcher(a=ref_norm, b=cand_norm, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "replace":
                continue
            term = ref_norm[i1:i2]
            span = cand_norm[j1:j2]
            if len(term) < int(min_term_len) or len(term) > int(max_term_len):
                continue
            if len(span) < int(min_span_len):
                continue
            if abs(len(term) - len(span)) > int(max_len_gap):
                continue
            if term == span:
                continue
            dist = pinyin_distance(term, span)
            if dist == 0 and not bool(allow_same_pinyin):
                continue
            if dist > int(max_distance):
                continue
            item = by_term.setdefault(
                term,
                {
                    "term": term,
                    "term_pinyin": joined_pinyin(term),
                    "in_reference": True,
                    "candidate_support": [],
                    "phonetic_confusions": [],
                },
            )
            seen = {
                (conf.get("rank"), conf.get("span"), conf.get("start"), conf.get("end"))
                for conf in item["phonetic_confusions"]
            }
            key = (rank, span, j1, j2)
            if key in seen:
                continue
            if len(item["phonetic_confusions"]) < int(max_spans_per_term) * int(max_candidates):
                item["phonetic_confusions"].append(
                    {
                        "rank": rank,
                        "span": span,
                        "span_pinyin": joined_pinyin(span),
                        "start": j1,
                        "end": j2,
                        "pinyin_distance": int(dist),
                        "suggested_replacement": term,
                    }
                )

    evidence = list(by_term.values())
    evidence.sort(key=lambda item: (-len(item.get("phonetic_confusions", [])), -len(item["term"]), item["term"]))
    evidence = evidence[: int(max_terms)]
    for item in evidence:
        term = item["term"]
        supports = []
        for rank, candidate in enumerate(nbest[: int(max_candidates)], start=1):
            cand_norm = norm(candidate)
            keeps = term in cand_norm
            support = {"rank": rank, "keeps_term": keeps}
            if keeps:
                support["span"] = term
            supports.append(support)
        item["candidate_support"] = supports
        item["phonetic_confusions"] = item["phonetic_confusions"][: max(1, int(max_spans_per_term)) * 3]
    return evidence


def format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return ""
    lines = [
        "Phonetic hotword span evidence:",
        "下面给出专业术语的拼音定位证据。若某个候选局部 span 与 protected term 拼音接近，且该 term 在参考/上下文/候选证据中受支持，应优先把该局部 span 替换为术语；不要改动术语之外的稳定片段。",
    ]
    for idx, item in enumerate(evidence, start=1):
        lines.append(
            f"{idx}. term={item['term']} | pinyin={item['term_pinyin']} | in_reference={str(item['in_reference']).lower()}"
        )
        supports = []
        for support in item.get("candidate_support", [])[:10]:
            label = "keeps" if support.get("keeps_term") else "drops"
            supports.append(f"#{support.get('rank')}:{label}")
        if supports:
            lines.append(f"   candidate_support: {' '.join(supports)}")
        confusions = item.get("phonetic_confusions", [])[:6]
        if confusions:
            lines.append("   phonetic_confusions:")
            for conf in confusions:
                lines.append(
                    "   - "
                    f"cand#{conf['rank']} span[{conf['start']}:{conf['end']}]={conf['span']} "
                    f"pinyin={conf['span_pinyin']} dist={conf['pinyin_distance']} "
                    f"=> copy_term={conf['suggested_replacement']}"
                )
    return "\n".join(lines)


def append_to_user_message(messages: List[Dict[str, Any]], evidence_text: str) -> List[Dict[str, Any]]:
    out = [dict(item) for item in messages]
    if not evidence_text:
        return out
    for idx, message in enumerate(out):
        if message.get("role") == "user":
            content = str(message.get("content", ""))
            marker = '请输出 JSON：{"text":"纠错后的完整句子"}'
            if marker in content:
                content = content.replace(marker, evidence_text + "\n" + marker)
            else:
                content = content.rstrip() + "\n" + evidence_text
            out[idx] = {**message, "content": content}
            return out
    out.insert(0, {"role": "user", "content": evidence_text})
    return out


def make_messages(record: Dict[str, Any], input_block: Dict[str, Any], evidence_text: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    messages = list(record.get("messages", []) or [])
    if messages:
        return append_to_user_message(messages, evidence_text)

    if bool(args.compact_prompt):
        nbest = [str(item or "").strip() for item in list(input_block.get("nbest", []) or []) if str(item or "").strip()]
        if not nbest and str(input_block.get("asr_top1", "") or "").strip():
            nbest = [str(input_block.get("asr_top1", "") or "").strip()]
        protected = as_text_list(input_block.get("protected_hotwords"))
        prompt_hotwords = as_text_list(input_block.get("prompt_hotwords"))
        kws_hotwords = as_text_list(input_block.get("kws_hotwords"))[: int(args.max_terms)]
        lines = [
            "任务：根据 ASR top-1、N-best 候选、热词和音素证据，输出主体内容最可信的完整中文转写。",
            "规则：优先保留稳定主体内容；若某个局部候选 span 与真实领域词拼音接近，且证据提示 copy_term，应只替换该局部 span；不要因为热词误报而插入无上下文支持的词；不要为了口语填充词破坏主体内容。",
            '必须只输出 JSON：{"text":"纠错后的完整句子"}',
            f"ASR top-1: {str(input_block.get('asr_top1', '') or '')}",
        ]
        if protected:
            lines.append("protected_hotwords: " + "、".join(protected))
        if prompt_hotwords:
            lines.append("prompt_hotwords: " + "、".join(prompt_hotwords[: int(args.max_terms)]))
        if kws_hotwords:
            lines.append("kws_hotwords: " + "、".join(kws_hotwords))
        lines.append("N-best candidates:")
        for idx, cand in enumerate(nbest[: int(args.max_candidates)], start=1):
            lines.append(f"{idx}. {cand}")
        if evidence_text:
            lines.append(evidence_text)
        return [
            {"role": "system", "content": CONTENT_SELECTOR_SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join(lines)},
            {
                "role": "assistant",
                "content": json.dumps({"text": str(record.get("reference", ""))}, ensure_ascii=False, separators=(",", ":")),
            },
        ]

    bridge_args = argparse.Namespace(
        max_nbest=args.max_nbest,
        max_pinyin=args.max_pinyin,
        include_pinyin=True,
        max_hotwords=args.max_terms,
        max_prompt_hotwords=args.max_terms,
        max_candidates_with_scores=10,
        hotword_source="all",
        prompt_mode="selector_content",
        protect_supported_hotwords=True,
        clean_nbest=False,
        include_consensus_spans=True,
        include_hotword_evidence=True,
        max_confusables=4,
    )
    user = build_user_prompt({**record, "input": input_block}, bridge_args)
    if evidence_text:
        user = user.rstrip() + "\n" + evidence_text
    return [
        {"role": "system", "content": CONTENT_SELECTOR_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": json.dumps({"text": str(record.get("reference", ""))}, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def augment_record(record: Dict[str, Any], args: argparse.Namespace, domain_terms: List[str]) -> Dict[str, Any] | None:
    reference = str(record.get("reference", "") or "").strip()
    if not reference:
        return None
    input_block = dict(record.get("input", {}) or {})
    if bool(args.diff_reference_terms):
        evidence = build_diff_phonetic_evidence(
            input_block=input_block,
            reference=reference,
            max_terms=int(args.max_terms),
            max_candidates=int(args.max_candidates),
            max_spans_per_term=int(args.max_spans_per_term),
            max_distance=int(args.max_pinyin_distance),
            min_term_len=int(args.min_term_len),
            max_term_len=int(args.max_term_len),
            min_span_len=int(args.min_span_len),
            max_len_gap=int(args.max_len_gap),
            allow_same_pinyin=not bool(args.exclude_same_pinyin),
        )
    else:
        evidence = build_phonetic_evidence(
            input_block=input_block,
            reference=reference,
            domain_terms=domain_terms,
            max_terms=int(args.max_terms),
            max_candidates=int(args.max_candidates),
            max_spans_per_term=int(args.max_spans_per_term),
            max_distance=int(args.max_pinyin_distance),
            auto_reference_terms=bool(args.auto_reference_terms),
            auto_max_terms=int(args.auto_max_terms),
            min_term_len=int(args.min_term_len),
            max_term_len=int(args.max_term_len),
            min_span_len=int(args.min_span_len),
            max_len_gap=int(args.max_len_gap),
            allow_same_pinyin=not bool(args.exclude_same_pinyin),
            require_confusion=bool(args.require_confusion),
        )
    if not evidence and bool(args.require_evidence):
        return None
    input_block["phonetic_hotword_span_evidence"] = evidence
    evidence_text = format_evidence(evidence)
    return {
        **record,
        "input": input_block,
        "messages": make_messages(record, input_block, evidence_text, args),
    }


def explode_augmented_record(record: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    evidence = list((record.get("input", {}) or {}).get("phonetic_hotword_span_evidence", []) or [])
    if not evidence or not bool(args.explode_evidence):
        return [record]
    rows: List[Dict[str, Any]] = []
    max_items = max(1, int(args.max_exploded_per_row))
    for idx, item in enumerate(evidence[:max_items], start=1):
        input_block = dict(record.get("input", {}) or {})
        input_block["phonetic_hotword_span_evidence"] = [item]
        base_record = {key: value for key, value in record.items() if key != "messages"}
        base_record["input"] = input_block
        base_record["id"] = f"{record.get('id', '')}#phon{idx}"
        rows.append(
            {
                **base_record,
                "messages": make_messages(base_record, input_block, format_evidence([item]), args),
            }
        )
    return rows


def load_domain_terms(path: str) -> List[str]:
    terms = list(DEFAULT_DOMAIN_TERMS)
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            item = norm(line.strip())
            if item and item not in terms:
                terms.append(item)
    return sorted(set(terms), key=lambda item: (-len(item), item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--train-output", default="")
    parser.add_argument("--dev-output", default="")
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=706)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--domain-terms", default="")
    parser.add_argument("--max-terms", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-spans-per-term", type=int, default=2)
    parser.add_argument("--max-pinyin-distance", type=int, default=1)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--max-pinyin", type=int, default=10)
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--require-confusion", action="store_true", help="Keep only terms that have at least one phonetic-confusion span")
    parser.add_argument("--auto-reference-terms", action="store_true", help="Add reference n-grams as candidate terms when no fixed domain lexicon is available")
    parser.add_argument("--diff-reference-terms", action="store_true", help="Build phonetic evidence only from candidate-vs-reference replace spans")
    parser.add_argument("--auto-max-terms", type=int, default=64)
    parser.add_argument("--min-term-len", type=int, default=2)
    parser.add_argument("--max-term-len", type=int, default=4)
    parser.add_argument("--min-span-len", type=int, default=2)
    parser.add_argument("--max-len-gap", type=int, default=1)
    parser.add_argument("--exclude-same-pinyin", action="store_true", help="Drop exact same-pinyin spans, useful for removing mostly traditional/simplified variants")
    parser.add_argument("--compact-prompt", action="store_true", help="Write a short phonetic-focused prompt instead of the full CB-Whisper bridge prompt")
    parser.add_argument("--explode-evidence", action="store_true", help="Create one focused SFT row per evidence item")
    parser.add_argument("--max-exploded-per-row", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domain_terms = load_domain_terms(args.domain_terms)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with_evidence = 0
    for idx, record in enumerate(read_jsonl(args.input), start=1):
        augmented = augment_record(record, args, domain_terms)
        if augmented is None:
            skipped += 1
        else:
            exploded = explode_augmented_record(augmented, args)
            rows.extend(exploded)
            if augmented.get("input", {}).get("phonetic_hotword_span_evidence"):
                with_evidence += len(exploded)
        if args.limit and idx >= int(args.limit):
            break

    if args.train_output and args.dev_output:
        rng = random.Random(int(args.seed))
        rng.shuffle(rows)
        dev_size = max(0, min(int(args.dev_size), len(rows)))
        dev = rows[:dev_size]
        train = rows[dev_size:]
        for row in train:
            row["split"] = "train"
        for row in dev:
            row["split"] = "dev"
        train_written = write_jsonl(args.train_output, train)
        dev_written = write_jsonl(args.dev_output, dev)
        result = {
            "input": args.input,
            "train_output": args.train_output,
            "dev_output": args.dev_output,
            "train_written": train_written,
            "dev_written": dev_written,
            "skipped": skipped,
            "with_evidence": with_evidence,
        }
    elif args.output:
        written = write_jsonl(args.output, rows)
        result = {
            "input": args.input,
            "output": args.output,
            "written": written,
            "skipped": skipped,
            "with_evidence": with_evidence,
        }
    else:
        raise ValueError("provide --output or both --train-output/--dev-output")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
