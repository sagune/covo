#!/usr/bin/env python
"""Build non-leak Shuili-style normalization/domain SFT data for COVO."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pypinyin import Style, lazy_pinyin


DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.text import normalize_chinese_text  # type: ignore  # noqa: E402


SYSTEM_MESSAGE = (
    "你是一个保守的中文 ASR 后纠错器。根据 ASR top-1、N-best、拼音、领域术语证据和输出规范，"
    "输出最终纠错后的中文句子。必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
    "不要输出解释、推理过程、Markdown 或额外字段。"
)

FILLERS = ["那么", "这个", "这一个", "那个", "咱们", "我们", "的话", "就是", "呢", "啊", "呃", "嗯"]
DOMAIN_CONFUSIONS: List[Tuple[str, str]] = [
    ("参建单位", "参见单位"),
    ("重力式码头", "重力是码头"),
    ("挖运方案", "Volume 5"),
    ("开挖", "开发"),
    ("土石堤防", "土石敌方"),
    ("围海造田", "为海造田"),
    ("地基", "低级"),
    ("填筑", "天主"),
    ("闸门", "扎门"),
    ("水闸", "水杂"),
    ("混凝土", "红凝土"),
    ("导流", "倒流"),
    ("水轮机", "水论机"),
    ("基坑", "机坑"),
]

TEMPLATES = [
    "那么我们介绍{term}的施工方法",
    "这个部分主要讨论{term}的技术要求",
    "咱们下面看一下{term}的工程特点",
    "在水利工程施工中{term}非常重要",
    "这个案例说明了{term}的控制要点",
    "那么{term}的方案应该怎么选择",
    "我们再看{term}相关的施工组织",
    "这里需要重点关注{term}的质量控制",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramc-train", required=True)
    parser.add_argument("--kb", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--dev-size", type=int, default=3000)
    parser.add_argument("--domain-ratio", type=float, default=0.22)
    parser.add_argument("--noop-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=707)
    parser.add_argument("--max-nbest", type=int, default=6)
    return parser.parse_args()


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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
    return normalize_chinese_text(str(text or ""))


def pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))


def build_opencc():
    try:
        from opencc import OpenCC

        return OpenCC("s2t")
    except Exception:
        return None


def unique(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = norm(item)
        if key and key not in seen:
            seen.add(key)
            out.append(str(item))
    return out


def duplicate_span(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    start = rng.randint(0, max(0, len(text) - 2))
    length = rng.randint(1, min(4, len(text) - start))
    span = text[start : start + length]
    return text[: start + length] + span + text[start + length :]


def delete_filler_or_function(text: str, rng: random.Random) -> str:
    candidates = [item for item in FILLERS + ["的", "了", "一", "个", "这"] if item in text]
    if not candidates:
        return text
    item = rng.choice(candidates)
    out = text.replace(item, "", 1)
    return out or text


def insert_filler_or_function(text: str, rng: random.Random) -> str:
    item = rng.choice(["那么", "这个", "这一个", "的", "是", "一"])
    pos = rng.randint(0, len(text))
    return text[:pos] + item + text[pos:]


def apply_domain_confusion(text: str, rng: random.Random) -> tuple[str, str, str] | None:
    pairs = [(good, bad) for good, bad in DOMAIN_CONFUSIONS if good in text]
    if not pairs:
        return None
    good, bad = rng.choice(pairs)
    return text.replace(good, bad, 1), good, bad


def corrupt_text(text: str, rng: random.Random, opencc: Any) -> tuple[str, List[str], List[tuple[str, str]]]:
    corrupted = text
    tags: List[str] = []
    domain_pairs: List[tuple[str, str]] = []
    if rng.random() < 0.45 and opencc is not None:
        corrupted = opencc.convert(corrupted)
        tags.append("traditional")
    if rng.random() < 0.30:
        corrupted = delete_filler_or_function(corrupted, rng)
        tags.append("delete_function")
    if rng.random() < 0.18:
        corrupted = insert_filler_or_function(corrupted, rng)
        tags.append("insert_function")
    if rng.random() < 0.22:
        corrupted = duplicate_span(corrupted, rng)
        tags.append("repeat")
    domain = apply_domain_confusion(corrupted, rng)
    if domain is not None:
        corrupted, good, bad = domain
        tags.append("domain_confusion")
        domain_pairs.append((good, bad))
    if corrupted == text and opencc is not None and rng.random() < 0.5:
        corrupted = opencc.convert(corrupted)
        tags.append("traditional")
    return corrupted, tags, domain_pairs


def load_kb_terms(path: str | Path, limit: int = 80) -> List[Dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        term = norm(row.get("term", ""))
        if term and len(term) >= 2:
            rows.append({**row, "term": term})
    rows.sort(key=lambda item: (-int(item.get("support", 0) or 0), -len(item["term"]), item["term"]))
    return rows[:limit]


def qwen_row(row_id: str, source: str, target: str, asr_top1: str, nbest: List[str], kb_terms: List[str], tags: List[str]) -> Dict[str, Any]:
    nbest = unique(nbest)[:6]
    if asr_top1 not in nbest:
        nbest = [asr_top1] + nbest
    pinyin_rows = [pinyin(item) for item in nbest[:6]]
    lines = [
        "任务：根据 ASR top-1、N-best、拼音和领域术语证据进行中文 ASR 后纠错。",
        "输出规范：默认输出简体中文；压缩明显重复片段；保留上下文支持的口语功能词；不要无证据插入领域词。",
        f"ASR top-1: {asr_top1}",
        "N-best candidates:",
    ]
    for idx, item in enumerate(nbest[:6], start=1):
        lines.append(f"{idx}. {item}")
    lines.append("Pinyin:")
    for idx, item in enumerate(pinyin_rows, start=1):
        lines.append(f"{idx}. {item}")
    if kb_terms:
        lines.append("Domain KB evidence (non-reference, soft evidence):")
        for idx, term in enumerate(kb_terms[:6], start=1):
            lines.append(f"{idx}. {term} | pinyin={pinyin(term)} | evidence=synthetic_supported")
    if tags:
        lines.append("Observed risk types: " + ", ".join(tags))
    lines.append("请输出 JSON：{\"text\":\"纠错后的完整句子\"}")
    return {
        "id": row_id,
        "source": source,
        "reference": target,
        "input": {
            "asr_top1": asr_top1,
            "nbest": nbest[:6],
            "nbest_pinyin": pinyin_rows,
            "domain_kb_evidence": [{"term": term, "pinyin": pinyin(term)} for term in kb_terms[:6]],
            "tags": tags,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "\n".join(lines)},
            {"role": "assistant", "content": json.dumps({"text": target}, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def ramc_target(record: Dict[str, Any]) -> str:
    for field in ("reference", "text", "target"):
        value = norm(record.get(field, ""))
        if value:
            return value
    metadata = record.get("metadata", {}) or {}
    return norm(metadata.get("original_text", ""))


def build_ramc_row(record: Dict[str, Any], idx: int, rng: random.Random, opencc: Any, noop: bool) -> Dict[str, Any] | None:
    target = ramc_target(record)
    if len(target) < 4 or len(target) > 80:
        return None
    if noop:
        asr_top1 = target
        tags = ["noop_simplified_target"]
        nbest = [target, duplicate_span(target, rng), delete_filler_or_function(target, rng)]
    else:
        asr_top1, tags, _ = corrupt_text(target, rng, opencc)
        if norm(asr_top1) == target:
            return None
        nbest = [
            asr_top1,
            target,
            delete_filler_or_function(target, rng),
            duplicate_span(target, rng),
            insert_filler_or_function(target, rng),
        ]
    return qwen_row(f"ramc_norm_{idx}", "ramc_oral_norm_domain_synthetic", target, asr_top1, nbest, [], tags)


def build_domain_row(kb_terms: List[Dict[str, Any]], idx: int, rng: random.Random, opencc: Any, noop: bool) -> Dict[str, Any]:
    term = rng.choice(kb_terms)["term"]
    template = rng.choice(TEMPLATES)
    target = norm(template.format(term=term))
    if noop:
        asr_top1 = target
        tags = ["domain_noop"]
        nbest = [target, duplicate_span(target, rng)]
    else:
        asr_top1, tags, domain_pairs = corrupt_text(target, rng, opencc)
        if not domain_pairs and rng.random() < 0.7:
            for good, bad in DOMAIN_CONFUSIONS:
                if good in target:
                    asr_top1 = target.replace(good, bad, 1)
                    tags.append("domain_confusion")
                    break
        nbest = [asr_top1, target, duplicate_span(target, rng), delete_filler_or_function(target, rng)]
    kb_evidence = sorted({term, *[good for good, _ in DOMAIN_CONFUSIONS if good in target]}, key=lambda x: (-len(x), x))
    return qwen_row(f"shuili_domain_norm_{idx}", "shuili_domain_norm_synthetic_nonleak", target, asr_top1, nbest, kb_evidence, tags)


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    opencc = build_opencc()
    ramc = list(read_jsonl(args.ramc_train))
    kb_terms = load_kb_terms(args.kb)
    if not ramc:
        raise ValueError("empty RAMC input")
    if not kb_terms:
        raise ValueError("empty KB input")
    total = int(args.train_size) + int(args.dev_size)
    rows: List[Dict[str, Any]] = []
    attempts = 0
    while len(rows) < total and attempts < total * 20:
        attempts += 1
        use_domain = rng.random() < float(args.domain_ratio)
        noop = rng.random() < float(args.noop_ratio)
        if use_domain:
            row = build_domain_row(kb_terms, len(rows), rng, opencc, noop)
        else:
            row = build_ramc_row(rng.choice(ramc), len(rows), rng, opencc, noop)
        if row is not None:
            rows.append(row)
    if len(rows) < total:
        raise RuntimeError(f"only built {len(rows)} rows, expected {total}")
    rng.shuffle(rows)
    dev = rows[: int(args.dev_size)]
    train = rows[int(args.dev_size) :]
    train_written = write_jsonl(args.train_output, train)
    dev_written = write_jsonl(args.dev_output, dev)
    summary = {
        "train_output": args.train_output,
        "dev_output": args.dev_output,
        "train_rows": train_written,
        "dev_rows": dev_written,
        "ramc_source_rows": len(ramc),
        "kb_terms": len(kb_terms),
        "domain_ratio": args.domain_ratio,
        "noop_ratio": args.noop_ratio,
        "attempts": attempts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
