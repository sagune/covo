#!/usr/bin/env python
"""Build Shuili-style oral COVO SFT data from RAMC text records.

The generated data keeps the RAMC reference text but makes the n-best evidence
look closer to Shuili lectures: more candidates, natural adjacent repetitions,
short fragments, and no-op preservation rows. Optional external no-op rows can
be mixed in to reduce over-editing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, List

from pypinyin import Style, lazy_pinyin


PUNCT_RE = re.compile(r"[\s，。？！、,.!?；;：:“”\"'（）()\[\]{}<>《》]")
FILLERS = [
    "然后就是",
    "就是说",
    "就是",
    "然后",
    "那个",
    "这个",
    "的话",
    "那么",
    "咱们",
    "我们",
    "嗯",
    "呃",
    "啊",
    "呢",
    "嘛",
    "呀",
    "吧",
]
FALSE_DOMAIN = ["水利工程", "闸门", "地基", "石方", "施工", "基坑", "开挖", "填筑"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramc-raw", required=True)
    parser.add_argument("--noop-source", default="", help="Backward-compatible single external no-op/hotword source.")
    parser.add_argument(
        "--external-source",
        action="append",
        default=[],
        help="External Qwen-message JSONL to mix in, e.g. hotword no-op or hotword-aware CB-Whisper SFT data.",
    )
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--train-size", type=int, default=24000)
    parser.add_argument("--dev-size", type=int, default=1200)
    parser.add_argument("--noop-ratio", type=float, default=0.25)
    parser.add_argument("--external-noop-ratio", type=float, default=0.15)
    parser.add_argument("--hotword-aware-ramc", action="store_true")
    parser.add_argument("--seed", type=int, default=705)
    parser.add_argument("--max-nbest", type=int, default=10)
    return parser.parse_args()


def norm(text: Any) -> str:
    return PUNCT_RE.sub("", str(text or "")).strip()


def read_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        raw = norm(value)
        if raw and raw not in seen:
            out.append(raw)
            seen.add(raw)
    return out


def pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL, errors="ignore"))


def choose_hotwords(text: str, rng: random.Random) -> tuple[List[str], List[str]]:
    protected = [word for word in FALSE_DOMAIN if word in text]
    spans = []
    max_unit = min(4, max(2, len(text) // 2))
    for unit in range(max_unit, 1, -1):
        for start in range(0, max(0, len(text) - unit + 1)):
            span = text[start : start + unit]
            if span and span not in FILLERS and len(set(span)) > 1:
                spans.append(span)
    rng.shuffle(spans)
    target_n = rng.randint(1, 3)
    for span in spans:
        if span not in protected:
            protected.append(span)
        if len(protected) >= target_n:
            break
    protected = protected[:3]
    false_items = [word for word in FALSE_DOMAIN if word not in text and word not in protected]
    rng.shuffle(false_items)
    prompt = unique(protected + false_items[: rng.randint(1, 3)])
    return protected, prompt


def remove_one_filler(text: str, rng: random.Random) -> str:
    present = [item for item in FILLERS if item in text]
    if not present:
        return text
    return text.replace(rng.choice(present), "", 1)


def remove_some_fillers(text: str, rng: random.Random, max_remove: int = 3) -> str:
    out = text
    present = [item for item in FILLERS if item in out]
    rng.shuffle(present)
    for item in present[: max(0, max_remove)]:
        out = out.replace(item, "", 1)
    return out or text


def short_fragment(text: str, rng: random.Random) -> str:
    if len(text) <= 8:
        return text
    if rng.random() < 0.45:
        return text[rng.randint(1, min(5, len(text) - 4)) :]
    if rng.random() < 0.70:
        return text[: rng.randint(max(4, len(text) - 8), len(text) - 1)]
    start = rng.randint(0, max(0, len(text) // 3))
    end = rng.randint(max(start + 4, len(text) // 2), len(text))
    return text[start:end]


def duplicate_span(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    start = rng.randint(0, len(text) - 2)
    unit = rng.randint(1, min(4, len(text) - start))
    span = text[start : start + unit]
    return text[: start + unit] + span + text[start + unit :]


def natural_repeat_variant(text: str, rng: random.Random) -> str:
    if len(text) < 4:
        return text
    starts = list(range(0, max(1, len(text) - 3)))
    rng.shuffle(starts)
    for start in starts:
        for unit in (2, 3, 4):
            span = text[start : start + unit]
            if len(span) == unit and len(set(span)) > 1:
                return text[: start + unit] + span + text[start + unit :]
    return duplicate_span(text, rng)


def false_insert(text: str, rng: random.Random) -> str:
    word = rng.choice(FALSE_DOMAIN)
    pos = rng.randint(0, len(text))
    return text[:pos] + word + text[pos:]


def make_variants(ref: str, rng: random.Random) -> List[str]:
    variants = [
        ref,
        remove_one_filler(ref, rng),
        remove_some_fillers(ref, rng, 2),
        remove_some_fillers(ref, rng, 4),
        short_fragment(ref, rng),
        duplicate_span(ref, rng),
        natural_repeat_variant(ref, rng),
    ]
    for _ in range(3):
        base = rng.choice(variants)
        op = rng.choice([remove_one_filler, short_fragment, duplicate_span, natural_repeat_variant, false_insert])
        variants.append(op(base, rng))
    if rng.random() < 0.45:
        variants.append(false_insert(remove_some_fillers(ref, rng, 2), rng))
    return unique(variants)


def diff_summary(base: str, candidate: str) -> str:
    if base == candidate:
        return "same"
    prefix = 0
    while prefix < min(len(base), len(candidate)) and base[prefix] == candidate[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(base), len(candidate)) - prefix
        and base[len(base) - 1 - suffix] == candidate[len(candidate) - 1 - suffix]
    ):
        suffix += 1
    old = base[prefix : len(base) - suffix if suffix else len(base)]
    new = candidate[prefix : len(candidate) - suffix if suffix else len(candidate)]
    return f"{prefix}:{len(base)-suffix} {old or '∅'}->{new or '∅'}"


def to_messages(record: dict) -> dict:
    inp = record["input"]
    nbest = list(inp.get("nbest", []) or [])
    pinyins = list(inp.get("nbest_pinyin", []) or [])
    base = str(inp.get("asr_top1", ""))
    protected_hotwords = list(inp.get("protected_hotwords", []) or [])
    prompt_hotwords = list(inp.get("prompt_hotwords", []) or [])
    false_hotwords = [item for item in prompt_hotwords if item not in protected_hotwords]
    confusables = []
    for idx, cand in enumerate(nbest[1:], start=2):
        hotword_note = {
            "keeps_hotword": [word for word in protected_hotwords if word in cand],
            "drops_hotword": [word for word in protected_hotwords if word not in cand],
            "keeps_false_hotword": [word for word in false_hotwords if word in cand],
        }
        confusables.append(
            f"- cand#{idx}: {cand} | diff={diff_summary(base, cand)} | hotword_delta={json.dumps(hotword_note, ensure_ascii=False, separators=(',', ':'))}"
        )
        if len(confusables) >= 4:
            break
    user_lines = [
        "任务：利用 N-best 中的一致部分、不一致片段和易混淆候选进行 ASR 后纠错。",
        "候选可能包含正确改法，也可能包含短片段、口语重复或同音误导；请优先选择有多候选证据且语义完整的结果。",
        "如果证据不足，保持 ASR top-1 不变；如果候选中有完整的口语化表达且与上下文一致，可以补回被漏掉的口语词或自然重复。",
        "热词规则：候选证据支持的 protected_hotwords 应尽量保留；prompt 中但证据不足的热词不要强行插入。",
        f"ASR top-1: {base}",
    ]
    if prompt_hotwords or protected_hotwords:
        user_lines += [
            "Hotword evidence:",
            f"- protected_hotwords: {'，'.join(protected_hotwords) if protected_hotwords else '无'}",
            f"- prompt_hotwords: {'，'.join(prompt_hotwords) if prompt_hotwords else '无'}",
            f"- false_hotword_warning: {'，'.join(false_hotwords) if false_hotwords else '无'}",
            "Candidate hotword support:",
        ]
        for idx, text in enumerate(nbest, start=1):
            keeps = [word for word in protected_hotwords if word in text]
            drops = [word for word in protected_hotwords if word not in text]
            user_lines.append(f"{idx}. keeps={','.join(keeps) or '无'} drops={','.join(drops) or '无'}")
    user_lines.append("N-best:")
    user_lines += [f"{idx}. {text}" for idx, text in enumerate(nbest, start=1)]
    if pinyins:
        user_lines.append("Pinyin:")
        user_lines += [f"{idx}. {text}" for idx, text in enumerate(pinyins[: len(nbest)], start=1)]
    if confusables:
        user_lines.append("Confusable candidates:")
        user_lines += confusables
    user_lines.append('请输出 JSON：{"text":"纠错后的完整句子"}')
    return {
        **record,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个保守的中文 ASR 后纠错器。根据 ASR top-1、N-best 候选、拼音和共识片段，"
                    "输出最终纠错后的中文句子。必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
                    "不要输出解释、推理过程、Markdown 或额外字段。"
                ),
            },
            {"role": "user", "content": "\n".join(user_lines)},
            {"role": "assistant", "content": json.dumps({"text": record["reference"]}, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def build_ramc_record(row: dict, rng: random.Random, max_nbest: int, force_noop: bool, hotword_aware: bool) -> dict | None:
    ref = norm(row.get("reference", ""))
    if len(ref) < 4:
        return None
    variants = make_variants(ref, rng)
    if force_noop:
        top1 = ref
    else:
        non_ref = [item for item in variants if item != ref]
        top1 = rng.choice(non_ref or [ref])
    candidates = unique([top1] + rng.sample(variants, k=len(variants)))
    if ref not in candidates:
        candidates.insert(rng.randint(1, min(len(candidates), 5)), ref)
    candidates = candidates[:max_nbest]
    if ref not in candidates:
        candidates[-1] = ref
    protected_hotwords, prompt_hotwords = choose_hotwords(ref, rng) if hotword_aware else ([], [])
    return to_messages(
        {
            "id": row.get("id", ""),
            "source": "magicdata_ramc_shuili_style_synthetic",
            "split": "",
            "reference": ref,
            "input": {
                "asr_top1": top1,
                "nbest": candidates,
                "nbest_pinyin": [pinyin(item) for item in candidates],
                "protected_hotwords": protected_hotwords,
                "prompt_hotwords": prompt_hotwords,
                "kws_hotwords": prompt_hotwords,
                "metadata": dict((row.get("input", {}) or {}).get("metadata", {}) or {}),
            },
        }
    )


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    raw_rows = list(read_jsonl(args.ramc_raw))
    rng.shuffle(raw_rows)

    total = int(args.train_size) + int(args.dev_size)
    external_sources = list(args.external_source or [])
    if args.noop_source:
        external_sources.append(args.noop_source)

    external_noop_n = int(total * max(0.0, float(args.external_noop_ratio))) if external_sources else 0
    ramc_total = total - external_noop_n
    ramc_noop_n = int(ramc_total * max(0.0, float(args.noop_ratio)))
    ramc_rewrite_n = ramc_total - ramc_noop_n

    built = []
    cursor = 0
    for force_noop, count in [(False, ramc_rewrite_n), (True, ramc_noop_n)]:
        made = 0
        while made < count and cursor < len(raw_rows):
            record = build_ramc_record(
                raw_rows[cursor],
                rng,
                int(args.max_nbest),
                force_noop=force_noop,
                hotword_aware=bool(args.hotword_aware_ramc),
            )
            cursor += 1
            if record is None:
                continue
            built.append(record)
            made += 1

    if external_sources and external_noop_n > 0:
        per_source = max(1, external_noop_n // len(external_sources))
        external_rows = []
        for source in external_sources:
            rows = list(read_jsonl(source))
            rng.shuffle(rows)
            external_rows.extend(rows[:per_source])
        if len(external_rows) < external_noop_n:
            for source in external_sources:
                rows = list(read_jsonl(source))
                rng.shuffle(rows)
                external_rows.extend(rows[: max(0, external_noop_n - len(external_rows))])
                if len(external_rows) >= external_noop_n:
                    break
        rng.shuffle(external_rows)
        built.extend(external_rows[:external_noop_n])

    rng.shuffle(built)
    train = built[: int(args.train_size)]
    dev = built[int(args.train_size) : int(args.train_size) + int(args.dev_size)]
    for row in train:
        row["split"] = "train"
    for row in dev:
        row["split"] = "dev"

    train_count = write_jsonl(args.train_output, train)
    dev_count = write_jsonl(args.dev_output, dev)
    print(
        json.dumps(
            {
                "train_output": args.train_output,
                "dev_output": args.dev_output,
                "train_written": train_count,
                "dev_written": dev_count,
                "ramc_rewrite_target": ramc_rewrite_n,
                "ramc_noop_target": ramc_noop_n,
                "external_noop_target": external_noop_n,
                "external_sources": external_sources,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
