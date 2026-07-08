#!/usr/bin/env python
"""Convert Shuili CB-Whisper candidate pools to COVO Qwen-message data."""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SYSTEM_MESSAGE = (
    "你是一个保守的中文 ASR 后纠错器。根据 ASR top-1、N-best 候选、拼音、"
    "共识片段和易混淆候选，输出最终纠错后的中文句子。必须只输出一个合法 JSON 对象，"
    "格式为 {\"text\":\"...\"}。不要输出解释、推理过程、Markdown 或额外字段。"
)

INSTRUCTION = (
    "任务：利用 N-best 中的一致部分、不一致片段和易混淆候选进行中文 ASR 后纠错。"
    "Stable spans 是多候选共同支持的内容，应优先保留；Uncertain spans 是候选分歧位置，"
    "只在拼音、上下文和多候选证据支持时修改。不要为了让句子更书面而删除口语词，"
    "不要自由添加 N-best 中没有证据的前后缀。如果证据不足，保持 ASR top-1 不变。"
)

SOURCE_AWARE_GUIDANCE = (
    "候选来源说明：neutral_whisper 是未注入热词的声学识别结果，通常更干净但可能漏掉口语词；"
    "cbwhisper 候选包含热词/上下文偏置，可能补出口语词或专业词，也可能产生热词幻觉。"
    "请优先以 neutral_whisper 作为干净锚点，只在 CB 候选、拼音和上下文共同支持时吸收其局部内容。"
)

SOURCE_AWARE_STRICT_GUIDANCE = (
    "请尽量直接输出某一个 N-best 候选，或只做最小必要合并；不要自由改写。"
    "当 neutral_whisper 与 CB 候选冲突时，只有多个 CB 候选共同支持某个口语词、专业词或局部片段时才采用 CB 局部；"
    "如果只是单条 CB 候选出现热词或奇怪重复，视为可能的上下文偏置幻觉，应保持 neutral_whisper。"
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def normalize_text(text: Any) -> str:
    return "".join(str(text or "").split())


def unique_texts(items: Iterable[Any], limit: int) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        output.append(text)
        seen.add(key)
        if len(output) >= limit:
            break
    return output


def unique_texts_with_sources(raw_input: Dict[str, Any], limit: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    nbest = list(raw_input.get("nbest", []) or [])
    raw_sources = list(raw_input.get("nbest_sources", []) or [])
    output: List[str] = []
    sources: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(nbest):
        text = str(item or "").strip()
        key = normalize_text(text)
        if not text or not key or key in seen:
            continue
        source = raw_sources[idx] if idx < len(raw_sources) and isinstance(raw_sources[idx], dict) else {}
        output.append(text)
        sources.append(source)
        seen.add(key)
        if len(output) >= limit:
            break
    return output, sources


def edit_distance(a: str, b: str) -> int:
    a = normalize_text(a)
    b = normalize_text(b)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if char_a == char_b else 1),
            )
        previous = current
    return previous[-1]


def merge_support_spans(
    base: str,
    support_ratios: List[float],
    threshold: float,
    stable: bool,
    max_spans: int,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    start = None
    values: List[float] = []
    sentinel = 2.0 if stable else -1.0
    for idx, ratio in enumerate(support_ratios + [sentinel]):
        selected = ratio >= threshold if stable else ratio < threshold
        if selected and start is None:
            start = idx
            values = [ratio]
        elif selected:
            values.append(ratio)
        elif start is not None:
            text = base[start:idx]
            if text:
                spans.append(
                    {
                        "start": int(start),
                        "end": int(idx),
                        "text": text,
                        "support": float(sum(values) / max(len(values), 1)),
                    }
                )
            start = None
            values = []
    spans.sort(key=lambda row: (-(int(row["end"]) - int(row["start"])), int(row["start"])))
    return spans[: max(0, max_spans)]


def span_variant(base: str, candidate: str, start: int, end: int) -> str:
    matcher = SequenceMatcher(a=base, b=candidate)
    pieces: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i2 <= start or i1 >= end:
            continue
        if tag == "equal":
            left = max(i1, start)
            right = min(i2, end)
            offset = left - i1
            pieces.append(candidate[j1 + offset:j1 + offset + (right - left)])
        else:
            pieces.append(candidate[j1:j2])
    return "".join(pieces).strip()


def build_nbest_consensus(
    nbest: List[str],
    threshold: float,
    max_stable: int,
    max_uncertain: int,
    max_variants: int,
) -> Dict[str, Any]:
    cleaned = [normalize_text(item) for item in nbest if normalize_text(item)]
    if not cleaned:
        return {"threshold": float(threshold), "stable_spans": [], "uncertain_spans": []}
    base = cleaned[0]
    equal_counts = [0 for _ in base]
    for candidate in cleaned:
        for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=base, b=candidate).get_opcodes():
            if tag == "equal":
                for idx in range(i1, i2):
                    if 0 <= idx < len(equal_counts):
                        equal_counts[idx] += 1
    denom = max(1, len(cleaned))
    support_ratios = [count / denom for count in equal_counts]
    stable_spans = merge_support_spans(base, support_ratios, threshold, True, max_stable)
    uncertain_spans = merge_support_spans(base, support_ratios, threshold, False, max_uncertain)
    enriched = []
    for span in uncertain_spans:
        variants: List[str] = []
        seen = set()
        for candidate in cleaned:
            variant = span_variant(base, candidate, int(span["start"]), int(span["end"]))
            key = normalize_text(variant)
            if key and key not in seen:
                variants.append(variant)
                seen.add(key)
            if len(variants) >= max_variants:
                break
        enriched.append({**span, "variants": variants})
    return {"threshold": float(threshold), "stable_spans": stable_spans, "uncertain_spans": enriched}


def char_distance(a: str, b: str) -> int:
    matcher = SequenceMatcher(a=normalize_text(a), b=normalize_text(b))
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            distance += max(i2 - i1, j2 - j1)
    return distance


def diff_summary(base: str, candidate: str, max_parts: int) -> str:
    parts: List[str] = []
    matcher = SequenceMatcher(a=normalize_text(base), b=normalize_text(candidate))
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src = normalize_text(base)[i1:i2] or "∅"
        dst = normalize_text(candidate)[j1:j2] or "∅"
        parts.append(f"{i1}:{i2} {src}->{dst}")
        if len(parts) >= max_parts:
            break
    return "; ".join(parts)


def confusable_candidates(nbest: List[str], max_items: int) -> List[Dict[str, Any]]:
    if len(nbest) <= 1 or max_items <= 0:
        return []
    top1 = nbest[0]
    top1_norm = normalize_text(top1)
    rows: List[Tuple[int, float, int, str]] = []
    seen = {top1_norm}
    for rank, candidate in enumerate(nbest[1:], 2):
        key = normalize_text(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        sim = SequenceMatcher(a=top1_norm, b=key).ratio()
        dist = char_distance(top1, candidate)
        if sim >= 0.55 and dist > 0:
            rows.append((dist, -sim, rank, candidate))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        {
            "rank": rank,
            "text": candidate,
            "similarity": float(-neg_sim),
            "distance": int(dist),
            "diff": diff_summary(top1, candidate, 4),
        }
        for dist, neg_sim, rank, candidate in rows[:max_items]
    ]


def format_consensus(consensus: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    stable = list(consensus.get("stable_spans", []) or [])
    uncertain = list(consensus.get("uncertain_spans", []) or [])
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
            variants = json.dumps(item.get("variants", []), ensure_ascii=False, separators=(",", ":"))
            lines.append(
                f"- {int(item.get('start', 0))}:{int(item.get('end', 0))} "
                f"{item.get('text', '')} support={float(item.get('support', 0.0)):.3f} variants={variants}"
            )
    return lines


def _source_label(source: Dict[str, Any]) -> str:
    name = str(source.get("source", "unknown") or "unknown")
    if "neutral_whisper" in name:
        return "neutral_whisper"
    if name.startswith("cbwhisper"):
        return name
    if name.startswith("multiprompt"):
        prompt = source.get("prompt")
        return f"{name}/{prompt}" if prompt else name
    return name


def build_user_prompt(
    input_block: Dict[str, Any],
    max_pinyin: int,
    max_confusables: int,
    include_source_tags: bool = False,
    source_aware_guidance: bool = False,
    source_aware_strict: bool = False,
) -> str:
    lines = [INSTRUCTION]
    if source_aware_guidance:
        lines.append(SOURCE_AWARE_GUIDANCE)
    if source_aware_strict:
        lines.append(SOURCE_AWARE_STRICT_GUIDANCE)
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    lines.append(f"ASR top-1: {asr_top1}")
    nbest = list(input_block.get("nbest", []) or [])
    if nbest:
        lines.append("N-best:")
        for idx, hyp in enumerate(nbest, 1):
            lines.append(f"{idx}. {hyp}")
    if include_source_tags:
        sources = list(input_block.get("nbest_sources", []) or [])
        if sources:
            lines.append("Candidate sources:")
            for idx, source in enumerate(sources[: len(nbest)], 1):
                lines.append(f"{idx}. {_source_label(source)}")
    pinyin = list(input_block.get("nbest_pinyin", []) or [])[:max_pinyin]
    if pinyin:
        lines.append("Pinyin:")
        for idx, item in enumerate(pinyin, 1):
            lines.append(f"{idx}. {item}")
    consensus_lines = format_consensus(input_block.get("nbest_consensus", {}) or {})
    if consensus_lines:
        lines.extend(consensus_lines)
    confusables = list(input_block.get("confusable_candidates", []) or [])[:max_confusables]
    if confusables:
        lines.append("Confusable candidates:")
        for item in confusables:
            lines.append(
                f"- cand#{int(item.get('rank', 0))} sim={float(item.get('similarity', 0.0)):.3f} "
                f"dist={int(item.get('distance', 0))}: {item.get('text', '')} | diff={item.get('diff', '')}"
            )
    lines.append('请输出 JSON：{"text":"纠错后的完整句子"}')
    return "\n".join(lines)


def convert_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    for idx, record in enumerate(read_jsonl(Path(args.input))):
        reference = str(record.get(args.reference_field, "")).strip()
        raw_input = dict(record.get("input", {}) or {})
        nbest, nbest_sources = unique_texts_with_sources(raw_input, int(args.max_nbest))
        if not nbest:
            top1 = str(raw_input.get("asr_top1", record.get("asr_top1", ""))).strip()
            nbest = [top1] if top1 else []
            nbest_sources = [{} for _ in nbest]
        asr_top1 = nbest[0] if nbest else ""
        pinyin = list(raw_input.get("nbest_pinyin", []) or [])[: len(nbest)]
        consensus = build_nbest_consensus(
            nbest=nbest,
            threshold=float(args.consensus_threshold),
            max_stable=int(args.max_stable_spans),
            max_uncertain=int(args.max_uncertain_spans),
            max_variants=int(args.max_span_variants),
        )
        input_block = {
            "asr_top1": asr_top1,
            "nbest": nbest,
            "nbest_sources": nbest_sources,
            "nbest_pinyin": pinyin,
            "asr_top1_pinyin": pinyin[0] if pinyin else "",
            "nbest_consensus": consensus,
            "confusable_candidates": confusable_candidates(nbest, int(args.max_confusables)),
        }
        output = {
            "id": str(record.get("id", f"shuili-{idx}")),
            "source": "shuili/cbwhisper-v3-candidate-pool",
            "dataset": "shuili",
            "split": str(record.get("split", "test")),
            "reference": reference,
            "input": input_block,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        input_block,
                        max_pinyin=int(args.max_pinyin),
                        max_confusables=int(args.max_confusables),
                        include_source_tags=bool(args.include_source_tags),
                        source_aware_guidance=bool(args.source_aware_guidance),
                        source_aware_strict=bool(args.source_aware_strict),
                    ),
                },
            ],
        }
        if reference:
            output["messages"].append(
                {
                    "role": "assistant",
                    "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
                }
            )
        yield output
        if args.limit and idx + 1 >= int(args.limit):
            break


def summarize(path: Path, summary_path: Path) -> Dict[str, Any]:
    rows = list(read_jsonl(path))
    exact_ref_in_nbest = 0
    with_consensus = 0
    with_confusables = 0
    nbest_total = 0
    for row in rows:
        input_block = row.get("input", {}) or {}
        nbest = list(input_block.get("nbest", []) or [])
        ref = normalize_text(row.get("reference", ""))
        if ref and any(normalize_text(item) == ref for item in nbest):
            exact_ref_in_nbest += 1
        consensus = input_block.get("nbest_consensus", {}) or {}
        if consensus.get("stable_spans") or consensus.get("uncertain_spans"):
            with_consensus += 1
        if input_block.get("confusable_candidates"):
            with_confusables += 1
        nbest_total += len(nbest)
    summary = {
        "rows": len(rows),
        "avg_nbest": nbest_total / max(len(rows), 1),
        "exact_ref_in_nbest": exact_ref_in_nbest,
        "rows_with_consensus_spans": with_consensus,
        "rows_with_confusable_candidates": with_confusables,
        "output": str(path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--max-pinyin", type=int, default=5)
    parser.add_argument("--max-confusables", type=int, default=6)
    parser.add_argument("--consensus-threshold", type=float, default=0.75)
    parser.add_argument("--max-stable-spans", type=int, default=8)
    parser.add_argument("--max-uncertain-spans", type=int, default=8)
    parser.add_argument("--max-span-variants", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-source-tags", action="store_true")
    parser.add_argument("--source-aware-guidance", action="store_true")
    parser.add_argument("--source-aware-strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    written = write_jsonl(output, convert_records(args))
    summary_path = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.json")
    summary = summarize(output, summary_path)
    print(json.dumps({"written": written, "summary": summary, "summary_output": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
