#!/usr/bin/env python
"""Bridge CB-Whisper evidence JSONL to covo/Qwen text-rewrite inference."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


SYSTEM_MESSAGE = (
    "你是一个保守的中文 ASR 后纠错器。根据 CB-Whisper 输出、N-best 候选、"
    "KWS 热词、拼音和候选分数，只做必要的最小修改。必须只输出一个合法 JSON 对象，"
    "格式为 {\"text\":\"...\"}。不要输出解释、推理过程、Markdown 或额外字段。"
)

SELECTOR_SYSTEM_MESSAGE = (
    "你是一个中文 ASR N-best 候选选择器。你的首要任务是比较 ASR top-1 与所有 N-best 候选，"
    "选择最可信、最完整、最符合音频线索和热词证据的中文转写。不要默认保守复制 top-1。"
    "必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
    "不要输出解释、推理过程、Markdown 或额外字段。"
)

CONTENT_SELECTOR_SYSTEM_MESSAGE = (
    "你是一个中文 ASR N-best 候选选择器。你的首要任务是选择主体内容、领域词、专名和数字最可靠的中文转写。"
    "语气词和口语填充词不是高优先级纠错目标；不要为了补全或删除语气词而破坏已经正确的主要内容。"
    "必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
    "不要输出解释、推理过程、Markdown 或额外字段。"
)

SPOKEN_SELECTOR_SYSTEM_MESSAGE = (
    "你是一个中文课堂讲授 ASR N-best 候选选择器。你的首要任务是选择最符合口语课堂转写风格、"
    "同时保留领域词和主要语义的完整中文转写。不要把自然口语改写成书面摘要。"
    "必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
    "不要输出解释、推理过程、Markdown 或额外字段。"
)

INSTRUCTION = (
    "任务：融合 CB-Whisper 的最终输出、N-best 候选和 KWS 热词证据进行中文 ASR 后纠错。"
    "优先保留 CB-Whisper 输出；只有当 N-best、拼音或高置信热词共同支持时才修改。"
    "N-best 中 trusted_scored 候选比 supplemental_unscored 候选更可信；"
    "补充候选只能作为纠错线索，不能单独推翻 ASR top-1 或高分 trusted_scored 候选。"
    "如果 ASR top-1 或 trusted_scored 候选已经包含 prompt 热词，不要把该热词改成同音常见词，"
    "除非多个可信候选和上下文都明确支持替换。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；它用于保护专名/领域词，"
    "但不能因为热词分数高就强行插入无上下文支持的词。"
    "如果证据不足，保持 ASR top-1 不变。"
)

SELECTOR_INSTRUCTION = (
    "任务：从 CB-Whisper 的 ASR top-1、N-best 候选、拼音和 KWS 热词证据中选择最可信的完整中文转写。"
    "ASR top-1 只是候选之一，不是默认答案；必须逐条比较 N-best。"
    "如果非 top-1 候选更完整、少漏字少多字、拼音更接近、上下文更顺，或正确包含受支持热词，"
    "应优先直接采用该非 top-1 候选。"
    "N-best 中 trusted_scored 候选通常可信，但 score 不是唯一依据；"
    "要比较候选是否完整、是否多字漏字、是否包含无关热词、是否符合上下文。"
    "当多个候选都合理时，优先选择字符级错误最少且保留自然口语成分的候选。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；不要因为热词分数高就强行插入无上下文支持的词。"
    "输出应尽量等于某个高质量 N-best 候选；不要为了保守而复制 top-1。"
    "只有所有候选都存在明显局部错字时，才在最佳候选基础上做小幅修正。"
)

CONTENT_SELECTOR_INSTRUCTION = (
    "任务：从 CB-Whisper 的 ASR top-1、N-best 候选、拼音和 KWS 热词证据中选择主体内容最可信的完整中文转写。"
    "ASR top-1 只是候选之一，但如果它的领域词、数字、专名和主要语义已经正确，不要为了语气词差异而改动它。"
    "语气词和口语填充词（如：呢、啊、嗯、呃、那么、这个、这一个、的话、就是）是低优先级线索。"
    "如果多个候选只在这些语气词上不同，优先选择领域词、数字、专名和句子主体更稳定的候选。"
    "不要主动插入语气词；也不要为了删除语气词而改变主体内容。"
    "如果非 top-1 候选明显补全了缺失的主体内容，或修正了领域词/数字/专名，应采用该候选。"
    "受支持的水利/施工领域词优先级高于语气词，例如水利工程、地基、水闸、施工、运输、浇筑、导流、坝体等。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；不要强行插入无上下文支持的词，也不要把已正确出现的领域词改成同音常见词。"
    "输出应尽量等于某个高质量 N-best 候选；只有候选存在明显局部错字时才做小幅修正。"
)

SPOKEN_SELECTOR_INSTRUCTION = (
    "任务：从 CB-Whisper 的 ASR top-1、N-best 候选、拼音和 KWS 热词证据中选择最可信的课堂口语转写。"
    "ASR top-1 只是候选之一，但输出必须保持课堂讲授的口语风格，不要主动书面化、摘要化或删去自然停顿词。"
    "如果 top-1 或多个可信 N-best 候选中包含自然口语词（如：呢、啊、嗯、呃、那么、这个、这一个、的话、就是、咱们、我们），"
    "且这些词不破坏语义，应优先保留。不要为了让句子更书面、更简洁而删除这些词。"
    "当多个候选主体内容相近时，优先选择既保留领域词/数字/专名，又保留自然口语成分的候选。"
    "如果候选之间只在明显无意义重复、乱码或污染尾巴上不同，可以选择更干净的候选；但普通课堂语气词不是污染。"
    "受支持的水利/施工领域词仍然必须优先保留，不要把水利工程、施工、浇筑、运输、地基、基坑、闸门、导流、坝体等改成同音常见词。"
    "热词证据来自 CB-Whisper/KWS，不是参考答案；不要强行插入无上下文支持的词，也不要把已正确出现的领域词改成同音常见词。"
    "输出应尽量等于某个高质量 N-best 候选；只有候选存在明显局部错字时才做小幅修正。"
)

PROTECTED_HOTWORD_INSTRUCTION = (
    "受保护热词是已经出现在 ASR top-1 或 trusted_scored 候选中的 prompt 热词。"
    "最终输出必须逐字保留受保护热词；不要把它改成同音、近音、繁简异体或更常见写法。"
    "如果受保护热词附近还有其他明显 ASR 错误，只能修改热词之外的字符。"
)

CHINESEHP_EVIDENCE_INSTRUCTION = (
    "下面额外给出 N-best 共识片段和易混候选。Stable spans 是多数候选一致支持的内容，"
    "应优先保留；Uncertain spans 是候选分歧大的位置，应结合 variants、拼音、上下文和热词证据判断。"
    "Confusable candidates 是与 top-1 很像但局部不同的候选，可能包含正确改法，也可能是同音误导。"
)

HOTWORD_AWARE_EVIDENCE_INSTRUCTION = (
    "下面额外给出结构化热词证据。protected_hotwords 必须保留；prompt_hotwords 是本轮提示热词；"
    "kws_hotwords 是 KWS 预测热词，可能包含误报。variant_keeps_hotword 表示该候选保留的热词，"
    "variant_drops_hotword 表示该候选相对 top-1/受保护集合丢掉的热词；"
    "false_hotword_warning=yes 表示候选可能只是被热词误导，不能仅凭热词出现就采用。"
)

SAME_LENGTH_PRIOR_INSTRUCTION = (
    "长度约束：除非多个高质量 N-best 候选都明确支持漏字或多字，最终输出在去除标点和空格后"
    "应尽量与 ASR top-1 保持相同字数。优先做等长的同音/近音字替换，不要主动扩写、删减或改写句子。"
)

COMPACT_EVIDENCE_NOTE = (
    "证据说明：下面是压缩后的候选证据。若 ASR top-1 分数明显最高、句子完整，且没有候选支持的热词冲突，"
    "应优先保持 top-1；不要只因为拼音相同或某个 KWS 热词高分就改成低分同音候选。"
)


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


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


_PUNCT_RE = re.compile(r"[\s,，。.!！？?；;：:“”\"'‘’、（）()\[\]【】《》<>-]+")


def simplify_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    if _OPENCC is not None:
        value = _OPENCC.convert(value)
    return value


def normalize_text(text: Any) -> str:
    value = simplify_text(text)
    return _PUNCT_RE.sub("", value)


def _repetition_ratio(text: str) -> float:
    key = normalize_text(text)
    if not key:
        return 1.0
    counts = Counter(key)
    return max(counts.values()) / max(len(key), 1)


def _unique_texts(values: Iterable[Any]) -> List[Tuple[str, str]]:
    output: List[Tuple[str, str]] = []
    seen = set()
    for value in values:
        raw = simplify_text(value)
        key = normalize_text(raw)
        if raw and key and key not in seen:
            output.append((raw, key))
            seen.add(key)
    return output


def clean_nbest_for_prompt(
    nbest: List[Any],
    max_nbest: int,
    min_ratio: float,
    max_ratio: float,
    length_slack: int,
    anchor_suffix_filter: bool,
    drop_polluted_top1: bool,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    unique = _unique_texts(nbest)
    if len(unique) <= 1:
        return [item[0] for item in unique[:max_nbest]], []

    pollution_markers = (
        "请不吝点赞", "订阅", "转发", "打赏", "明镜", "点点栏目",
        "优优独播", "YoYo", "Television", "Exclusive", "Series",
    )

    def severe_reason(raw: str, key: str) -> str:
        if any(marker in raw for marker in pollution_markers):
            return "bad_phrase"
        if "�" in raw:
            return "replacement_char"
        latin_alpha = sum(1 for ch in raw if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        if latin_alpha >= 8 and latin_alpha / max(len(raw), 1) > 0.20:
            return "latin_tail"
        if _repetition_ratio(raw) > 0.45 and len(key) >= 8:
            return "repeat_heavy"
        return ""

    dropped: List[Dict[str, Any]] = []
    severe_clean: List[Tuple[str, str, int]] = []
    for rank, (raw, key) in enumerate(unique, start=1):
        reason = severe_reason(raw, key)
        if reason and (rank > 1 or drop_polluted_top1):
            dropped.append({"rank": rank, "text": raw, "reason": reason, "norm_len": len(key)})
            continue
        severe_clean.append((raw, key, rank))
    if not severe_clean:
        raw, key = unique[0]
        severe_clean = [(raw, key, 1)]

    lengths = sorted(len(key) for _, key, _ in severe_clean if key)
    median_len = lengths[len(lengths) // 2] if lengths else len(unique[0][1])
    min_len = max(2, int(median_len * float(min_ratio)))
    max_len = max(int(median_len * float(max_ratio)), median_len + max(0, int(length_slack)))
    shortest_key = min((key for _, key, _ in severe_clean if key), key=len, default="")
    use_short_anchor = (
        bool(anchor_suffix_filter)
        and len(shortest_key) >= 8
        and len(shortest_key) >= int(max(1, median_len) * 0.55)
    )

    kept: List[str] = []
    for raw, key, rank in severe_clean:
        key_len = len(key)
        reason = ""
        if key_len < min_len:
            reason = "too_short"
        elif key_len > max_len:
            reason = "too_long"
        elif (
            use_short_anchor
            and key != shortest_key
            and key.startswith(shortest_key)
            and key_len - len(shortest_key) >= max(5, int(length_slack) // 2)
        ):
            reason = "short_anchor_long_suffix"
        elif _repetition_ratio(raw) > 0.45 and key_len >= 8:
            reason = "repeat_heavy"
        else:
            latin_alpha = sum(1 for ch in raw if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
            if latin_alpha >= 8 and latin_alpha / max(len(raw), 1) > 0.25:
                reason = "latin_tail"

        if reason:
            dropped.append({"rank": rank, "text": raw, "reason": reason, "norm_len": key_len})
            continue
        kept.append(raw)
        if len(kept) >= max_nbest:
            break
    if not kept:
        kept = [unique[0][0]]
    return kept, dropped


def _char_distance(left: str, right: str) -> int:
    a = list(normalize_text(left))
    b = list(normalize_text(right))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + int(ca != cb),
            )
        previous = current
    return int(previous[-1])


def _diff_summary(base: str, candidate: str, max_parts: int = 4) -> str:
    base_norm = normalize_text(base)
    cand_norm = normalize_text(candidate)
    parts: List[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=base_norm, b=cand_norm).get_opcodes():
        if tag == "equal":
            continue
        src = base_norm[i1:i2] or "∅"
        dst = cand_norm[j1:j2] or "∅"
        parts.append(f"{i1}:{i2} {src}->{dst}")
        if len(parts) >= max_parts:
            break
    return "; ".join(parts)


def _span_variant(base: str, candidate: str, start: int, end: int) -> str:
    base_norm = normalize_text(base)
    cand_norm = normalize_text(candidate)
    pieces: List[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=base_norm, b=cand_norm).get_opcodes():
        if tag == "insert":
            if start <= i1 <= end:
                pieces.append(cand_norm[j1:j2])
            continue
        if i2 <= start or i1 >= end:
            continue
        if tag == "equal":
            overlap_start = max(start, i1)
            overlap_end = min(end, i2)
            cand_start = j1 + (overlap_start - i1)
            cand_end = j1 + (overlap_end - i1)
            pieces.append(cand_norm[cand_start:cand_end])
        else:
            pieces.append(cand_norm[j1:j2])
    return "".join(pieces)


def _merge_support_spans(
    base: str,
    support_ratios: List[float],
    threshold: float,
    stable: bool,
    max_spans: int,
) -> List[Dict[str, Any]]:
    base_norm = normalize_text(base)
    spans: List[Dict[str, Any]] = []
    start = None
    values: List[float] = []
    for idx, ratio in enumerate(support_ratios + [2.0 if stable else -1.0]):
        selected = (ratio >= threshold) if stable else (ratio < threshold)
        if selected and start is None:
            start = idx
            values = [ratio]
        elif selected:
            values.append(ratio)
        elif start is not None:
            end = idx
            text = base_norm[start:end]
            if text:
                spans.append({
                    "start": int(start),
                    "end": int(end),
                    "text": text,
                    "support": float(sum(values) / max(len(values), 1)),
                })
            start = None
            values = []
    spans.sort(key=lambda row: (-(int(row["end"]) - int(row["start"])), int(row["start"])))
    return spans[: max(0, int(max_spans))]


def build_nbest_consensus(
    nbest: List[str],
    threshold: float,
    max_stable: int,
    max_uncertain: int,
    max_variants: int,
) -> Dict[str, Any]:
    cleaned = [str(item).strip() for item in nbest if str(item).strip()]
    if not cleaned:
        return {"threshold": threshold, "stable_spans": [], "uncertain_spans": []}
    base = normalize_text(cleaned[0])
    if not base:
        return {"threshold": threshold, "stable_spans": [], "uncertain_spans": []}

    equal_counts = [0 for _ in base]
    normalized_candidates = [normalize_text(item) for item in cleaned if normalize_text(item)]
    for candidate in normalized_candidates:
        for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=base, b=candidate).get_opcodes():
            if tag != "equal":
                continue
            for idx in range(i1, i2):
                if 0 <= idx < len(equal_counts):
                    equal_counts[idx] += 1
    denom = max(1, len(normalized_candidates))
    support_ratios = [count / denom for count in equal_counts]
    stable = _merge_support_spans(base, support_ratios, threshold, True, max_stable)
    uncertain = _merge_support_spans(base, support_ratios, threshold, False, max_uncertain)
    enriched_uncertain = []
    for span in uncertain:
        variants = []
        seen = set()
        for candidate in normalized_candidates:
            variant = _span_variant(base, candidate, int(span["start"]), int(span["end"]))
            if variant and variant not in seen:
                variants.append(variant)
                seen.add(variant)
            if len(variants) >= max(1, int(max_variants)):
                break
        enriched_uncertain.append({**span, "variants": variants})
    return {
        "threshold": float(threshold),
        "stable_spans": stable,
        "uncertain_spans": enriched_uncertain,
    }


def _hotword_texts(rows: List[Dict[str, Any]], prompt_only: bool = False) -> List[str]:
    output = []
    seen = set()
    for item in rows:
        if prompt_only and not bool(item.get("in_prompt", False)):
            continue
        text = str(item.get("text", "")).strip()
        key = normalize_text(text)
        if text and key and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def _hotword_delta_for_text(
    text: str,
    top1: str,
    hotword_rows: List[Dict[str, Any]],
    protected_hotwords: List[str],
) -> Dict[str, Any]:
    text_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=False)
    prompt_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=True)
    top_hits = _matched_hotword_texts(top1, hotword_rows, prompt_only=False)
    protected_norms = {normalize_text(item) for item in protected_hotwords if normalize_text(item)}
    text_norms = {normalize_text(item) for item in text_hits}
    top_norms = {normalize_text(item) for item in top_hits}

    dropped = []
    for item in _hotword_texts(hotword_rows, prompt_only=False):
        key = normalize_text(item)
        if not key:
            continue
        if (key in top_norms or key in protected_norms) and key not in text_norms:
            dropped.append(item)

    inserted = []
    for item in text_hits:
        key = normalize_text(item)
        if key and key not in top_norms and key not in protected_norms:
            inserted.append(item)

    protected_dropped = [
        item for item in protected_hotwords
        if normalize_text(item) and normalize_text(item) not in text_norms
    ]
    false_warning = bool(inserted) and not bool(prompt_hits)
    return {
        "variant_keeps_hotword": text_hits,
        "variant_drops_hotword": dropped,
        "protected_dropped": protected_dropped,
        "inserted_hotwords": inserted,
        "false_hotword_warning": false_warning,
    }


def _variant_hotword_delta(
    variant: str,
    span_text: str,
    hotword_rows: List[Dict[str, Any]],
    protected_hotwords: List[str],
) -> Dict[str, Any]:
    variant_norm = normalize_text(variant)
    span_norm = normalize_text(span_text)
    keeps = []
    drops = []
    inserted = []
    for item in _hotword_texts(hotword_rows, prompt_only=False):
        key = normalize_text(item)
        if not key:
            continue
        in_variant = key in variant_norm
        in_span = key in span_norm
        if in_variant:
            keeps.append(item)
            if not in_span:
                inserted.append(item)
        elif in_span:
            drops.append(item)
    return {
        "text": variant,
        "variant_keeps_hotword": keeps,
        "variant_drops_hotword": drops,
        "false_hotword_warning": bool(inserted),
    }


def format_consensus_spans(
    consensus: Dict[str, Any],
    hotword_rows: List[Dict[str, Any]] | None = None,
    protected_hotwords: List[str] | None = None,
    compact: bool = False,
) -> List[str]:
    output: List[str] = []
    hotword_rows = hotword_rows or []
    protected_hotwords = protected_hotwords or []
    stable = list(consensus.get("stable_spans", []) or [])
    uncertain = list(consensus.get("uncertain_spans", []) or [])
    if stable:
        output.append("Stable spans:")
        for item in stable:
            output.append(
                f"- {int(item.get('start', 0))}:{int(item.get('end', 0))} "
                f"{item.get('text', '')} support={float(item.get('support', 0.0)):.3f}"
            )
    if uncertain:
        output.append("Uncertain spans:")
        for item in uncertain:
            variants = list(item.get("variants", []) or [])
            suffix = ""
            if variants:
                if compact:
                    suffix = " variants=" + json.dumps(variants, ensure_ascii=False, separators=(",", ":"))
                elif hotword_rows:
                    variants_for_prompt = [
                        _variant_hotword_delta(
                            variant=str(variant),
                            span_text=str(item.get("text", "")),
                            hotword_rows=hotword_rows,
                            protected_hotwords=protected_hotwords,
                        )
                        for variant in variants
                    ]
                    suffix = " variants=" + json.dumps(variants_for_prompt, ensure_ascii=False, separators=(",", ":"))
                else:
                    variants_for_prompt = variants
                    suffix = " variants=" + json.dumps(variants_for_prompt, ensure_ascii=False, separators=(",", ":"))
            output.append(
                f"- {int(item.get('start', 0))}:{int(item.get('end', 0))} "
                f"{item.get('text', '')} support={float(item.get('support', 0.0)):.3f}{suffix}"
            )
    return output


def format_confusable_candidates(
    nbest: List[str],
    hotword_rows: List[Dict[str, Any]],
    protected_hotwords: List[str],
    max_items: int,
) -> List[str]:
    if max_items <= 0 or len(nbest) <= 1:
        return []
    top1 = str(nbest[0]).strip()
    rows = []
    seen = {normalize_text(top1)}
    for rank, candidate in enumerate(nbest[1:], start=2):
        text = str(candidate).strip()
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        sim = SequenceMatcher(a=normalize_text(top1), b=key).ratio()
        if sim < 0.55:
            continue
        distance = _char_distance(top1, text)
        if distance <= 0:
            continue
        prompt_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=True)
        top_prompt_hits = _matched_hotword_texts(top1, hotword_rows, prompt_only=True)
        delta = _hotword_delta_for_text(
            text=text,
            top1=top1,
            hotword_rows=hotword_rows,
            protected_hotwords=protected_hotwords,
        )
        rows.append((distance, -sim, rank, text, prompt_hits, top_prompt_hits, delta))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    output = []
    for distance, neg_sim, rank, text, prompt_hits, top_prompt_hits, delta in rows[:max_items]:
        keeps = ",".join(prompt_hits) if prompt_hits else "none"
        top_keeps = ",".join(top_prompt_hits) if top_prompt_hits else "none"
        delta_json = json.dumps(delta, ensure_ascii=False, separators=(",", ":"))
        output.append(
            f"- cand#{rank} sim={-neg_sim:.3f} dist={distance}: {text} | "
            f"diff={_diff_summary(top1, text)} keeps_prompt_hotwords={keeps} "
            f"top1_keeps={top_keeps} hotword_delta={delta_json}"
        )
    return output


def _candidate_score_index(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        text = str(item.get("text", "")).strip()
        key = normalize_text(text)
        if key and key not in indexed:
            indexed[key] = item
    return indexed


def build_context_hotwords(input_block: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    source_mode = str(getattr(args, "hotword_source", "prompt")).strip().lower()
    use_kws = source_mode in {"all", "kws"}
    use_prompt = source_mode in {"all", "prompt"}
    for rank, item in enumerate(list(input_block.get("hotwords", []) or [])[: int(args.max_hotwords)], 1):
        if not use_kws:
            continue
        text = simplify_text(item.get("text", ""))
        if not text:
            continue
        row = merged.setdefault(
            text,
            {
                "text": text,
                "score": 0.0,
                "prompt_weight": 0.0,
                "sources": [],
                "kws_rank": rank,
                "in_prompt": False,
            },
        )
        row["score"] = max(float(row.get("score", 0.0)), _as_float(item.get("score", 0.0)))
        if "kws" not in row["sources"]:
            row["sources"].append("kws")

    for rank, item in enumerate(list(input_block.get("prompt_hotwords", []) or [])[: int(args.max_prompt_hotwords)], 1):
        if not use_prompt:
            continue
        text = simplify_text(item.get("text", ""))
        if not text:
            continue
        row = merged.setdefault(
            text,
            {
                "text": text,
                "score": 0.0,
                "prompt_weight": 0.0,
                "sources": [],
                "kws_rank": 999999,
                "in_prompt": False,
            },
        )
        row["prompt_weight"] = max(float(row.get("prompt_weight", 0.0)), _as_float(item.get("weight", item.get("score", 0.0))))
        row["prompt_rank"] = rank
        row["in_prompt"] = True
        if "prompt" not in row["sources"]:
            row["sources"].append("prompt")

    rows = list(merged.values())
    rows.sort(
        key=lambda row: (
            -int(bool(row.get("in_prompt", False))),
            -float(row.get("score", 0.0)),
            -float(row.get("prompt_weight", 0.0)),
            int(row.get("kws_rank", 999999)),
            -len(str(row.get("text", ""))),
            str(row.get("text", "")),
        )
    )
    return rows[: int(args.max_hotwords)]


def _matched_hotword_texts(text: str, rows: List[Dict[str, Any]], prompt_only: bool = False) -> List[str]:
    normalized = normalize_text(text)
    matched = []
    for item in rows:
        if prompt_only and not bool(item.get("in_prompt", False)):
            continue
        keyword = simplify_text(item.get("text", ""))
        if keyword and normalize_text(keyword) in normalized:
            matched.append(keyword)
    return matched


def build_protected_hotwords(input_block: Dict[str, Any], rows: List[Dict[str, Any]]) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    candidates = list(cbw.get("candidates", []) or [])
    trusted_text = normalize_text(input_block.get("asr_top1", ""))
    scored_support = set()
    for item in candidates:
        trusted_text += normalize_text(item.get("text", ""))
        exact_stats = item.get("exact_stats", {}) or {}
        for keyword in list(exact_stats.get("matched_keywords", []) or []) + list(item.get("consensus_keywords", []) or []):
            normalized_keyword = normalize_text(keyword)
            if normalized_keyword:
                scored_support.add(normalized_keyword)

    protected = []
    seen = set()
    for item in rows:
        if not bool(item.get("in_prompt", False)):
            continue
        keyword = str(item.get("text", "")).strip()
        normalized = normalize_text(keyword)
        if normalized and normalized in trusted_text and normalized in scored_support and normalized not in seen:
            protected.append(keyword)
            seen.add(normalized)
    return protected


def _supported_hotword_sets(rows: List[Dict[str, Any]], nbest: List[str]) -> Tuple[List[str], List[str], List[str]]:
    candidate_text = normalize_text("".join(str(item or "") for item in nbest))
    supported: List[str] = []
    unsupported_prompt: List[str] = []
    unsupported_kws: List[str] = []
    seen_supported = set()
    seen_unsupported = set()
    for item in rows:
        text = simplify_text(item.get("text", ""))
        key = normalize_text(text)
        if not text or not key:
            continue
        if key in candidate_text:
            if key not in seen_supported:
                supported.append(text)
                seen_supported.add(key)
        elif item.get("in_prompt", False):
            if key not in seen_unsupported:
                unsupported_prompt.append(text)
                seen_unsupported.add(key)
        else:
            unsupported_kws.append(text)
    return supported, unsupported_prompt, unsupported_kws


def format_context_hotwords(
    rows: List[Dict[str, Any]],
    nbest: List[str] | None = None,
    compact: bool = False,
) -> List[str]:
    if compact:
        nbest = nbest or []
        supported, unsupported_prompt, _unsupported_kws = _supported_hotword_sets(rows, nbest)
        output = []
        if supported:
            output.append("- supported_hotwords_in_candidates=" + ",".join(supported))
        if unsupported_prompt:
            output.append(
                "- unsupported_prompt_hotwords="
                + ",".join(unsupported_prompt)
                + " (not found in any candidate; do not force)"
            )
        return output

    output = []
    for item in rows:
        text = simplify_text(item.get("text", ""))
        if not text:
            continue
        score = _as_float(item.get("score", 0.0))
        prompt_weight = _as_float(item.get("prompt_weight", 0.0))
        sources = ",".join(str(source) for source in item.get("sources", []) or [])
        in_prompt = "yes" if item.get("in_prompt", False) else "no"
        output.append(
            f"- {text} kws_score={score:.4f} prompt_weight={prompt_weight:.4f} "
            f"in_prompt={in_prompt} source={sources}"
        )
    return output


def format_nbest(
    nbest: List[str],
    input_block: Dict[str, Any],
    hotword_rows: List[Dict[str, Any]],
    max_items: int,
    compact: bool = False,
) -> List[str]:
    cbw = input_block.get("cbwhisper", {}) or {}
    scored_candidates = list(cbw.get("candidates", []) or [])
    scored_index = _candidate_score_index(scored_candidates)
    output = []
    seen = set()
    for idx, hyp in enumerate(nbest[:max_items], 1):
        text = simplify_text(hyp)
        if not text:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        scored = scored_index.get(normalized)
        if scored is None:
            source = "supplemental_unscored"
            score_bits = "score=NA"
        else:
            source = "trusted_scored"
            if compact:
                bits = [
                    f"rank={int(scored.get('rank', idx))}",
                    f"score={float(scored.get('total_score', 0.0)):.3f}",
                    f"asr={float(scored.get('asr_score', 0.0)):.3f}",
                ]
                exact = float(scored.get("exact_score", 0.0))
                phon = float(scored.get("phonetic_score", 0.0))
                if abs(exact) > 1e-6:
                    bits.append(f"exact={exact:.3f}")
                if abs(phon) > 1e-6:
                    bits.append(f"phon={phon:.3f}")
                score_bits = " ".join(bits)
            else:
                score_bits = (
                    f"score_rank={int(scored.get('rank', idx))} "
                    f"total={float(scored.get('total_score', 0.0)):.4f} "
                    f"asr={float(scored.get('asr_score', 0.0)):.4f} "
                    f"exact={float(scored.get('exact_score', 0.0)):.4f} "
                    f"phon={float(scored.get('phonetic_score', 0.0)):.4f}"
                )
        prompt_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=True)
        kws_hits = _matched_hotword_texts(text, hotword_rows, prompt_only=False)
        if compact:
            hit_bits = []
            if prompt_hits:
                hit_bits.append("prompt_hw=" + ",".join(prompt_hits))
            context_only_hits = [item for item in kws_hits if item not in set(prompt_hits)]
            if context_only_hits:
                hit_bits.append("kws_hw=" + ",".join(context_only_hits))
            hit_msg = (" " + " ".join(hit_bits)) if hit_bits else ""
            output.append(f"{idx}. {text} | {source} {score_bits}{hit_msg}")
        else:
            prompt_msg = ",".join(prompt_hits) if prompt_hits else "none"
            kws_msg = ",".join(kws_hits) if kws_hits else "none"
            output.append(
                f"{idx}. {text} | source={source} {score_bits} "
                f"keeps_prompt_hotwords={prompt_msg} keeps_context_hotwords={kws_msg}"
            )
    return output


def build_hotword_aware_evidence(
    nbest: List[str],
    hotword_rows: List[Dict[str, Any]],
    protected_hotwords: List[str],
    max_items: int,
) -> Dict[str, Any]:
    top1 = str(nbest[0]).strip() if nbest else ""
    prompt_hotwords = _hotword_texts(hotword_rows, prompt_only=True)
    kws_hotwords = _hotword_texts(hotword_rows, prompt_only=False)
    rows = []
    conflicts = []
    for idx, text in enumerate(nbest[: max(1, int(max_items))], 1):
        delta = _hotword_delta_for_text(
            text=str(text),
            top1=top1,
            hotword_rows=hotword_rows,
            protected_hotwords=protected_hotwords,
        )
        row = {
            "candidate": int(idx),
            "variant_keeps_hotword": delta["variant_keeps_hotword"],
            "variant_drops_hotword": delta["variant_drops_hotword"],
            "protected_dropped": delta["protected_dropped"],
            "inserted_hotwords": delta["inserted_hotwords"],
            "false_hotword_warning": bool(delta["false_hotword_warning"]),
        }
        rows.append(row)
        if row["protected_dropped"] or row["inserted_hotwords"] or row["false_hotword_warning"]:
            conflicts.append(row)
    return {
        "prompt_hotwords": prompt_hotwords,
        "kws_hotwords": kws_hotwords,
        "protected_hotwords": protected_hotwords,
        "hotword_conflict": conflicts,
        "candidate_hotword_status": rows,
    }


def format_hotword_aware_evidence(evidence: Dict[str, Any], compact: bool = False) -> List[str]:
    if compact:
        protected = list(evidence.get("protected_hotwords", []) or [])
        prompt = list(evidence.get("prompt_hotwords", []) or [])
        kws = list(evidence.get("kws_hotwords", []) or [])
        rows = list(evidence.get("candidate_hotword_status", []) or [])
        supported_norms = set()
        for row in rows:
            for item in list(row.get("variant_keeps_hotword", []) or []):
                key = normalize_text(item)
                if key:
                    supported_norms.add(key)
        protected_norms = {normalize_text(item) for item in protected if normalize_text(item)}
        unsupported_prompt = [
            item for item in prompt
            if normalize_text(item) and normalize_text(item) not in supported_norms and normalize_text(item) not in protected_norms
        ]
        supported = [
            item for item in kws
            if normalize_text(item) and normalize_text(item) in supported_norms
        ]
        output = ["Hotword evidence:"]
        if protected:
            output.append("- protected_hotwords=" + ",".join(protected))
        if supported:
            output.append("- candidate_supported_hotwords=" + ",".join(supported))
        if unsupported_prompt:
            output.append(
                "- unsupported_prompt_hotwords="
                + ",".join(unsupported_prompt)
                + " (not found in candidates; do not force)"
            )
        conflicts = list(evidence.get("hotword_conflict", []) or [])
        compact_conflicts = [
            row for row in conflicts
            if row.get("protected_dropped") or row.get("inserted_hotwords") or row.get("false_hotword_warning")
        ]
        if compact_conflicts:
            output.append(
                "- conflicts="
                + json.dumps(compact_conflicts, ensure_ascii=False, separators=(",", ":"))
            )
        if len(output) == 1:
            output.append("- no candidate-supported hotword evidence")
        return output

    output = ["Hotword-aware evidence:"]
    for key in ("protected_hotwords", "prompt_hotwords", "kws_hotwords"):
        output.append(f"- {key}=" + json.dumps(evidence.get(key, []), ensure_ascii=False, separators=(",", ":")))
    conflicts = list(evidence.get("hotword_conflict", []) or [])
    output.append(
        "- hotword_conflict="
        + (json.dumps(conflicts, ensure_ascii=False, separators=(",", ":")) if conflicts else "none")
    )
    rows = list(evidence.get("candidate_hotword_status", []) or [])
    if rows:
        output.append("Candidate hotword status:")
        for row in rows:
            output.append(
                f"- cand#{int(row.get('candidate', 0))} "
                f"variant_keeps_hotword={json.dumps(row.get('variant_keeps_hotword', []), ensure_ascii=False, separators=(',', ':'))} "
                f"variant_drops_hotword={json.dumps(row.get('variant_drops_hotword', []), ensure_ascii=False, separators=(',', ':'))} "
                f"protected_dropped={json.dumps(row.get('protected_dropped', []), ensure_ascii=False, separators=(',', ':'))} "
                f"inserted_hotwords={json.dumps(row.get('inserted_hotwords', []), ensure_ascii=False, separators=(',', ':'))} "
                f"false_hotword_warning={'yes' if row.get('false_hotword_warning', False) else 'no'}"
            )
    return output


def format_candidates(candidates: List[Dict[str, Any]], max_items: int) -> List[str]:
    output = []
    for item in candidates[:max_items]:
        text = simplify_text(item.get("text", ""))
        if not text:
            continue
        rank = int(item.get("rank", len(output) + 1))
        total = float(item.get("total_score", 0.0))
        exact = float(item.get("exact_score", 0.0))
        phon = float(item.get("phonetic_score", 0.0))
        asr = float(item.get("asr_score", 0.0))
        output.append(
            f"{rank}. {text} | total={total:.4f} asr={asr:.4f} exact={exact:.4f} phon={phon:.4f}"
        )
    return output


def build_user_prompt(record: Dict[str, Any], args: argparse.Namespace) -> str:
    input_block = record.get("input", {}) or {}
    compact = bool(getattr(args, "compact_evidence", True))
    prompt_mode = str(getattr(args, "prompt_mode", "correction")).strip().lower()
    if prompt_mode == "selector_spoken":
        lines = [SPOKEN_SELECTOR_INSTRUCTION]
    elif prompt_mode == "selector_content":
        lines = [CONTENT_SELECTOR_INSTRUCTION]
    elif prompt_mode == "selector":
        lines = [SELECTOR_INSTRUCTION]
    else:
        lines = [INSTRUCTION]
    if compact:
        lines.append(COMPACT_EVIDENCE_NOTE)
    if bool(getattr(args, "prefer_same_length", False)):
        lines.append(SAME_LENGTH_PRIOR_INSTRUCTION)
    if bool(getattr(args, "protect_supported_hotwords", False)) and not compact:
        lines.append(PROTECTED_HOTWORD_INSTRUCTION)
    if (bool(getattr(args, "include_consensus_spans", False)) or int(getattr(args, "max_confusables", 0)) > 0) and not compact:
        lines.append(CHINESEHP_EVIDENCE_INSTRUCTION)
    if bool(getattr(args, "include_hotword_evidence", False)) and not compact:
        lines.append(HOTWORD_AWARE_EVIDENCE_INSTRUCTION)
    asr_top1 = simplify_text(input_block.get("asr_top1", ""))
    lines.append(f"ASR top-1: {asr_top1}")
    hotword_rows = build_context_hotwords(input_block, args)
    protected_hotwords = build_protected_hotwords(input_block, hotword_rows)
    if bool(getattr(args, "protect_supported_hotwords", False)) and protected_hotwords:
        lines.append("Protected hotwords that must be preserved exactly: " + ",".join(protected_hotwords))
    if hotword_rows:
        asr_prompt_hits = _matched_hotword_texts(asr_top1, hotword_rows, prompt_only=True)
        if asr_prompt_hits or not compact:
            hit_msg = ",".join(asr_prompt_hits) if asr_prompt_hits else "none"
            lines.append(f"ASR top-1 keeps prompt hotwords: {hit_msg}")

    nbest = list(input_block.get("nbest", []) or [])[: max(1, int(args.max_nbest))]
    if nbest:
        lines.append(
            "N-best with reliability labels "
            "(trusted_scored=CB-Whisper scored beam, supplemental_unscored=extra diversity candidate):"
        )
        lines.extend(format_nbest(nbest, input_block, hotword_rows, max_items=max(1, int(args.max_nbest)), compact=compact))

    if bool(getattr(args, "include_consensus_spans", False)) and nbest:
        consensus = input_block.get("nbest_consensus")
        if not isinstance(consensus, dict):
            consensus = build_nbest_consensus(
                nbest=nbest,
                threshold=float(getattr(args, "consensus_span_threshold", 0.75)),
                max_stable=int(getattr(args, "max_stable_spans", 8)),
                max_uncertain=int(getattr(args, "max_uncertain_spans", 8)),
                max_variants=int(getattr(args, "max_span_variants", 6)),
            )
        consensus_lines = format_consensus_spans(
            consensus,
            hotword_rows=hotword_rows if bool(getattr(args, "include_hotword_evidence", False)) else None,
            protected_hotwords=protected_hotwords,
            compact=compact,
        )
        if consensus_lines:
            lines.extend(consensus_lines)

    if bool(getattr(args, "include_hotword_evidence", False)) and nbest:
        evidence = input_block.get("hotword_aware_evidence")
        if not isinstance(evidence, dict):
            evidence = build_hotword_aware_evidence(
                nbest=nbest,
                hotword_rows=hotword_rows,
                protected_hotwords=protected_hotwords,
                max_items=max(1, int(args.max_nbest)),
            )
        lines.extend(format_hotword_aware_evidence(evidence, compact=compact))

    confusable_lines = format_confusable_candidates(
        nbest=nbest,
        hotword_rows=hotword_rows,
        protected_hotwords=protected_hotwords,
        max_items=int(getattr(args, "max_confusables", 0)),
    )
    if confusable_lines:
        lines.append("Confusable candidates:")
        lines.extend(confusable_lines)

    if bool(args.include_pinyin):
        pinyin = list(input_block.get("nbest_pinyin", []) or [])[: max(1, int(args.max_pinyin))]
        if pinyin:
            pinyin_keys = []
            seen_pinyin = set()
            for item in pinyin:
                key = re.sub(r"\s+", " ", str(item or "").strip())
                if key and key not in seen_pinyin:
                    pinyin_keys.append(key)
                    seen_pinyin.add(key)
            if compact and len(pinyin_keys) == 1:
                lines.append("Pinyin: all listed candidates share " + pinyin_keys[0])
            else:
                lines.append("Pinyin:")
                for idx, item in enumerate(pinyin, 1):
                    lines.append(f"{idx}. {item}")

    hotword_lines = (
        []
        if (compact and bool(getattr(args, "include_hotword_evidence", False)))
        else format_context_hotwords(hotword_rows, nbest=nbest, compact=compact)
    )
    if hotword_lines:
        lines.append("CB-Whisper hotword evidence (predicted, not gold):")
        lines.extend(hotword_lines)

    if not compact:
        cbw = input_block.get("cbwhisper", {}) or {}
        candidates = list(cbw.get("candidates", []) or [])
        candidate_lines = format_candidates(candidates, max_items=int(args.max_candidates_with_scores))
        if candidate_lines:
            lines.append("CB-Whisper candidate scores:")
            lines.extend(candidate_lines)

    lines.append('请输出 JSON：{"text":"纠错后的完整句子"}')
    return "\n".join(lines)


def _simplify_hotword_rows(rows: List[Any]) -> List[Any]:
    output: List[Any] = []
    for item in rows:
        if isinstance(item, dict):
            copied = dict(item)
            copied["text"] = simplify_text(copied.get("text", ""))
            if copied["text"]:
                output.append(copied)
        else:
            text = simplify_text(item)
            if text:
                output.append(text)
    return output


def _simplify_candidates(rows: List[Any]) -> List[Any]:
    output: List[Any] = []
    for item in rows:
        if not isinstance(item, dict):
            output.append(item)
            continue
        copied = copy.deepcopy(item)
        copied["text"] = simplify_text(copied.get("text", ""))
        exact_stats = copied.get("exact_stats")
        if isinstance(exact_stats, dict):
            exact_stats["matched_keywords"] = [
                simplify_text(keyword)
                for keyword in list(exact_stats.get("matched_keywords", []) or [])
                if simplify_text(keyword)
            ]
        copied["consensus_keywords"] = [
            simplify_text(keyword)
            for keyword in list(copied.get("consensus_keywords", []) or [])
            if simplify_text(keyword)
        ]
        output.append(copied)
    return output


def simplify_input_block(input_block: Dict[str, Any]) -> Dict[str, Any]:
    simplified = copy.deepcopy(input_block)
    if "asr_top1" in simplified:
        simplified["asr_top1"] = simplify_text(simplified.get("asr_top1", ""))
    if "nbest" in simplified:
        simplified["nbest"] = [
            simplify_text(item)
            for item in list(simplified.get("nbest", []) or [])
            if simplify_text(item)
        ]
    if "hotwords" in simplified:
        simplified["hotwords"] = _simplify_hotword_rows(list(simplified.get("hotwords", []) or []))
    if "prompt_hotwords" in simplified:
        simplified["prompt_hotwords"] = _simplify_hotword_rows(list(simplified.get("prompt_hotwords", []) or []))
    if "keyword_mentions" in simplified:
        simplified["keyword_mentions"] = _simplify_hotword_rows(list(simplified.get("keyword_mentions", []) or []))
    if "oracle_hotwords" in simplified:
        simplified["oracle_hotwords"] = _simplify_hotword_rows(list(simplified.get("oracle_hotwords", []) or []))
    cbw = simplified.get("cbwhisper")
    if isinstance(cbw, dict) and "candidates" in cbw:
        cbw["candidates"] = _simplify_candidates(list(cbw.get("candidates", []) or []))
    return simplified


def build_system_message(args: argparse.Namespace) -> str:
    prompt_mode = str(getattr(args, "prompt_mode", "correction")).strip().lower()
    if prompt_mode == "selector_spoken":
        return SPOKEN_SELECTOR_SYSTEM_MESSAGE
    if prompt_mode == "selector_content":
        return CONTENT_SELECTOR_SYSTEM_MESSAGE
    return SELECTOR_SYSTEM_MESSAGE if prompt_mode == "selector" else SYSTEM_MESSAGE


def prepare_records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    count = 0
    for record in read_jsonl(args.input):
        reference = str(nested_get(record, args.reference_field, "")).strip()
        input_block = dict(record.get("input", {}) or {})
        if not str(input_block.get("asr_top1", "")).strip():
            asr_top1 = str(record.get("asr_top1", "")).strip()
            if asr_top1:
                input_block["asr_top1"] = asr_top1
        if not input_block.get("nbest") and record.get("nbest"):
            input_block["nbest"] = list(record.get("nbest", []) or [])
        if not input_block.get("nbest") and input_block.get("asr_top1"):
            input_block["nbest"] = [str(input_block.get("asr_top1", "")).strip()]
        if bool(getattr(args, "simplify_input", True)):
            input_block = simplify_input_block(input_block)
        if bool(getattr(args, "clean_nbest", False)):
            original_nbest = list(input_block.get("nbest", []) or [])
            original_pinyin = list(input_block.get("nbest_pinyin", []) or [])
            pinyin_by_key = {
                normalize_text(text): str(original_pinyin[idx])
                for idx, text in enumerate(original_nbest)
                if idx < len(original_pinyin) and normalize_text(text)
            }
            cleaned_nbest, dropped = clean_nbest_for_prompt(
                nbest=original_nbest,
                max_nbest=max(1, int(args.max_nbest)),
                min_ratio=float(args.clean_min_length_ratio),
                max_ratio=float(args.clean_max_length_ratio),
                length_slack=int(args.clean_length_slack),
                anchor_suffix_filter=bool(args.clean_anchor_suffix_filter),
                drop_polluted_top1=bool(args.clean_drop_polluted_top1),
            )
            input_block["nbest"] = cleaned_nbest
            if original_pinyin:
                input_block["nbest_pinyin"] = [
                    pinyin_by_key.get(normalize_text(text), "")
                    for text in cleaned_nbest
                    if pinyin_by_key.get(normalize_text(text), "") != ""
                ]
            input_block["nbest_cleaning"] = {
                "enabled": True,
                "before": len(_unique_texts(original_nbest)),
                "after": len(_unique_texts(cleaned_nbest)),
                "dropped": dropped[:20],
                "dropped_count": len(dropped),
            }
        if bool(getattr(args, "include_consensus_spans", False)):
            input_block["nbest_consensus"] = build_nbest_consensus(
                nbest=list(input_block.get("nbest", []) or []),
                threshold=float(getattr(args, "consensus_span_threshold", 0.75)),
                max_stable=int(getattr(args, "max_stable_spans", 8)),
                max_uncertain=int(getattr(args, "max_uncertain_spans", 8)),
                max_variants=int(getattr(args, "max_span_variants", 6)),
            )
        input_block["covo_hotwords"] = build_context_hotwords(input_block, args)
        if bool(getattr(args, "include_hotword_evidence", False)):
            input_block["hotword_aware_evidence"] = build_hotword_aware_evidence(
                nbest=list(input_block.get("nbest", []) or []),
                hotword_rows=list(input_block.get("covo_hotwords", []) or []),
                protected_hotwords=build_protected_hotwords(
                    input_block,
                    list(input_block.get("covo_hotwords", []) or []),
                ),
                max_items=max(1, int(args.max_nbest)),
            )
        output = {
            "id": str(record.get("id", "")),
            "source": record.get("source", "cbwhisper"),
            "dataset": record.get("dataset", ""),
            "split": record.get("split", ""),
            "reference": reference,
            "input": input_block,
            "messages": [
                {"role": "system", "content": build_system_message(args)},
                {"role": "user", "content": build_user_prompt({**record, "input": input_block}, args)},
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
        count += 1
        if args.limit and count >= int(args.limit):
            break


def cmd_prepare(args: argparse.Namespace) -> int:
    written = write_jsonl(args.output, prepare_records(args))
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


def post_filter_predictions(args: argparse.Namespace) -> int:
    """Conservative post-filter for COVO predictions.

    The Shuili video set is very sensitive to insertion/deletion errors.  This
    filter keeps COVO substitutions that preserve the normalized character
    length of ASR top-1, and otherwise falls back to top-1.  It intentionally
    uses the same light text normalization as the prompt builder: simplify,
    drop punctuation/space, but do not remove fillers or rewrite numbers.
    """
    if not bool(getattr(args, "post_filter_same_length", False)):
        return 0

    prediction_path = Path(args.prediction_output)
    rows = list(read_jsonl(prediction_path))
    kept = 0
    reverted = 0
    unchanged = 0
    for row in rows:
        input_block = row.get("input", {}) or {}
        baseline = simplify_text(input_block.get("asr_top1", ""))
        prediction = simplify_text(row.get("prediction", ""))
        base_key = normalize_text(baseline)
        pred_key = normalize_text(prediction)
        decision = "unchanged"
        if pred_key == base_key:
            unchanged += 1
        elif pred_key and len(pred_key) == len(base_key):
            kept += 1
            decision = "kept_same_length"
        else:
            row["prediction_before_post_filter"] = row.get("prediction", "")
            row["raw_prediction_before_post_filter"] = row.get("raw_prediction", "")
            row["prediction"] = baseline
            row["raw_prediction"] = baseline
            reverted += 1
            decision = "reverted_length_mismatch"
        row["post_filter"] = {
            "same_length": True,
            "decision": decision,
            "baseline_norm_len": len(base_key),
            "prediction_norm_len": len(pred_key),
        }

    write_jsonl(prediction_path, rows)
    print(
        json.dumps(
            {
                "post_filter": "same_length",
                "prediction_output": str(prediction_path),
                "kept": kept,
                "reverted": reverted,
                "unchanged": unchanged,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return kept + reverted + unchanged


def cmd_run(args: argparse.Namespace) -> int:
    cmd_prepare(args)
    covo_dir = Path(args.covo_dir).resolve()
    infer_script = covo_dir / "scripts" / "infer_lora_text.py"
    eval_script = covo_dir / "scripts" / "evaluate_correction_jsonl.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"missing covo infer script: {infer_script}")
    message_path = Path(args.output).resolve()
    prediction_path = Path(args.prediction_output).resolve()

    command = [
        args.python,
        str(infer_script),
        "--input",
        str(message_path),
        "--output",
        str(prediction_path),
        "--model-name-or-path",
        str(args.model_name_or_path),
        "--adapter-path",
        str(args.adapter_path),
        "--batch-size",
        str(args.batch_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--device",
        str(args.device),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.disable_thinking:
        command.append("--disable-thinking")
    if args.infer_limit:
        command.extend(["--limit", str(args.infer_limit)])
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(covo_dir), check=True)
    post_filter_predictions(args)

    if args.evaluate and eval_script.exists():
        eval_command = [
            args.python,
            str(eval_script),
            "--input",
            str(prediction_path),
            "--prediction-field",
            "prediction",
            "--reference-field",
            "reference",
            "--baseline-field",
            "input.asr_top1",
        ]
        print("+ " + " ".join(eval_command), flush=True)
        subprocess.run(eval_command, cwd=str(covo_dir), check=True)
    return 0


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="CB-Whisper evidence JSONL from CBW_EVIDENCE_OUT")
    parser.add_argument("--output", required=True, help="Qwen/covo messages JSONL")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--max-nbest", type=int, default=8)
    parser.add_argument("--max-pinyin", type=int, default=5)
    parser.add_argument("--max-hotwords", type=int, default=12)
    parser.add_argument("--max-prompt-hotwords", type=int, default=6)
    parser.add_argument("--max-candidates-with-scores", type=int, default=8)
    parser.add_argument(
        "--hotword-source",
        choices=["prompt", "kws", "all"],
        default="prompt",
        help="Which CB-Whisper hotwords to inject into covo prompts.",
    )
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        choices=["correction", "selector", "selector_content", "selector_spoken"],
        default="correction",
        help="Prompt style for COVO: conservative correction or n-best selector.",
    )
    parser.add_argument(
        "--protect-supported-hotwords",
        action="store_true",
        help="Add a hard prompt constraint to preserve prompt hotwords already present in ASR/trusted candidates.",
    )
    parser.add_argument(
        "--clean-nbest",
        action="store_true",
        help="Deduplicate and remove polluted/length-outlier n-best candidates before prompting.",
    )
    parser.add_argument("--clean-min-length-ratio", type=float, default=0.65)
    parser.add_argument("--clean-max-length-ratio", type=float, default=1.35)
    parser.add_argument("--clean-length-slack", type=int, default=8)
    parser.add_argument("--clean-anchor-suffix-filter", action="store_true")
    parser.add_argument("--clean-drop-polluted-top1", action="store_true", default=True)
    parser.add_argument(
        "--no-simplify-input",
        dest="simplify_input",
        action="store_false",
        default=True,
        help="Keep original text surfaces in COVO prompts instead of converting ASR/N-best/hotwords to simplified Chinese.",
    )
    parser.add_argument(
        "--include-consensus-spans",
        action="store_true",
        help="Add ChineseHP-style stable/uncertain n-best span evidence to the prompt.",
    )
    parser.add_argument(
        "--include-hotword-evidence",
        action="store_true",
        help="Add structured hotword-aware evidence fields for candidate keep/drop/conflict status.",
    )
    parser.add_argument(
        "--prefer-same-length",
        action="store_true",
        help="Tell COVO to prefer equal-length homophone substitutions over insertions/deletions.",
    )
    parser.add_argument(
        "--no-compact-evidence",
        dest="compact_evidence",
        action="store_false",
        default=True,
        help="Use the older verbose prompt with repeated score blocks and full hotword status.",
    )
    parser.add_argument("--consensus-span-threshold", type=float, default=0.75)
    parser.add_argument("--max-stable-spans", type=int, default=8)
    parser.add_argument("--max-uncertain-spans", type=int, default=8)
    parser.add_argument("--max-span-variants", type=int, default=6)
    parser.add_argument(
        "--max-confusables",
        type=int,
        default=0,
        help="Add up to N ChineseHP-style confusable candidates with local diff summaries.",
    )
    parser.add_argument("--limit", type=int, default=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert CB-Whisper evidence to Qwen messages")
    add_prepare_args(prepare)
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="Prepare messages, then run covo LoRA inference")
    add_prepare_args(run)
    run.add_argument("--prediction-output", required=True)
    run.add_argument("--covo-dir", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--model-name-or-path", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B")
    run.add_argument("--adapter-path", default="/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch")
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--max-new-tokens", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--device", default="auto")
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--infer-limit", type=int, default=0)
    run.add_argument("--disable-thinking", action="store_true")
    run.add_argument(
        "--post-filter-same-length",
        action="store_true",
        help="After COVO inference, keep only predictions with the same normalized character length as ASR top-1.",
    )
    run.add_argument("--evaluate", action="store_true")
    run.set_defaults(func=cmd_run)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
