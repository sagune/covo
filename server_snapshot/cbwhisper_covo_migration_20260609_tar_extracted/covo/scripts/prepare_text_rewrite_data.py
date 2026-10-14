#!/usr/bin/env python
"""Build Qwen SFT data for top-k ASR hypothesis fusion as final-text rewriting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl


SYSTEM_MESSAGE = (
    "你是一个保守的中文 ASR 后纠错器。根据 ASR top-1、N-best 候选、拼音和共识片段，"
    "输出最终纠错后的中文句子。必须只输出一个合法 JSON 对象，格式为 {\"text\":\"...\"}。"
    "不要输出解释、推理过程、Markdown 或额外字段。"
)

INSTRUCTION = (
    "任务：利用 N-best 中的一致部分和不一致片段进行 ASR 后纠错。"
    "优先保留多条候选共同支持的内容；只在 N-best 分歧、拼音近似或明显同音错字处修改；"
    "如果证据不足，保持 ASR top-1 不变。"
)

HARD_NEGATIVE_INSTRUCTION = (
    "任务：利用 N-best 中的一致部分、不一致片段和易混淆候选进行 ASR 后纠错。"
    "易混淆候选是与 ASR top-1 很相近但局部不同的候选，它们可能包含正确改法，也可能是同音误导；"
    "请结合多候选共识、拼音和上下文判断，不要因为某个候选相似就盲目采用。"
    "如果证据不足，保持 ASR top-1 不变。"
)

ASR_ONLY_INSTRUCTION = (
    "任务：根据 ASR top-1 进行中文 ASR 后纠错。"
    "只做有把握的必要修改；如果证据不足，保持 ASR top-1 不变。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Internal JSONL with ASR/N-best/reference fields")
    parser.add_argument("--output", required=True, help="Qwen messages JSONL output")
    parser.add_argument("--target-field", default="reference", help="Training target text field")
    parser.add_argument("--evidence-mode", choices=["asr-only", "nbest", "consensus"], default="consensus")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--max-pinyin", type=int, default=5)
    parser.add_argument("--max-hard-negatives", type=int, default=0)
    parser.add_argument("--evidence-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def _nested_get(record: Dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _format_span(item: Dict[str, Any]) -> str:
    start = int(item.get("start", 0))
    end = int(item.get("end", 0))
    text = str(item.get("text", ""))
    support = float(item.get("support", 0.0))
    variants = item.get("variants", [])
    suffix = ""
    if isinstance(variants, list) and variants:
        suffix = " variants=" + json.dumps(variants[:8], ensure_ascii=False, separators=(",", ":"))
    return f"- {start}:{end} {text} support={support:.3f}{suffix}"


def _stable_rng(record: Dict[str, Any], seed: int) -> Any:
    key = f"{seed}:{record.get('split','')}:{record.get('id','')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    import random

    return random.Random(int(digest[:16], 16))


def _drop_items(items: List[Any], rate: float, rng: Any, min_keep: int = 1) -> List[Any]:
    if not items or rate <= 0:
        return items
    rate = max(0.0, min(float(rate), 0.95))
    kept = [item for item in items if rng.random() >= rate]
    if len(kept) >= min_keep:
        return kept
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[: min(len(shuffled), max(0, min_keep))]


def _char_distance(a: str, b: str) -> int:
    matcher = SequenceMatcher(a=a, b=b)
    distance = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        distance += max(i2 - i1, j2 - j1)
    return distance


def _diff_summary(base: str, candidate: str, max_parts: int = 4) -> str:
    parts: List[str] = []
    matcher = SequenceMatcher(a=base, b=candidate)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        src = base[i1:i2] or "∅"
        dst = candidate[j1:j2] or "∅"
        parts.append(f"{i1}:{i2} {src}->{dst}")
        if len(parts) >= max_parts:
            break
    return "; ".join(parts)


def _format_hard_negatives(input_block: Dict[str, Any], max_items: int) -> List[str]:
    if max_items <= 0:
        return []
    top1 = str(input_block.get("asr_top1", "")).strip()
    if not top1:
        return []
    nbest = list(input_block.get("nbest", []) or [])
    rows = []
    seen = {top1}
    for rank, hyp in enumerate(nbest, 1):
        hyp = str(hyp).strip()
        if not hyp or hyp in seen:
            continue
        seen.add(hyp)
        ratio = SequenceMatcher(a=top1, b=hyp).ratio()
        dist = _char_distance(top1, hyp)
        if dist <= 0:
            continue
        if ratio < 0.55:
            continue
        rows.append((dist, -ratio, rank, hyp, _diff_summary(top1, hyp)))
    rows.sort()
    output = []
    for dist, neg_ratio, rank, hyp, diff in rows[:max_items]:
        ratio = -neg_ratio
        output.append(f"- cand#{rank} sim={ratio:.3f} dist={dist}: {hyp} | diff={diff}")
    return output


def _format_user(
    record: Dict[str, Any],
    evidence_mode: str,
    max_nbest: int,
    max_pinyin: int,
    include_pinyin: bool,
    max_hard_negatives: int,
    evidence_dropout: float,
    seed: int,
) -> str:
    input_block = record.get("input", {}) or {}
    evidence_mode = str(evidence_mode).strip().lower()
    rng = _stable_rng(record, seed)
    dropout = max(0.0, float(evidence_dropout))
    if evidence_mode == "asr-only":
        instruction = ASR_ONLY_INSTRUCTION
    elif max_hard_negatives > 0:
        instruction = HARD_NEGATIVE_INSTRUCTION
    else:
        instruction = INSTRUCTION
    lines: List[str] = [instruction]

    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    lines.append(f"ASR top-1: {asr_top1}")

    if evidence_mode in {"nbest", "consensus"}:
        nbest = list(input_block.get("nbest", []) or [])[: max(1, max_nbest)]
        nbest = _drop_items(nbest, dropout, rng, min_keep=min(3, len(nbest)))
        if nbest:
            lines.append("N-best:")
            for idx, hyp in enumerate(nbest, 1):
                lines.append(f"{idx}. {hyp}")

        if include_pinyin:
            pinyin = list(input_block.get("nbest_pinyin", []) or [])[: max(1, max_pinyin)]
            pinyin = _drop_items(pinyin, dropout, rng, min_keep=min(2, len(pinyin)))
            if pinyin:
                lines.append("Pinyin:")
                for idx, item in enumerate(pinyin, 1):
                    lines.append(f"{idx}. {item}")

        hard_negatives = _format_hard_negatives(input_block, max_hard_negatives)
        hard_negatives = _drop_items(hard_negatives, dropout * 0.5, rng, min_keep=min(2, len(hard_negatives)))
        if hard_negatives:
            lines.append("Confusable candidates:")
            lines.extend(hard_negatives)

    if evidence_mode == "consensus":
        consensus = input_block.get("nbest_consensus")
        if isinstance(consensus, dict):
            stable = list(consensus.get("stable_spans", []) or [])
            uncertain = list(consensus.get("uncertain_spans", []) or [])
            stable = _drop_items(stable, dropout, rng, min_keep=min(2, len(stable)))
            uncertain = _drop_items(uncertain, dropout * 0.5, rng, min_keep=min(1, len(uncertain)))
            if stable:
                lines.append("Stable spans:")
                lines.extend(_format_span(item) for item in stable[:12])
            if uncertain:
                lines.append("Uncertain spans:")
                lines.extend(_format_span(item) for item in uncertain[:12])

    lines.append('请输出 JSON：{"text":"纠错后的完整句子"}')
    return "\n".join(lines)


def _records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    count = 0
    for record in read_jsonl(args.input):
        target = str(_nested_get(record, args.target_field, "")).strip()
        if not target:
            continue
        yield {
            "id": str(record.get("id", "")),
            "source": record.get("source", ""),
            "split": record.get("split", ""),
            "reference": target,
            "input": record.get("input", {}) or {},
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": _format_user(
                        record,
                        evidence_mode=str(args.evidence_mode),
                        max_nbest=int(args.max_nbest),
                        max_pinyin=int(args.max_pinyin),
                        include_pinyin=bool(args.include_pinyin),
                        max_hard_negatives=int(args.max_hard_negatives),
                        evidence_dropout=float(args.evidence_dropout),
                        seed=int(args.seed),
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"text": target}, ensure_ascii=False, separators=(",", ":")),
                },
            ],
        }
        count += 1
        if args.limit and count >= int(args.limit):
            break


def main() -> int:
    args = parse_args()
    written = write_jsonl(args.output, _records(args))
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
