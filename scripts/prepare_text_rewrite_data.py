#!/usr/bin/env python
"""Build Qwen SFT data for top-k ASR hypothesis fusion as final-text rewriting."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Internal JSONL with ASR/N-best/reference fields")
    parser.add_argument("--output", required=True, help="Qwen messages JSONL output")
    parser.add_argument("--target-field", default="reference", help="Training target text field")
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--max-pinyin", type=int, default=5)
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


def _format_user(record: Dict[str, Any], max_nbest: int, max_pinyin: int, include_pinyin: bool) -> str:
    input_block = record.get("input", {}) or {}
    lines: List[str] = [INSTRUCTION]

    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    lines.append(f"ASR top-1: {asr_top1}")

    nbest = list(input_block.get("nbest", []) or [])[: max(1, max_nbest)]
    if nbest:
        lines.append("N-best:")
        for idx, hyp in enumerate(nbest, 1):
            lines.append(f"{idx}. {hyp}")

    if include_pinyin:
        pinyin = list(input_block.get("nbest_pinyin", []) or [])[: max(1, max_pinyin)]
        if pinyin:
            lines.append("Pinyin:")
            for idx, item in enumerate(pinyin, 1):
                lines.append(f"{idx}. {item}")

    consensus = input_block.get("nbest_consensus")
    if isinstance(consensus, dict):
        stable = list(consensus.get("stable_spans", []) or [])
        uncertain = list(consensus.get("uncertain_spans", []) or [])
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
                        max_nbest=int(args.max_nbest),
                        max_pinyin=int(args.max_pinyin),
                        include_pinyin=bool(args.include_pinyin),
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
