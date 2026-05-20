"""Formatting helpers for SFT/chat datasets."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def format_input_block(input_block: Dict[str, Any]) -> str:
    lines: List[str] = []
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    if asr_top1:
        lines.append(f"ASR: {asr_top1}")

    nbest = list(input_block.get("nbest", []) or [])
    if nbest:
        lines.append("N-best:")
        for idx, hyp in enumerate(nbest, 1):
            lines.append(f"{idx}. {hyp}")

    pinyin = list(input_block.get("nbest_pinyin", []) or [])
    if pinyin:
        lines.append("Pinyin:")
        for idx, item in enumerate(pinyin[: len(nbest) or len(pinyin)], 1):
            lines.append(f"{idx}. {item}")

    hotwords = list(input_block.get("hotwords", []) or [])
    if hotwords:
        lines.append("Hotwords:")
        for item in hotwords:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                score = item.get("score")
                if score is None:
                    lines.append(f"- {text}")
                else:
                    lines.append(f"- {text} (score={float(score):.4f})")
            else:
                lines.append(f"- {item}")

    return "\n".join(lines).strip()


def output_to_text(output: Dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def to_qwen_messages(record: Dict[str, Any]) -> Dict[str, Any]:
    instruction = str(record.get("instruction", "")).strip()
    user_content = format_input_block(record.get("input", {}) or {})
    if instruction:
        user_content = f"{instruction}\n\n{user_content}".strip()
    return {
        "id": str(record.get("id", "")),
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个保守的中文 ASR 后纠错器。只输出合法 JSON，"
                    "格式为 {\"edits\":[{\"from\":\"...\",\"to\":\"...\",\"reason\":\"...\"}]}。"
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_to_text(record.get("output", {}) or {})},
        ],
    }


def to_prompt_completion(record: Dict[str, Any]) -> Dict[str, str]:
    prompt = str(record.get("instruction", "")).strip()
    input_text = format_input_block(record.get("input", {}) or {})
    if input_text:
        prompt = f"{prompt}\n\n{input_text}".strip()
    return {
        "prompt": prompt,
        "completion": output_to_text(record.get("output", {}) or {}),
    }


def to_covoger_style(record: Dict[str, Any]) -> Dict[str, str]:
    input_block = record.get("input", {}) or {}
    nbest = list(input_block.get("nbest", []) or [])
    hypotheses = "\n".join(f"{idx}. {hyp}" for idx, hyp in enumerate(nbest, 1))
    query = (
        "Correct the ASR transcription using the N-best hypotheses and pinyin evidence. "
        "Return JSON edits only.\n"
        f"{hypotheses}"
    ).strip()
    return {
        "query": query,
        "response": output_to_text(record.get("output", {}) or {}),
    }


def convert_record_format(record: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    fmt = str(fmt).lower()
    if fmt in {"internal", "covo"}:
        return record
    if fmt in {"qwen", "qwen-messages", "messages"}:
        return to_qwen_messages(record)
    if fmt in {"prompt-completion", "prompt"}:
        return to_prompt_completion(record)
    if fmt in {"covoger", "query-response"}:
        return to_covoger_style(record)
    raise ValueError(f"Unsupported format: {fmt}")
