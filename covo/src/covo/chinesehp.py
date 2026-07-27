"""ChineseHP conversion helpers."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from .edits import Edit
from .text import joined_pinyin, normalize_chinese_text


def make_minimal_edits(source: str, target: str) -> List[Edit]:
    """Create one conservative middle-span replacement from source to target."""
    source_norm = normalize_chinese_text(source)
    target_norm = normalize_chinese_text(target)
    if source_norm == target_norm:
        return []

    prefix = 0
    max_prefix = min(len(source_norm), len(target_norm))
    while prefix < max_prefix and source_norm[prefix] == target_norm[prefix]:
        prefix += 1

    source_suffix = len(source_norm)
    target_suffix = len(target_norm)
    while (
        source_suffix > prefix
        and target_suffix > prefix
        and source_norm[source_suffix - 1] == target_norm[target_suffix - 1]
    ):
        source_suffix -= 1
        target_suffix -= 1

    from_text = source_norm[prefix:source_suffix]
    to_text = target_norm[prefix:target_suffix]
    if not from_text:
        # Pure insertions cannot be applied with a simple from/to replace. Expand
        # left by one anchor character when possible.
        if prefix > 0:
            prefix -= 1
        else:
            source_suffix = min(len(source_norm), source_suffix + 1)
            target_suffix = min(len(target_norm), target_suffix + 1)
        from_text = source_norm[prefix:source_suffix]
        to_text = target_norm[prefix:target_suffix]
    if not from_text:
        return []
    return [Edit(from_text=from_text, to_text=to_text, reason="middle_span_replace")]


def make_sft_record(
    record: Dict[str, Any],
    *,
    nbest_size: int = 5,
    output_mode: str = "edits",
) -> Dict[str, Any]:
    nbest = [str(item) for item in record.get("nbest", []) if str(item).strip()]
    nbest = nbest[: max(1, int(nbest_size))]
    asr_top1 = nbest[0] if nbest else ""
    reference = str(record.get("reference", ""))
    nbest_pinyin = [str(item) for item in record.get("nbest_pinyin", [])][: len(nbest)]
    if len(nbest_pinyin) < len(nbest):
        nbest_pinyin.extend(joined_pinyin(item) for item in nbest[len(nbest_pinyin):])

    evidence = {
        "asr_top1": asr_top1,
        "nbest": nbest,
        "nbest_pinyin": nbest_pinyin,
        "asr_top1_pinyin": nbest_pinyin[0] if nbest_pinyin else joined_pinyin(asr_top1),
    }
    output: Dict[str, Any]
    if output_mode == "text":
        output = {"corrected_text": reference}
    elif output_mode == "edits":
        output = {"edits": [edit.to_json() for edit in make_minimal_edits(asr_top1, reference)]}
    else:
        raise ValueError(f"Unsupported output_mode: {output_mode}")

    return {
        "id": str(record.get("id", "")),
        "source": str(record.get("dataset", "chinesehp/aishell-1")),
        "split": str(record.get("split", "")),
        "task": "asr_constrained_correction",
        "instruction": (
            "根据 ASR 输出、N-best 候选和拼音证据进行中文 ASR 后纠错。"
            "只做必要的最小修改；没有必要纠错时输出空 edits。"
        ),
        "input": evidence,
        "output": output,
        "reference": reference,
    }


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
