#!/usr/bin/env python
"""Build COVO SFT rows from actual model failures on CB-Whisper evidence."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_COVO_SRC = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
if DEFAULT_COVO_SRC not in sys.path:
    sys.path.insert(0, DEFAULT_COVO_SRC)

from covo.metrics import edit_distance  # type: ignore
from covo.text import normalize_chinese_text  # type: ignore


def norm(value: Any) -> str:
    return normalize_chinese_text(str(value or ""))


def dist(left: Any, right: Any) -> int:
    return int(edit_distance(list(norm(left)), list(norm(right))))


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def unique_texts(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        key = norm(text)
        if key and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def best_candidate(reference: str, candidates: List[str]) -> Tuple[str, int]:
    if not candidates:
        return "", 10**9
    scored = [(dist(candidate, reference), idx, candidate) for idx, candidate in enumerate(candidates)]
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2], scored[0][0]


def keyword_mentions(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("keyword_mentions", []) or []:
        if isinstance(item, dict):
            text = norm(item.get("mention", ""))
            if text:
                output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def prompt_hotwords(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if isinstance(item, dict):
            text = norm(item.get("text", ""))
            if text:
                output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def replace_assistant_with_reference(row: Dict[str, Any], tags: List[str], stats: Dict[str, Any], copy_idx: int = 0) -> Dict[str, Any]:
    reference = str(row.get("reference", "")).strip()
    messages = [dict(message) for message in row.get("messages", []) if isinstance(message, dict)]
    target = json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":"))
    replaced = False
    for message in reversed(messages):
        if message.get("role") == "assistant":
            message["content"] = target
            replaced = True
            break
    if not replaced:
        messages.append({"role": "assistant", "content": target})
    out = {
        "id": f"{row.get('id', '')}:actual_error_sft:{copy_idx}",
        "source": row.get("source", "cbwhisper"),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "reference": reference,
        "input": row.get("input", {}),
        "messages": messages,
        "actual_error_sft_tags": tags,
        "actual_error_sft_stats": stats,
    }
    return out


def classify(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    reference = str(row.get("reference", "")).strip()
    prediction = str(row.get("prediction", "")).strip()
    input_block = dict(row.get("input", {}) or {})
    base = str(input_block.get("asr_top1", "")).strip()
    nbest = unique_texts([base] + list(input_block.get("nbest", []) or []))
    nbest_best, nbest_d = best_candidate(reference, nbest)
    base_d = dist(base, reference)
    pred_d = dist(prediction, reference)
    ref_norm = norm(reference)
    base_norm = norm(base)
    pred_norm = norm(prediction)
    mentions = keyword_mentions(input_block)
    prompts = prompt_hotwords(input_block)
    hotword_lost = [kw for kw in mentions if kw in base_norm and kw not in pred_norm]
    hotword_missing = [kw for kw in mentions if kw in ref_norm and kw not in pred_norm]
    false_prompt_in_pred = [kw for kw in prompts if kw not in ref_norm and kw in pred_norm]

    tags: List[str] = []
    if pred_d > base_d:
        tags.append("actual_worsened")
    if base_d == 0 and pred_d > 0:
        tags.append("base_correct_broken")
    if base_d > 0 and pred_d >= base_d:
        tags.append("missed_correction")
    if nbest_d < pred_d:
        tags.append("nbest_better_than_prediction")
    if hotword_lost:
        tags.append("true_hotword_lost")
    if hotword_missing:
        tags.append("true_hotword_missing")
    if false_prompt_in_pred:
        tags.append("false_prompt_inserted")
    if base_d == 0 and pred_d == 0:
        tags.append("correct_preservation")
    elif pred_d == 0:
        tags.append("successful_repair")

    stats = {
        "base_d": base_d,
        "pred_d": pred_d,
        "nbest_oracle_d": nbest_d,
        "nbest_oracle": nbest_best,
        "hotword_lost": hotword_lost,
        "hotword_missing": hotword_missing,
        "false_prompt_in_pred": false_prompt_in_pred,
    }
    return tags, stats


def repeat_for_tags(tags: List[str], args: argparse.Namespace) -> int:
    tag_set = set(tags)
    if "base_correct_broken" in tag_set or "actual_worsened" in tag_set:
        return int(args.worsened_repeats)
    if "true_hotword_lost" in tag_set or "true_hotword_missing" in tag_set:
        return int(args.hotword_repeats)
    if "nbest_better_than_prediction" in tag_set or "missed_correction" in tag_set:
        return int(args.error_repeats)
    if "correct_preservation" in tag_set:
        return int(args.correct_repeats)
    if "successful_repair" in tag_set:
        return int(args.success_repeats)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-error-rows", type=int, default=0)
    parser.add_argument("--max-correct-rows", type=int, default=160)
    parser.add_argument("--worsened-repeats", type=int, default=5)
    parser.add_argument("--hotword-repeats", type=int, default=4)
    parser.add_argument("--error-repeats", type=int, default=3)
    parser.add_argument("--correct-repeats", type=int, default=1)
    parser.add_argument("--success-repeats", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    error_rows = []
    correct_rows = []
    tag_counts: Dict[str, int] = {}
    for row in read_jsonl(args.predictions):
        tags, stats = classify(row)
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        pred_d = int(stats["pred_d"])
        if pred_d > 0 and not (set(tags) <= {"successful_repair"}):
            error_rows.append((row, tags, stats))
        elif "correct_preservation" in tags or "successful_repair" in tags:
            correct_rows.append((row, tags, stats))

    rng.shuffle(error_rows)
    rng.shuffle(correct_rows)
    if int(args.max_error_rows) > 0:
        error_rows = error_rows[: int(args.max_error_rows)]
    correct_rows = correct_rows[: max(0, int(args.max_correct_rows))]

    rows = []
    for row, tags, stats in error_rows + correct_rows:
        repeats = max(0, repeat_for_tags(tags, args))
        for copy_idx in range(repeats):
            rows.append(replace_assistant_with_reference(row, tags, stats, copy_idx=copy_idx))
    rng.shuffle(rows)
    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "predictions": args.predictions,
                "output": args.output,
                "written": written,
                "source_counts": {
                    "error_rows": len(error_rows),
                    "correct_rows": len(correct_rows),
                },
                "tag_counts": tag_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
