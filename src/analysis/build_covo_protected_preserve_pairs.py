#!/usr/bin/env python
"""Build train-only COVO preference pairs for preserving protected hotwords.

The input is a COVO prediction JSONL that still contains the original prompt
messages. We mine cases where ASR top-1 already contains a true prompt hotword
but the model prediction rewrites it away. The chosen response is the reference;
the rejected response is the model prediction. This targets the failure mode
"do not turn protected named entities into common homophones".
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_COVO_SRC = str(Path(__file__).resolve().parents[2] / "covo" / "src")
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


def prompt_hotwords(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("prompt_hotwords", []) or []:
        if not isinstance(item, dict):
            continue
        text = norm(item.get("text", ""))
        if text:
            output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def true_mentions(input_block: Dict[str, Any]) -> List[str]:
    output = []
    for item in input_block.get("keyword_mentions", []) or []:
        if not isinstance(item, dict):
            continue
        text = norm(item.get("mention", ""))
        if text:
            output.append(text)
    return sorted(set(output), key=lambda item: (-len(item), item))


def prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [{"role": str(row.get("role", "")), "content": str(row.get("content", ""))} for row in messages]


def make_pair(record: Dict[str, Any], min_hotword_len: int, max_pred_d: int) -> Dict[str, Any] | None:
    reference = str(record.get("reference", "")).strip()
    prediction = str(record.get("prediction", "")).strip()
    input_block = dict(record.get("input", {}) or {})
    base = str(input_block.get("asr_top1", "")).strip()
    ref_norm = norm(reference)
    base_norm = norm(base)
    pred_norm = norm(prediction)
    if not ref_norm or not prediction or prediction == reference:
        return None

    allowed = set(prompt_hotwords(input_block))
    mentions = [kw for kw in true_mentions(input_block) if len(kw) >= int(min_hotword_len)]
    protected = [kw for kw in mentions if kw in allowed and kw in ref_norm and kw in base_norm]
    lost = [kw for kw in protected if kw not in pred_norm]
    if not lost:
        return None

    base_d = dist(base, reference)
    pred_d = dist(prediction, reference)
    if int(max_pred_d) > 0 and pred_d > int(max_pred_d):
        return None

    return {
        "id": str(record.get("id", "")),
        "source": record.get("source", "cbwhisper"),
        "split": record.get("split", ""),
        "pair_type": "actual_protected_hotword_preserve_dpo",
        "lost_protected_hotwords": lost,
        "base_d": base_d,
        "pred_d": pred_d,
        "prompt_messages": prompt_messages(record),
        "chosen": {"text": reference},
        "rejected": {"text": prediction},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-hotword-len", type=int, default=2)
    parser.add_argument("--max-pred-d", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for record in read_jsonl(args.predictions):
        pair = make_pair(record, int(args.min_hotword_len), int(args.max_pred_d))
        if pair is not None:
            rows.append(pair)
    random.Random(int(args.seed)).shuffle(rows)
    if int(args.limit) > 0:
        rows = rows[: int(args.limit)]
    written = write_jsonl(args.output, rows)
    summary = {
        "predictions": args.predictions,
        "output": args.output,
        "written": written,
        "min_hotword_len": int(args.min_hotword_len),
        "max_pred_d": int(args.max_pred_d),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
