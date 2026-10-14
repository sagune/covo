#!/usr/bin/env python
"""Create synthetic hotword-use SFT rows from CB-SenseVoice ASR/reference differences."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cbsensevoice_covo_bridge import SYSTEM_MESSAGE, build_user_prompt  # noqa: E402

try:
    from opencc import OpenCC

    _OPENCC = OpenCC("t2s")
except Exception:
    _OPENCC = None


_PUNCT_RE = re.compile(r"[\s,，。.!！?？:：;；、\"“”‘’《》<>\[\]()（）-]+")


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    if _OPENCC is not None:
        text = _OPENCC.convert(text)
    return _PUNCT_RE.sub("", text)


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


def stable_rng(row: Dict[str, Any], seed: int) -> random.Random:
    key = f"{seed}:{row.get('split','')}:{row.get('id','')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def add_span(rows: List[str], value: str, min_len: int, max_len: int) -> None:
    value = norm(value)
    if min_len <= len(value) <= max_len and value not in rows:
        rows.append(value)


def diff_hotword_spans(reference: str, asr_top1: str, min_len: int, max_len: int, max_items: int) -> List[str]:
    ref = norm(reference)
    asr = norm(asr_top1)
    if not ref or not asr or ref == asr:
        return []
    output: List[str] = []
    matcher = SequenceMatcher(a=asr, b=ref)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        add_span(output, ref[j1:j2], min_len, max_len)
        add_span(output, ref[max(0, j1 - 1) : min(len(ref), j2 + 1)], min_len, max_len)
        add_span(output, ref[max(0, j1 - 2) : min(len(ref), j2 + 2)], min_len, max_len)
        if len(output) >= max_items:
            break
    return output[:max_items]


def make_record(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any] | None:
    input_block = dict(row.get("input", {}) or {})
    reference = str(row.get("reference", "")).strip()
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    spans = diff_hotword_spans(
        reference,
        asr_top1,
        min_len=int(args.min_hotword_len),
        max_len=int(args.max_hotword_len),
        max_items=int(args.max_positive_hotwords),
    )
    if not spans:
        return None

    rng = stable_rng(row, int(args.seed))
    existing_hotwords = [item for item in input_block.get("hotwords", []) or [] if isinstance(item, dict)]
    existing_prompt = [item for item in input_block.get("prompt_hotwords", []) or [] if isinstance(item, dict)]
    positives = [
        {"text": text, "weight": round(1.0 - idx * 0.04, 4), "synthetic": True, "label": "diff_positive"}
        for idx, text in enumerate(spans)
    ]
    kws_positives = [
        {"text": text, "score": round(0.995 - idx * 0.015, 4), "synthetic": True, "label": "diff_positive"}
        for idx, text in enumerate(spans)
    ]
    prompt_hotwords = positives + existing_prompt[: max(0, int(args.max_prompt_hotwords) - len(positives))]
    hotwords = kws_positives + existing_hotwords
    dedup_hotwords = []
    seen = set()
    for item in hotwords:
        text = norm(item.get("text", ""))
        if text and text not in seen:
            dedup_hotwords.append(item)
            seen.add(text)

    input_block["prompt_hotwords"] = prompt_hotwords[: int(args.max_prompt_hotwords)]
    input_block["hotwords"] = dedup_hotwords[: int(args.max_hotwords)]
    input_block.setdefault("cbwhisper", {"candidates": []})

    bridge_args = argparse.Namespace(
        max_nbest=args.max_nbest,
        max_pinyin=args.max_pinyin,
        include_pinyin=args.include_pinyin,
        max_hotwords=args.max_hotwords,
        max_prompt_hotwords=args.max_prompt_hotwords,
        max_candidates_with_scores=0,
        hotword_source="all",
        protect_supported_hotwords=args.protect_supported_hotwords,
    )
    return {
        "id": str(row.get("id", "")) + ":synthetic_hotword_use",
        "source": row.get("source", "cbwhisper"),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "reference": reference,
        "synthetic_hotwords": spans,
        "input": input_block,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_user_prompt({**row, "input": input_block}, bridge_args)},
            {
                "role": "assistant",
                "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-hotword-len", type=int, default=2)
    parser.add_argument("--max-hotword-len", type=int, default=8)
    parser.add_argument("--max-positive-hotwords", type=int, default=3)
    parser.add_argument("--max-hotwords", type=int, default=6)
    parser.add_argument("--max-prompt-hotwords", type=int, default=4)
    parser.add_argument("--max-nbest", type=int, default=6)
    parser.add_argument("--max-pinyin", type=int, default=3)
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument("--protect-supported-hotwords", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.input):
        out = make_record(row, args)
        if out is not None:
            rows.append(out)
        if args.limit and len(rows) >= int(args.limit):
            break
    random.Random(int(args.seed)).shuffle(rows)
    written = write_jsonl(args.output, rows)
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
