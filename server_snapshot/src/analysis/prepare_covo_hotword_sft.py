#!/usr/bin/env python
"""Build hotword-aware covo/Qwen SFT data from AISHELL-style rewrite records."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List

from cbsensevoice_covo_bridge import SYSTEM_MESSAGE, build_user_prompt


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


def stable_rng(record: Dict[str, Any], seed: int) -> random.Random:
    key = f"{seed}:{record.get('split','')}:{record.get('id','')}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def char_distance(a: str, b: str) -> int:
    distance = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b).get_opcodes():
        if tag != "equal":
            distance += max(i2 - i1, j2 - j1)
    return distance


def add_unique(rows: List[str], text: str, max_len: int) -> None:
    text = "".join(str(text or "").split())
    if len(text) < 2:
        return
    if len(text) > max_len:
        text = text[:max_len]
    if text and text not in rows:
        rows.append(text)


def reference_hotwords(reference: str, asr_top1: str, max_items: int, max_len: int, rng: random.Random) -> List[str]:
    rows: List[str] = []
    matcher = SequenceMatcher(a=asr_top1, b=reference)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        add_unique(rows, reference[max(0, j1 - 2): min(len(reference), j2 + 2)], max_len)
        add_unique(rows, reference[j1:j2], max_len)
        if len(rows) >= max_items:
            return rows[:max_items]

    if len(rows) < max_items:
        starts = list(range(max(0, len(reference) - 1)))
        rng.shuffle(starts)
        for start in starts:
            span_len = rng.randint(2, max(2, min(max_len, 5)))
            add_unique(rows, reference[start: start + span_len], max_len)
            if len(rows) >= max_items:
                break
    return rows[:max_items]


def confusable_hotwords(
    reference: str,
    asr_top1: str,
    nbest: List[str],
    max_items: int,
    max_len: int,
    rng: random.Random,
) -> List[str]:
    rows: List[str] = []
    ranked: List[tuple[int, float, str]] = []
    seen = {asr_top1, reference}
    for hyp in nbest:
        hyp = "".join(str(hyp or "").split())
        if not hyp or hyp in seen:
            continue
        seen.add(hyp)
        ratio = SequenceMatcher(a=asr_top1, b=hyp).ratio()
        dist = char_distance(asr_top1, hyp)
        if dist <= 0 or ratio < 0.55:
            continue
        ranked.append((dist, -ratio, hyp))
    ranked.sort()
    for _, _, hyp in ranked:
        for tag, i1, i2, _j1, _j2 in SequenceMatcher(a=reference, b=hyp).get_opcodes():
            if tag == "equal":
                continue
            add_unique(rows, hyp[max(0, i1 - 1): min(len(hyp), i2 + 2)], max_len)
            if len(rows) >= max_items:
                return rows
    rng.shuffle(rows)
    return rows[:max_items]


def make_record(record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    input_block = dict(record.get("input", {}) or {})
    reference = str(record.get("reference", "")).strip()
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    rng = stable_rng(record, int(args.seed))

    positives = reference_hotwords(reference, asr_top1, int(args.max_positive_hotwords), int(args.max_hotword_chars), rng)
    negatives = confusable_hotwords(
        reference=reference,
        asr_top1=asr_top1,
        nbest=list(input_block.get("nbest", []) or []),
        max_items=int(args.max_negative_hotwords),
        max_len=int(args.max_hotword_chars),
        rng=rng,
    )
    if positives and rng.random() < float(args.drop_positive_rate):
        positives = positives[: max(1, len(positives) - 1)]
    if negatives and rng.random() < float(args.drop_negative_rate):
        negatives = negatives[: max(0, len(negatives) - 1)]

    prompt_hotwords = [
        {"text": text, "weight": round(1.0 - idx * 0.03, 4), "synthetic": True, "label": "positive"}
        for idx, text in enumerate(positives)
    ]
    kws_hotwords = [
        {"text": text, "score": round(0.96 - idx * 0.02, 4), "synthetic": True, "label": "positive"}
        for idx, text in enumerate(positives)
    ]
    for idx, text in enumerate(negatives):
        kws_hotwords.append(
            {"text": text, "score": round(0.88 - idx * 0.02, 4), "synthetic": True, "label": "confusable"}
        )

    rng.shuffle(kws_hotwords)
    input_block["prompt_hotwords"] = prompt_hotwords
    input_block["hotwords"] = kws_hotwords[: int(args.max_context_hotwords)]
    input_block.setdefault("cbwhisper", {"candidates": []})

    bridge_args = argparse.Namespace(
        max_nbest=args.max_nbest,
        max_pinyin=args.max_pinyin,
        include_pinyin=args.include_pinyin,
        max_hotwords=args.max_context_hotwords,
        max_prompt_hotwords=args.max_positive_hotwords,
        max_candidates_with_scores=args.max_candidates_with_scores,
        hotword_source=args.hotword_source,
    )
    output = {
        "id": str(record.get("id", "")),
        "source": record.get("source", "chinesehp/aishell-1"),
        "split": record.get("split", ""),
        "reference": reference,
        "input": input_block,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_user_prompt({**record, "input": input_block}, bridge_args)},
            {
                "role": "assistant",
                "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-nbest", type=int, default=8)
    parser.add_argument("--max-pinyin", type=int, default=5)
    parser.add_argument("--include-pinyin", action="store_true")
    parser.add_argument("--max-positive-hotwords", type=int, default=4)
    parser.add_argument("--max-negative-hotwords", type=int, default=3)
    parser.add_argument("--max-context-hotwords", type=int, default=8)
    parser.add_argument("--max-hotword-chars", type=int, default=8)
    parser.add_argument("--max-candidates-with-scores", type=int, default=0)
    parser.add_argument("--hotword-source", choices=["prompt", "kws", "all"], default="prompt")
    parser.add_argument("--drop-positive-rate", type=float, default=0.10)
    parser.add_argument("--drop-negative-rate", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def rows() -> Iterable[Dict[str, Any]]:
        for idx, record in enumerate(read_jsonl(args.input), 1):
            yield make_record(record, args)
            if args.limit and idx >= int(args.limit):
                break

    written = write_jsonl(args.output, rows())
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
