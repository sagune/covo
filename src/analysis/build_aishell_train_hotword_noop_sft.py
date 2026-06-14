#!/usr/bin/env python
"""Build no-op hotword-preservation SFT rows from AISHELL train alignments."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from cbwhisper_covo_bridge import SYSTEM_MESSAGE, build_user_prompt


def norm_text(text: Any) -> str:
    return "".join(str(text or "").split())


def read_transcripts(path: str | Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            rows[parts[0]] = norm_text(parts[1])
    return rows


def read_alignments(path: str | Path) -> Dict[str, List[str]]:
    by_utt: Dict[str, List[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            keyword = norm_text(parts[0])
            utt_id = parts[1].strip()
            if keyword and utt_id and keyword not in by_utt[utt_id]:
                by_utt[utt_id].append(keyword)
    return dict(by_utt)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def make_record(utt_id: str, reference: str, keywords: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    keywords = [kw for kw in keywords if kw and kw in reference]
    keywords = sorted(dict.fromkeys(keywords), key=lambda item: (-len(item), item))[: int(args.max_hotwords)]
    prompt_hotwords = [
        {"text": text, "weight": round(1.0 - idx * 0.02, 4), "source": "aishell_train_alignment"}
        for idx, text in enumerate(keywords[: int(args.max_prompt_hotwords)])
    ]
    hotwords = [
        {"text": text, "score": round(0.98 - idx * 0.01, 4), "source": "aishell_train_alignment"}
        for idx, text in enumerate(keywords)
    ]
    mentions = [
        {
            "mention": text,
            "total_offset": reference.find(text),
            "end_offset": reference.find(text) + len(text),
            "ner_tag": "TRAIN",
        }
        for text in keywords
        if reference.find(text) >= 0
    ]
    input_block = {
        "asr_top1": reference,
        "nbest": [reference],
        "nbest_pinyin": [],
        "hotwords": hotwords,
        "prompt_hotwords": prompt_hotwords,
        "keyword_mentions": mentions,
        "oracle_hotwords": [],
        "cbwhisper": {"batch_idx": None, "candidate_count": 1, "candidates": []},
    }
    bridge_args = argparse.Namespace(
        max_nbest=1,
        max_pinyin=0,
        include_pinyin=False,
        max_hotwords=args.max_hotwords,
        max_prompt_hotwords=args.max_prompt_hotwords,
        max_candidates_with_scores=0,
        hotword_source="prompt",
    )
    return {
        "id": utt_id,
        "source": "aishell-train-hotword-noop",
        "dataset": "aishell",
        "split": "train",
        "reference": reference,
        "input": input_block,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_user_prompt({"input": input_block}, bridge_args)},
            {
                "role": "assistant",
                "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-hotwords", type=int, default=6)
    parser.add_argument("--max-prompt-hotwords", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    transcripts = read_transcripts(args.transcript)
    aligned = read_alignments(args.aligned)
    rows = []
    missing_transcript = 0
    empty_after_filter = 0
    for utt_id, keywords in aligned.items():
        reference = transcripts.get(utt_id, "")
        if not reference:
            missing_transcript += 1
            continue
        kept = [kw for kw in keywords if kw in reference]
        if not kept:
            empty_after_filter += 1
            continue
        rows.append(make_record(utt_id, reference, kept, args))
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    if args.limit:
        rows = rows[: int(args.limit)]
    written = write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "aligned_utterances": len(aligned),
                "written": written,
                "missing_transcript": missing_transcript,
                "empty_after_filter": empty_after_filter,
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
