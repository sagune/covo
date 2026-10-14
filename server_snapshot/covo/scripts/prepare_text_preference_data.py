#!/usr/bin/env python
"""Build preference pairs for final-text ASR correction.

Each output row keeps the original prompt messages and adds two assistant
responses:

{"prompt_messages": [...], "chosen": "{\"text\":\"...\"}", "rejected": "{\"text\":\"...\"}"}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.io import read_jsonl, write_jsonl
from covo.text import normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-nbest-rank", type=int, default=10)
    parser.add_argument("--nochange-ratio", type=float, default=0.45)
    parser.add_argument("--ref-in-nbest-ratio", type=float, default=0.25)
    return parser.parse_args()


def _json_text(text: str) -> str:
    return json.dumps({"text": str(text)}, ensure_ascii=False, separators=(",", ":"))


def _prompt_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = list(record.get("messages", []) or [])
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages]


def _closest_wrong(reference: str, candidates: List[str]) -> str:
    ref_norm = normalize_chinese_text(reference)
    rows = []
    seen = set()
    for rank, cand in enumerate(candidates, 1):
        cand = str(cand).strip()
        cand_norm = normalize_chinese_text(cand)
        if not cand_norm or cand_norm == ref_norm or cand_norm in seen:
            continue
        seen.add(cand_norm)
        rows.append((SequenceMatcher(a=ref_norm, b=cand_norm, autojunk=False).ratio(), -rank, cand))
    if not rows:
        return ""
    rows.sort(reverse=True)
    return rows[0][2]


def _make_pair(record: Dict[str, Any], max_nbest_rank: int) -> Dict[str, Any] | None:
    reference = str(record.get("reference", "")).strip()
    input_block = record.get("input", {}) or {}
    asr_top1 = str(input_block.get("asr_top1", "")).strip()
    nbest = [str(x).strip() for x in list(input_block.get("nbest", []) or [])[: max(1, int(max_nbest_rank))]]
    if not reference or not asr_top1:
        return None

    ref_norm = normalize_chinese_text(reference)
    top_norm = normalize_chinese_text(asr_top1)
    ref_in_nbest = any(normalize_chinese_text(x) == ref_norm for x in nbest)

    if top_norm == ref_norm:
        rejected = _closest_wrong(reference, nbest)
        pair_type = "nochange_vs_confusable"
    elif ref_in_nbest:
        rejected = asr_top1 if top_norm != ref_norm else _closest_wrong(reference, nbest)
        pair_type = "ref_candidate_vs_asr"
    else:
        rejected = _closest_wrong(reference, [asr_top1] + nbest) or asr_top1
        pair_type = "free_correction_vs_asr_or_confusable"

    if not rejected or normalize_chinese_text(rejected) == ref_norm:
        return None

    return {
        "id": str(record.get("id", "")),
        "source": record.get("source", ""),
        "split": record.get("split", ""),
        "pair_type": pair_type,
        "reference": reference,
        "input": input_block,
        "prompt_messages": _prompt_messages(record),
        "chosen": _json_text(reference),
        "rejected": _json_text(rejected),
    }


def _records(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    rng = random.Random(int(args.seed))
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "nochange_vs_confusable": [],
        "ref_candidate_vs_asr": [],
        "free_correction_vs_asr_or_confusable": [],
    }
    for record in read_jsonl(args.input):
        pair = _make_pair(record, int(args.max_nbest_rank))
        if pair is not None:
            buckets.setdefault(pair["pair_type"], []).append(pair)

    for values in buckets.values():
        rng.shuffle(values)

    if args.limit and int(args.limit) > 0:
        limit = int(args.limit)
        n_nochange = int(round(limit * float(args.nochange_ratio)))
        n_ref = int(round(limit * float(args.ref_in_nbest_ratio)))
        n_free = max(0, limit - n_nochange - n_ref)
        selected = (
            buckets["nochange_vs_confusable"][:n_nochange]
            + buckets["ref_candidate_vs_asr"][:n_ref]
            + buckets["free_correction_vs_asr_or_confusable"][:n_free]
        )
        if len(selected) < limit:
            used = {id(item) for item in selected}
            rest = [item for values in buckets.values() for item in values if id(item) not in used]
            selected.extend(rest[: limit - len(selected)])
    else:
        selected = [item for values in buckets.values() for item in values]

    rng.shuffle(selected)
    for item in selected:
        yield item


def main() -> int:
    args = parse_args()
    written = write_jsonl(args.output, _records(args))
    print(json.dumps({"input": args.input, "output": args.output, "written": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
