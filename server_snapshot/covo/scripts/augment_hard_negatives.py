#!/usr/bin/env python
"""Add evidence-conflicting hard negatives to ASR correction SFT data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from covo.edits import parse_edits
from covo.io import read_jsonl, write_jsonl
from covo.text import joined_pinyin, normalize_chinese_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input SFT JSONL")
    parser.add_argument("--output", required=True, help="Output augmented SFT JSONL")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--include-original", action="store_true", default=True)
    parser.add_argument("--no-include-original", dest="include_original", action="store_false")
    parser.add_argument(
        "--max-generated",
        type=int,
        default=0,
        help="Maximum generated hard negatives; 0 means no limit.",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Probability of keeping each eligible hard negative.",
    )
    parser.add_argument(
        "--nbest-size",
        type=int,
        default=10,
        help="Maximum N-best entries in generated hard negatives.",
    )
    return parser.parse_args()


def _unique_texts(texts: Iterable[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for text in texts:
        text = str(text).strip()
        key = normalize_chinese_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _has_gold_edits(record: Dict[str, Any]) -> bool:
    try:
        return bool(parse_edits(record.get("output", {}) or {}))
    except ValueError:
        return False


def make_reference_top1_negative(record: Dict[str, Any], *, nbest_size: int) -> Dict[str, Any] | None:
    """Create a no-edit example where top1 is already correct.

    The original wrong top1 and N-best alternatives are kept behind the reference
    as distracting evidence. This targets false corrections without inventing
    unsupported labels.
    """
    input_block = record.get("input", {}) or {}
    reference = normalize_chinese_text(record.get("reference", ""))
    asr_top1 = normalize_chinese_text(input_block.get("asr_top1", ""))
    if not reference or not asr_top1 or reference == asr_top1:
        return None
    if not _has_gold_edits(record):
        return None

    original_nbest = [str(item) for item in input_block.get("nbest", []) or []]
    nbest = _unique_texts([reference, asr_top1, *original_nbest], max(1, int(nbest_size)))
    if len(nbest) <= 1:
        return None

    new_input = {
        **input_block,
        "asr_top1": reference,
        "nbest": nbest,
        "nbest_pinyin": [joined_pinyin(item) for item in nbest],
        "asr_top1_pinyin": joined_pinyin(reference),
    }
    return {
        **record,
        "id": f"{record.get('id', '')}::hardneg_ref_top1",
        "input": new_input,
        "output": {"edits": []},
        "augmentation": {
            "type": "reference_top1_hard_negative",
            "source_id": str(record.get("id", "")),
        },
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(int(args.seed))
    stats = {
        "original": 0,
        "eligible": 0,
        "generated": 0,
        "sampled_out": 0,
        "max_generated_stop": 0,
    }

    def records():
        for record in read_jsonl(args.input):
            stats["original"] += 1
            if args.include_original:
                yield record
            if args.max_generated and stats["generated"] >= int(args.max_generated):
                stats["max_generated_stop"] += 1
                continue
            hard_negative = make_reference_top1_negative(record, nbest_size=int(args.nbest_size))
            if hard_negative is None:
                continue
            stats["eligible"] += 1
            if rng.random() > float(args.sample_rate):
                stats["sampled_out"] += 1
                continue
            stats["generated"] += 1
            yield hard_negative

    written = write_jsonl(args.output, records())
    print(json.dumps({"input": args.input, "output": args.output, "written": written, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
