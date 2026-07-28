#!/usr/bin/env python3
"""Build strong-top1 ST-CMDS SFT data with confusable N-best hard negatives."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_chinesehp_text_sft import SYSTEM


def build_messages(nbest: list[str], pinyin: list[str], reference: str) -> list[dict]:
    user_lines = ["N-best文本："]
    user_lines.extend(f"{index}. {text}" for index, text in enumerate(nbest, 1))
    user_lines.append("N-best拼音：")
    user_lines.extend(f"{index}. {text}" for index, text in enumerate(pinyin, 1))
    user_lines.append('请输出：{"text":"纠错后的完整句子"}')
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(user_lines)},
        {
            "role": "assistant",
            "content": json.dumps({"text": reference}, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def make_record(row: dict, nbest: list[str], pinyin: list[str], suffix: str) -> dict:
    reference = str(row.get("reference", "")).strip()
    return {
        "id": f'{row.get("id", "")}:{suffix}',
        "split": str(row.get("split", "train")),
        "reference": reference,
        "asr_top1": nbest[0],
        "input": {
            "asr_top1": nbest[0],
            "nbest": nbest,
            "nbest_pinyin": pinyin,
        },
        "messages": build_messages(nbest, pinyin, reference),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-copies", type=int, default=2)
    parser.add_argument("--max-nbest", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    rows = []
    counts = {
        "source": 0,
        "top1_exact": 0,
        "recoverable_error": 0,
        "outside_nbest_error": 0,
        "correction_records": 0,
        "preservation_records": 0,
    }
    with args.input.open(encoding="utf-8") as reader:
        for line in reader:
            if not line.strip():
                continue
            source = json.loads(line)
            counts["source"] += 1
            reference = str(source.get("reference", "")).strip()
            input_block = source.get("input", {}) or {}
            nbest = [
                str(item).strip()
                for item in input_block.get("nbest", [])
                if str(item).strip()
            ][: max(1, args.max_nbest)]
            if not nbest:
                continue
            pinyin = [str(item).strip() for item in input_block.get("nbest_pinyin", [])][
                : len(nbest)
            ]
            pinyin.extend("" for _ in range(len(nbest) - len(pinyin)))

            if nbest[0] == reference:
                counts["top1_exact"] += 1
                rows.append(make_record(source, nbest, pinyin, "preserve"))
                counts["preservation_records"] += 1
                continue

            rows.append(make_record(source, nbest, pinyin, "correct"))
            counts["correction_records"] += 1
            if reference not in nbest:
                counts["outside_nbest_error"] += 1
                continue

            counts["recoverable_error"] += 1
            ref_index = nbest.index(reference)
            preserve_nbest = [reference] + [
                text for index, text in enumerate(nbest) if index != ref_index
            ]
            preserve_pinyin = [pinyin[ref_index]] + [
                text for index, text in enumerate(pinyin) if index != ref_index
            ]
            for copy_index in range(max(1, args.preserve_copies)):
                rows.append(
                    make_record(
                        source,
                        preserve_nbest,
                        preserve_pinyin,
                        f"hard-preserve-{copy_index}",
                    )
                )
                counts["preservation_records"] += 1

    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "output_records": len(rows),
                "preservation_rate": counts["preservation_records"] / max(len(rows), 1),
                "seed": args.seed,
                "counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
