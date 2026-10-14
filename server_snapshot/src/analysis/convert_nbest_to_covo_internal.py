#!/usr/bin/env python3
"""Convert flat ASR N-best JSONL into the original COVO evidence schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            nbest = [str(item).strip() for item in row.get("nbest", []) if str(item).strip()]
            if not nbest:
                continue
            pinyin = [str(item).strip() for item in row.get("nbest_pinyin", [])]
            record = {
                "id": str(row.get("id", "")),
                "source": str(row.get("dataset", row.get("source", ""))),
                "split": str(row.get("split", "")),
                "task": "asr_constrained_correction",
                "reference": str(row.get("reference", "")).strip(),
                "input": {
                    "asr_top1": nbest[0],
                    "nbest": nbest,
                    "nbest_pinyin": pinyin[: len(nbest)],
                    "asr_top1_pinyin": pinyin[0] if pinyin else "",
                },
                "output": {"edits": []},
            }
            target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1

    print(json.dumps({"input": str(args.input), "output": str(args.output), "written": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
