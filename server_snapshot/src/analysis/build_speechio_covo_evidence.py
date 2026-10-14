#!/usr/bin/env python3
"""Wrap Speechio-Formal SenseVoice outputs as conservative COVO evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            text = str(row.get("prediction", ""))
            target.write(
                json.dumps(
                    {
                        "id": row.get("id", ""),
                        "source": "sensevoice_speechio_formal",
                        "dataset": "speechio_formal",
                        "split": "ZH00006",
                        "reference": row.get("reference", ""),
                        "verbatim_reference": row.get("verbatim_reference", ""),
                        "input": {"asr_top1": text, "nbest": [text], "covo_hotwords": []},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            written += 1
    print(json.dumps({"written": written, "output": str(args.output)}))


if __name__ == "__main__":
    main()
