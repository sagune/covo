#!/usr/bin/env python3
"""Emit the FRONT-END baseline of a prompt file as a predictions file.

The front end is what COVO receives: `input.asr_top1`. Writing it back out as
`prediction` lets restore_eval.py score it with the official harness, so a table's
"front end" row is produced by the same code as every other row instead of being quoted
from a different pipeline (the RESULTS doc warns that two front-end pipelines differ by
one edit on AISHELL).

    frontend_baseline.py --records <any .messages.jsonl or .predictions.jsonl> --out <file>
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = 0
    with Path(a.out).open("w", encoding="utf-8") as out:
        for line in Path(a.records).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            top = (r.get("input") or {}).get("asr_top1") or ""
            r["prediction"] = top
            r.pop("raw_prediction", None)
            r.pop("parse_warnings", None)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print("wrote %s rows=%d" % (a.out, n))


if __name__ == "__main__":
    main()
