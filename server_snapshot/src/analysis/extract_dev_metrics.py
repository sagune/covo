#!/usr/bin/env python
"""Rebuild the per-checkpoint AISHELL-dev table from the pipeline log.

Why this exists: the pipeline's P4 scrapes the CER out of the restore_eval output
with `grep -oE "CER=[0-9.]+"`, but `restore_eval.py` actually prints a row like

    restore[deployable]    CER  7.3109% | recall 94.99% (1632/1718) destroyed  17 | ...

with no `=`.  So `dev_metrics.tsv` ends up full of `none`.  The real numbers are
not lost -- the full table is appended to the pipeline log -- so this script
re-parses that log and rebuilds the table, which is what the "mid-course
checkpoint" requirement actually needs.

Usage:
    python extract_dev_metrics.py --log <pipeline.log> [--out dev_metrics.tsv]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

HEAD = re.compile(r"^#\s*(AISHELL dev[^\n]*)$")
ROW = re.compile(
    r"^\s*(input|raw|verifier|restore\[deployable\]|restore\[annotation\])\s+"
    r"CER\s+([0-9.]+)%\s*\|\s*recall\s+([0-9.]+)%\s*\((\d+)/(\d+)\)\s*"
    r"destroyed\s+(\d+)\s*\|\s*edited\s+(\d+)\s*\|\s*beyond-N-best\s+([+-][0-9.]+)\s*pp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    lines = Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines()
    current, blocks = None, []
    for line in lines:
        h = HEAD.match(line.strip())
        if h:
            current = {"label": h.group(1).strip(), "rows": {}}
            blocks.append(current)
            continue
        if current is not None:
            m = ROW.match(line)
            if m:
                current["rows"][m.group(1)] = {
                    "cer_pct": float(m.group(2)), "recall_pct": float(m.group(3)),
                    "hits": int(m.group(4)), "total": int(m.group(5)),
                    "destroyed": int(m.group(6)), "edited": int(m.group(7)),
                    "beyond_nbest_pp": float(m.group(8)),
                }

    blocks = [b for b in blocks if b["rows"]]
    if not blocks:
        print("no restore_eval blocks found in %s" % a.log)
        return

    hdr = ["label", "input_CER", "dep_CER", "dep_recall", "dep_destroyed",
           "dep_edited", "dep_beyond_nbest_pp"]
    out_lines = ["\t".join(hdr)]
    print("%-34s %9s %9s %9s %6s %7s %9s" % tuple(hdr))
    for b in blocks:
        d = b["rows"].get("restore[deployable]", {})
        i = b["rows"].get("input", {})
        vals = [b["label"],
                "%.4f" % i.get("cer_pct", float("nan")),
                "%.4f" % d.get("cer_pct", float("nan")),
                "%.2f" % d.get("recall_pct", float("nan")),
                d.get("destroyed", "?"), d.get("edited", "?"),
                "%.3f" % d.get("beyond_nbest_pp", float("nan"))]
        out_lines.append("\t".join(str(v) for v in vals))
        print("%-34s %9s %9s %9s %6s %7s %9s" % tuple(vals))

    if a.out:
        Path(a.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
