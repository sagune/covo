#!/usr/bin/env python
"""Apply the pre-registered AC-COVO gates to one arm's predictions.

The gates were fixed BEFORE the run (RESULTS section 10.12 / 15 and the goal
objective), so this script exists to make the verdict mechanical rather than a
judgement call made after seeing the numbers.  It runs the two established metric
tools (`restore_eval.py`, `edit_precision.py`) and compares their output against
the thresholds, then applies the pre-declared FALSIFICATION clause.

Documented baselines (RESULTS sections 11 and 14), used as the reference point:

    deployed baseline V-C : CER 4.9356 %  recall 95.98 %  destroyed 0
    text-only trained arm : CER 7.3109 %  recall 94.99 %  destroyed 17
                            edited 2133 (41.6 %), edit precision 27.1 %, net -1171

Break-even edit precision for this project is ~54 %: below it an edit costs more
characters than it gains, so precision in the 45-58 % band means the available
information is worth exactly what it costs -- which is the pre-declared
falsification of the joint-gate hypothesis.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
PY = "/root/autodl-tmp/great/bin/python"
RESTORE = WS / ".dsh_checks/restore_eval.py"
EDITPREC = WS / ".dsh_checks/edit_precision.py"

BASELINE = {"cer_pct": 4.9356, "recall_pct": 95.98, "destroyed": 0}

# pre-registered thresholds
GATE = {
    "edit_precision_pct": 70.0,     # >=
    "net_chars": 0.0,               # >
    "recall_pct": 95.98,            # >=
    "destroyed": 0,                 # <=
    "cer_pct": 4.9356,              # <
    "beyond_nbest_pp": 0.0,         # not degraded (>= 0 relative to the front end)
}


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def parse_restore(out: str):
    """Pull the restore[deployable] row and the input row."""
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s*(input|restore\[deployable\])\s+CER\s+([0-9.]+)%\s*\|\s*"
                     r"recall\s+([0-9.]+)%\s*\((\d+)/(\d+)\)\s*destroyed\s+(\d+)\s*\|\s*"
                     r"edited\s+(\d+)\s*\|\s*beyond-N-best\s+([+-][0-9.]+)\s*pp", line)
        if m:
            tag = "input" if m.group(1) == "input" else "deployable"
            res[tag] = {"cer_pct": float(m.group(2)), "recall_pct": float(m.group(3)),
                        "recall_hits": int(m.group(4)), "recall_total": int(m.group(5)),
                        "destroyed": int(m.group(6)), "edited": int(m.group(7)),
                        "beyond_nbest_pp": float(m.group(8))}
    return res


def parse_editprec(out: str):
    res = {}
    m = re.search(r"EDIT PRECISION\s*:\s*([0-9.]+)%", out)
    if m:
        res["edit_precision_pct"] = float(m.group(1))
    m = re.search(r"characters gained by edits\s*:\s*\+?(-?\d+)", out)
    if m:
        res["net_chars"] = float(m.group(1))
    m = re.search(r"of those\s+helped/hurt/tie\s*:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", out)
    if m:
        res["helped"], res["hurt"], res["tie"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", default="AC-COVO")
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--uttid-file", required=True)
    ap.add_argument("--policy", default="deployable")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    out_r = run([PY, str(RESTORE), "--records", a.records, "--label", a.label,
                 "--aligned", a.aligned, "--uttid-file", a.uttid_file])
    out_e = run([PY, str(EDITPREC), "--records", a.records, "--label", a.label,
                 "--policy", a.policy])
    m = {**parse_restore(out_r).get("deployable", {}), **parse_editprec(out_e)}
    if not m:
        print(out_r)
        print(out_e)
        print("FATAL: could not parse any metric; the run produced nothing usable.")
        return 3

    print("=" * 78)
    print("AC-COVO PRE-REGISTERED GATE REPORT   arm=%s" % a.label)
    print("=" * 78)
    rows = [
        ("edit precision  >= 70 %", m.get("edit_precision_pct"),
         m.get("edit_precision_pct", 0) >= GATE["edit_precision_pct"]),
        ("net edit chars  > 0", m.get("net_chars"),
         m.get("net_chars", -1) > GATE["net_chars"]),
        ("hotword recall  >= 95.98 %", m.get("recall_pct"),
         m.get("recall_pct", 0) >= GATE["recall_pct"]),
        ("protected destroyed <= 0", m.get("destroyed"),
         m.get("destroyed", 99) <= GATE["destroyed"]),
        ("ST-CMDS CER     < 4.9356 %", m.get("cer_pct"),
         m.get("cer_pct", 99) < GATE["cer_pct"]),
        ("beyond-N-best   >= 0 pp", m.get("beyond_nbest_pp"),
         m.get("beyond_nbest_pp", -99) >= GATE["beyond_nbest_pp"]),
    ]
    npass = 0
    for name, val, ok in rows:
        npass += 1 if ok else 0
        print("  [%s] %-28s %s" % ("PASS" if ok else "FAIL", name,
                                   "n/a" if val is None else val))
    print()
    print("  helped/hurt/tie : %s / %s / %s"
          % (m.get("helped", "?"), m.get("hurt", "?"), m.get("tie", "?")))
    print("  edited rows     : %s" % m.get("edited", "?"))
    print("  reference       : baseline V-C CER %.4f%% recall %.2f%% destroyed %d"
          % (BASELINE["cer_pct"], BASELINE["recall_pct"], BASELINE["destroyed"]))
    print("                    text-only trained arm CER 7.3109% precision 27.1% net -1171")
    print()

    prec = m.get("edit_precision_pct")
    verdict = []
    if prec is not None and 45.0 <= prec <= 58.0:
        verdict.append(
            "FALSIFICATION TRIGGERED: edit precision %.1f%% is inside the 45-58%% band "
            "around the ~54%% break-even. Per the pre-declared clause, the joint-gate "
            "hypothesis is falsified: record the negative result and STOP; add no further "
            "patches." % prec)
    if m.get("cer_pct") is not None and m["cer_pct"] < BASELINE["cer_pct"]:
        verdict.append("ST-CMDS CER improved on the deployed baseline.")
    else:
        verdict.append("ST-CMDS CER did NOT improve on the deployed baseline.")
    verdict.append("%d/6 gates pass." % npass)
    for v in verdict:
        print("  " + v)

    if a.json_out:
        Path(a.json_out).write_text(
            __import__("json").dumps({"label": a.label, "metrics": m,
                                      "gates": {n: ok for n, _, ok in rows},
                                      "n_pass": npass, "verdict": verdict},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nwrote %s" % a.json_out)
    return 0 if npass == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
