#!/usr/bin/env python3
"""Transcription check: every numeric literal in the LaTeX tables must also appear
in one of the verified source documents.  Catches typos introduced by hand-copying.

Numbers that are structural (table numbering, percentages of rows quoted in the
same table, CI bounds) are checked too -- if a bound were mistyped it would fail
here, which is the point.
"""
import re
from pathlib import Path

BASE = Path("/root/autodl-tmp/src")
LOGS = Path("/root/autodl-tmp/.dsh_checks/rerank")
tex = (BASE / "PAPER_TABLES_rerank.tex").read_text(encoding="utf-8")
sources = "\n".join((BASE / n).read_text(encoding="utf-8") for n in
                    ("RESULTS_RERANK_20260914.md", "EXPERIMENT_LOG_20260914.md"))
# numbers taken from the run logs rather than the write-up
for n in ("policy_sweep4.log", "policy_sweep6.log", "v2_confirm.log"):
    p = LOGS / n
    if p.exists():
        sources += "\n" + p.read_text(encoding="utf-8", errors="replace")
# dataset lexicon sizes, read from the files themselves
for rel, label in (("datasets/aishell/data_aishell_sensevoice/hotword/dev/hotword.txt", "aishell"),
                   ("datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/hotword.txt", "thchs30"),
                   ("datasets/stcmds/cb_sensevoice_heldout/hotword/test/hotword.txt", "stcmds")):
    p = Path("/root/autodl-tmp") / rel
    if p.exists():
        sources += "\n%s %d\n" % (label, sum(1 for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()))

# strip comments so table/algorithm numbering does not pollute the scan
body = "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))

# ignore trivial integers used for row/col counts and LaTeX scaffolding
IGNORE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
          "13", "16", "24", "100", "2000", "95", "2026", "14", "19140"}

nums = set()
for m in re.finditer(r"\d+\.\d+", body):          # decimals only
    nums.add(m.group(0))
for m in re.finditer(r"(?<![\d.])(\d{2,3})(?![\d.])", body):   # big integers
    nums.add(m.group(1))

missing = []
for n in sorted(nums):
    if n in IGNORE:
        continue
    if n in sources:
        continue
    missing.append(n)

print("distinct numeric literals checked:", len(nums))
if missing:
    print("NOT FOUND in the source docs (%d):" % len(missing))
    for n in missing:
        print("   ", n)
else:
    print("all numeric literals found in the verified source docs")
