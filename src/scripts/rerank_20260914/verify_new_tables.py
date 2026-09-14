#!/usr/bin/env python3
"""Re-derive Tables 9 and 10 from the prediction files and compare them to the .tex.

check_tex_numbers.py proves each numeric literal appears SOMEWHERE in the source docs.
That is a transcription check, not a correctness check: if a wrong number were copied into
the .tex and that same wrong number existed anywhere in a doc, it would pass.  Tables 9
and 10 are hand-assembled from two scripts, so they get a real end-to-end check here.

    verify_new_tables.py
"""
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
PY = "/root/autodl-tmp/great/bin/python"
TEX = (WS / "src/PAPER_TABLES_rerank.tex").read_text(encoding="utf-8")
VC = R / "e2eSTCMDS_VC.predictions.jsonl"


def tex_rows(label):
    """Rows of the table with the given \\label, split on & with LaTeX stripped."""
    m = re.search(r"\\label\{" + label + r"\}(.*?)\\end\{tabular\}", TEX, re.S)
    if not m:
        return []
    body = m.group(1)
    out = []
    for raw in body.split("\\\\"):
        cells = [c.strip() for c in raw.split("&")]
        cells = [re.sub(r"\\[a-zA-Z]+\{?|[{}$}]", "", c).replace("\\", "").strip() for c in cells]
        out.append(cells)
    return out


def nums(cells):
    return [c for c in cells if re.fullmatch(r"[-+]?\d+(\.\d+)?", c)]


def canon(v):
    """Strip the leading '+' so '+50' and '50' compare equal (the .tex writes the sign,
    the scripts' own output does not - that asymmetry produced 3 false mismatches)."""
    return str(v).lstrip("+")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return p.stdout


print("=" * 96)
print("Table 9: backend behaviour decomposition  (source: covo_error_anatomy.py)")
print("=" * 96)
out = run([PY, str(WS / ".dsh_checks/covo_error_anatomy.py"), "--records", str(VC), "--label", "V-C"])
live = {}
for line in out.splitlines():
    m = re.match(r"\s+(no_change|sel_win|sel_tie|sel_loss|edit_win|edit_tie|edit_loss)\s+"
                 r"(\d+)\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+([+-]\d+)", line)
    if m:
        live[m.group(1)] = [m.group(2), m.group(3), m.group(4), m.group(5), m.group(6).lstrip("+")]
fails = 0
for cells in tex_rows("tab:backend-behaviour"):
    if not cells:
        continue
    name = cells[0].replace("_", "\\_").strip()
    key = None
    for k in live:
        if k.replace("_", "") == name.replace("\\_", "").replace("_", ""):
            key = k
    if key is None:
        continue
    got = [canon(x) for x in live[key]]
    tex = [canon(x) for x in nums(cells)]
    # tex order: rows, %rows, e_in, e_out, net
    ok = (tex[:5] == got[:5])
    print("  %-12s tex=%s live=%s  %s" % (key, tex[:5], got, "OK" if ok else "MISMATCH"))
    if not ok:
        fails += 1

print()
print("=" * 96)
print("Table 10: gain split by pool reachability  (source: gain_split.py)")
print("=" * 96)
cases = [("ST-CMDS held-out", VC), ("THCHS-30", R / "e2eTHCHS.predictions.jsonl"),
         ("AISHELL dev", R / "dev_reranked.predictions.jsonl")]
liverows = []
for name, path in cases:
    o = run([PY, str(WS / ".dsh_checks/gain_split.py"), "--records", str(path), "--label", name])
    for line in o.splitlines():
        m = re.match(r"\s+ref (IN|NOT in) pool[^0-9]*(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([+-]\d+)", line)
        if m:
            liverows.append((name, "IN" if m.group(1) == "IN" else "NOT", 
                             m.group(2), m.group(3), m.group(4), m.group(5), m.group(6).lstrip("+")))
print("  live values from gain_split.py:")
for lr in liverows:
    print("    %-16s ref %-3s in pool  rows=%s chars=%s in=%s out=%s gain=%s" % lr)
tex10 = [nums(c) for c in tex_rows("tab:backend-gain") if nums(c)]
print("  tex values (rows, chars, in CER, out CER, gain):")
for t in tex10:
    print("    %s" % t)
live_flat = [[canon(lr[2]), canon(lr[3]), canon(lr[4]), canon(lr[5]), canon(lr[6])] for lr in liverows]
tex_flat = [[canon(x) for x in t] for t in tex10]
if live_flat == tex_flat:
    print("  MATCH: table 10 agrees with a fresh run")
else:
    print("  CHECK: compare the two lists above by hand")
    fails += 1

print()
print("VERDICT: %s" % ("both new tables reproduce from the scripts" if fails == 0
                       else "%d discrepancy/ies to inspect" % fails))
sys.exit(0)
