#!/usr/bin/env python3
"""One-off: mark the superseded interface_audit.py as deprecated in place (v2).

v1 forgot the closing triple quote and broke the file; restore from git first.
"""
import subprocess
from pathlib import Path

REPO = Path("/root/autodl-tmp")
REL = "src/scripts/rerank_20260914/interface_audit.py"
p = REPO / REL

subprocess.run(["git", "-C", str(REPO), "checkout", "--", REL], check=True)
s = p.read_text(encoding="utf-8")

NOTE = (
    '"""DEPRECATED -- use interface_audit2.py instead.\n'
    "\n"
    "This version assumed the prompt carries a CB-SenseVoice candidate-scores block\n"
    "rendered in original rank order.  It does not: compact_evidence defaults to True and\n"
    "format_candidates is only emitted when compact is false\n"
    "(cbsensevoice_covo_bridge.py:1140), so that block never appears in our prompts.  The\n"
    "metric computed here therefore has no interface consequence.  Kept only for\n"
    "provenance; interface_audit2.py measures the real artifact -- a score inversion\n"
    "between the first two n-best lines.\n"
    "\n"
    "The original docstring follows.\n"
    '"""\n'
    "\n"
)
marker = '"""Interface audit'
assert marker in s, "marker not found"

p.write_text(NOTE + s, encoding="utf-8")
out = subprocess.run(["python3", "-m", "py_compile", str(p)], capture_output=True, text=True)
if out.returncode != 0:
    out = subprocess.run(["/root/autodl-tmp/great/bin/python", "-m", "py_compile", str(p)],
                         capture_output=True, text=True)
print("compiles:", out.returncode == 0, out.stderr.strip()[:200])
print("bytes:", len(p.read_text(encoding="utf-8")))
print("---- head ----")
print("\n".join(p.read_text(encoding="utf-8").splitlines()[:18]))
print("---- the original docstring still opens ----")
print("\n".join(p.read_text(encoding="utf-8").splitlines()[18:24]))
