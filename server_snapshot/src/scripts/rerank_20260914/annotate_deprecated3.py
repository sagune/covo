#!/usr/bin/env python3
"""One-off v3: insert the deprecation note AFTER the shebang line."""
import subprocess
from pathlib import Path

REPO = Path("/root/autodl-tmp")
REL = "src/scripts/rerank_20260914/interface_audit.py"
p = REPO / REL
subprocess.run(["git", "-C", str(REPO), "checkout", "--", REL], check=True)

lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
assert lines and lines[0].startswith("#!"), "no shebang on line 1"

NOTE = [
    '"""DEPRECATED -- use interface_audit2.py instead.\n',
    "\n",
    "This version assumed the prompt carries a CB-SenseVoice candidate-scores block\n",
    "rendered in original rank order.  It does not: compact_evidence defaults to True and\n",
    "format_candidates is only emitted when compact is false\n",
    "(cbsensevoice_covo_bridge.py:1140), so that block never appears in our prompts.  The\n",
    "metric computed here therefore has no interface consequence.  Kept only for\n",
    "provenance; interface_audit2.py measures the real artifact -- a score inversion\n",
    "between the first two n-best lines.\n",
    "\n",
    "The original docstring follows.\n",
    '"""\n',
    "\n",
]
p.write_text("".join([lines[0]] + NOTE + lines[1:]), encoding="utf-8")

out = subprocess.run(["/root/autodl-tmp/great/bin/python", "-m", "py_compile", str(p)],
                     capture_output=True, text=True)
print("compiles:", out.returncode == 0, out.stderr.strip()[:200])
txt = p.read_text(encoding="utf-8").splitlines()
print("line1:", txt[0])
print("line2:", txt[1])
print("line16-18:", txt[15:18])
