#!/usr/bin/env python3
"""One-off: mark the superseded interface_audit.py as deprecated in place."""
from pathlib import Path

p = Path("/root/autodl-tmp/src/scripts/rerank_20260914/interface_audit.py")
s = p.read_text(encoding="utf-8")
if "DEPRECATED" in s:
    print("already annotated")
else:
    note = (
        '"""DEPRECATED -- use interface_audit2.py instead.\n'
        "\n"
        "This version assumed the prompt contains a CB-SenseVoice candidate-scores block\n"
        "rendered in original rank order.  It does not: compact_evidence defaults to True\n"
        "and format_candidates is only emitted when compact is false\n"
        "(cbsensevoice_covo_bridge.py:1140), so that block never appears in our prompts.\n"
        "The metric computed here therefore has no interface consequence.  Kept only for\n"
        "provenance; interface_audit2.py measures the real artifact (a score inversion\n"
        "between the first two n-best lines).\n"
        "\n"
        "Original docstring follows.\n"
        "\n"
    )
    marker = '"""Interface audit'
    if marker not in s:
        raise SystemExit("marker not found; refusing to edit")
    p.write_text(note + s.replace(marker, marker, 1), encoding="utf-8")
    print("annotated, %d -> %d bytes" % (len(s), len(p.read_text(encoding="utf-8"))))
