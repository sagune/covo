#!/usr/bin/env python3
"""Print one machine-readable verdict from a restore_eval log.

    get_cer.py --log <train_run_v2.log> --label-substr "ST-CMDS, trained adapter"
    -> CER=4.7123 RECALL=95.10 DESTROYED=0 EDITED=331 BEYOND_IMP=90 BEYOND_WOR=25

Exits 1 if no matching block with a restore[deployable] row is found, so a shell caller
can distinguish "target missed" from "the evaluation never ran".
"""
import argparse
import re
import sys
from pathlib import Path

LABEL = re.compile(r"^# (.+?)\s*$")
ROW = re.compile(r"^\s+restore\[deployable\]\s+CER\s+([0-9.]+)%\s*\|\s*(.*)$")
RECALL = re.compile(r"recall\s+([0-9.]+)%")
PROT = re.compile(r"\((\d+)/(\d+)\)")
DESTROY = re.compile(r"destroyed\s+(\d+)")
EDITED = re.compile(r"edited\s+(\d+)")
BEYOND = re.compile(r"beyond-N-best\s+([+-][0-9.]+) pp \(imp (\d+) wor (\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--label-substr", required=True)
    a = ap.parse_args()

    cur, hits = None, []
    for line in Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LABEL.match(line)
        if m:
            cur = m.group(1)
            continue
        if cur and a.label_substr in cur:
            m = ROW.match(line)
            if m:
                rest = m.group(2)
                hits.append((cur, float(m.group(1)), rest))
    if not hits:
        print("CER=none")
        return 1
    # last matching block wins (a repaired/re-run arm appears later in the log)
    label, cer, rest = hits[-1]
    out = ["CER=%.4f" % cer]
    for name, rx, fmt in (("RECALL", RECALL, "%s"), ("DESTROYED", DESTROY, "%s"),
                          ("EDITED", EDITED, "%s")):
        m = rx.search(rest)
        out.append("%s=%s" % (name, (fmt % m.group(1)) if m else "na"))
    m = PROT.search(rest)
    out.append("MENTIONS=%s" % ("%s/%s" % (m.group(1), m.group(2)) if m else "na"))
    m = BEYOND.search(rest)
    out.append("BEYOND=%s" % (m.group(1) if m else "na"))
    out.append("BEYOND_IMP=%s" % (m.group(2) if m else "na"))
    out.append("BEYOND_WOR=%s" % (m.group(3) if m else "na"))
    out.append('LABEL="%s"' % label)
    print(" ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
