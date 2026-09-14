#!/usr/bin/env python3
"""Parse train_run_v2.log into one table and pick the best checkpoint.

The orchestrator appends restore_eval.py output for every arm it evaluates:
    # AISHELL dev, checkpoint-2000
    rows 1334 | ...
      input                  CER  3.6821% | recall ... | edited   123 | ...
      restore[deployable]    CER  2.4123% | recall 95.20% (…) destroyed   1 | ...
This turns those blocks into a ranked table and answers the one question the
orchestrator cannot: is `final` actually the best checkpoint on AISHELL dev?

    consolidate_train.py [--log <path>] [--out <path>]
"""
import argparse
import re
import sys
from pathlib import Path

LABEL = re.compile(r"^# (.+?)\s*$")
ROW = re.compile(r"^\s+(\S+)\s+CER\s+([0-9.]+)%\s*\|\s*(.*)$")
RECALL = re.compile(r"recall\s+([0-9.]+)%")
DESTROY = re.compile(r"destroyed\s+(\d+)")
EDITED = re.compile(r"edited\s+(\d+)")
BEYOND = re.compile(r"beyond-N-best\s+([+-][0-9.]+) pp \(imp (\d+) wor (\d+)")
POLICY = "restore[deployable]"


def parse(path):
    blocks = []
    cur = None
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LABEL.match(line)
        if m:
            cur = {"label": m.group(1), "policies": {}}
            blocks.append(cur)
            continue
        m = ROW.match(line)
        if m and cur is not None:
            name, cer, rest = m.group(1), float(m.group(2)), m.group(3)
            r = RECALL.search(rest)
            d = DESTROY.search(rest)
            e = EDITED.search(rest)
            b = BEYOND.search(rest)
            cur["policies"][name] = dict(
                cer=cer,
                recall=float(r.group(1)) if r else None,
                destroyed=int(d.group(1)) if d else None,
                edited=int(e.group(1)) if e else None,
                beyond=float(b.group(1)) if b else None,
                imp=int(b.group(2)) if b else None,
                wor=int(b.group(3)) if b else None,
            )
    return [b for b in blocks if b["policies"]]


def cell(b, p=POLICY):
    r = b["policies"].get(p)
    if not r:
        return None
    parts = ["CER %7.4f%%" % r["cer"]]
    if r["recall"] is not None:
        parts.append("recall %5.2f%%" % r["recall"])
    if r["destroyed"] is not None:
        parts.append("destroyed %d" % r["destroyed"])
    if r["edited"] is not None:
        parts.append("edited %d" % r["edited"])
    if r["beyond"] is not None:
        parts.append("beyond %+.3f (imp %d wor %d)" % (r["beyond"], r["imp"], r["wor"]))
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/root/autodl-tmp/.dsh_checks/rerank/train_run_v2.log")
    ap.add_argument("--out", default="/root/autodl-tmp/.dsh_checks/rerank/TRAIN_SUMMARY.txt")
    a = ap.parse_args()

    blocks = parse(a.log)
    lines = []
    lines.append("=" * 108)
    lines.append("TRAINED-ADAPTER SUMMARY  (policy=%s, from %s)" % (POLICY, a.log))
    lines.append("=" * 108)
    if not blocks:
        lines.append("no restore_eval blocks found yet")
    for b in blocks:
        lines.append("")
        lines.append("%-44s %s" % (b["label"], cell(b) or "(no %s row)" % POLICY))

    dev = [b for b in blocks if b["label"].startswith("AISHELL dev") and POLICY in b["policies"]]
    lines.append("")
    lines.append("-" * 108)
    if dev:
        ranked = sorted(dev, key=lambda b: b["policies"][POLICY]["cer"])
        lines.append("AISHELL-dev ${POLICY} ranking (lower is better):")
        for i, b in enumerate(ranked):
            lines.append("  %d. %-40s %.4f%%" % (i + 1, b["label"], b["policies"][POLICY]["cer"]))
        best = ranked[0]["label"]
        fin = [b for b in dev if b["label"].endswith("final")]
        lines.append("")
        lines.append("BEST_CKPT=%s" % best)
        if fin:
            fc = fin[0]["policies"][POLICY]["cer"]
            bc = ranked[0]["policies"][POLICY]["cer"]
            lines.append("final=%.4f%%  best=%.4f%%  margin=%.4f pp" % (fc, bc, fc - bc))
            lines.append("VERDICT=%s" % ("keep final" if fc - bc < 0.03 else "final is NOT best - retrain-on-best advised"))
    else:
        lines.append("no AISHELL-dev block yet")

    st = [b for b in blocks if b["label"].startswith("ST-CMDS")]
    th = [b for b in blocks if b["label"].startswith("THCHS")]
    lines.append("")
    lines.append("-" * 108)
    for name, grp, base in (("ST-CMDS (target 4.5000%)", st, 4.9356), ("THCHS-30", th, 3.1674)):
        lines.append("%s   existing-adapter baseline %.4f%%" % (name, base))
        for b in grp:
            c = cell(b)
            if c:
                lines.append("   %-42s %s" % (b["label"], c))
        if grp and POLICY in grp[-1]["policies"]:
            got = grp[-1]["policies"][POLICY]["cer"]
            lines.append("   >>> delta vs baseline = %+.4f pp   %s" % (
                got - base, "TARGET MET" if got <= 4.5 else "target NOT met (need %.4f pp more)" % (got - 4.5)))

    txt = "\n".join(lines)
    print(txt)
    Path(a.out).write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
