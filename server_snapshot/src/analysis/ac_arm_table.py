#!/usr/bin/env python
"""Build the ST-CMDS comparison table over every arm that has predictions on disk.

A single number for AC-COVO means little without the arms it has to beat, and the
paper needs them side by side under the SAME protocol (`restore[deployable]`, the
same aligned.txt and uttid file). This runs restore_eval across all discovered arms
and prints one markdown table, so the comparison cannot drift between runs.

CPU only. Usage:

    python src/analysis/ac_arm_table.py                 # auto-discover
    python src/analysis/ac_arm_table.py --arm label=path ...
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
PY = "/root/autodl-tmp/great/bin/python"
S = WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test"

ROW = re.compile(
    r"\s*(input|raw|verifier|restore\[deployable\]|restore\[annotation\])\s+"
    r"CER\s+([0-9.]+)%\s*\|\s*recall\s+([0-9.]+)%\s*\((\d+)/(\d+)\)\s*"
    r"destroyed\s+(\d+)\s*\|\s*edited\s+(\d+)\s*\|\s*beyond-N-best\s+([+-][0-9.]+)\s*pp")

# documented in RESULTS; listed so the table is not missing its reference point.
# Reproduced exactly by e2eSTCMDS_VC.predictions, which is the check that this
# generator agrees with the recorded numbers.
DOCUMENTED = [
    ("baseline V-C (deployed)", 4.9356, 95.98, 0, 0, 0.191),
]


def evaluate(path: Path, label: str):
    p = subprocess.run([PY, str(WS / ".dsh_checks/restore_eval.py"),
                        "--records", str(path), "--label", label,
                        "--aligned", str(S / "aligned.txt"),
                        "--uttid-file", str(S / "uttid")],
                       capture_output=True, text=True)
    for line in (p.stdout + p.stderr).splitlines():
        m = ROW.match(line)
        if m and m.group(1) == "restore[deployable]":
            return {"cer": float(m.group(2)), "recall": float(m.group(3)),
                    "destroyed": int(m.group(6)), "edited": int(m.group(7)),
                    "beyond": float(m.group(8))}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="label=path; may be repeated. Default: auto-discover.")
    ap.add_argument("--include-final", action="store_true",
                    help="also include acoustic_v1/stcmds.predictions.jsonl")
    a = ap.parse_args()

    arms = []
    if a.arm:
        for spec in a.arm:
            label, _, path = spec.partition("=")
            arms.append((label, Path(path)))
    else:
        seen = set()
        for pat in ("*STCMDS*.predictions.jsonl", "stcmds_*.predictions.jsonl"):
            for f in sorted(R.glob(pat)):
                if "smoke" in f.name or f.name in seen:
                    continue
                seen.add(f.name)
                arms.append((f.stem, f))
        for extra, lab in ((R / "acoustic_v1/stcmds.predictions.jsonl", "AC-COVO (acoustic)"),):
            if a.include_final and extra.exists():
                arms.append((lab, extra))

    print("| arm | CER % | recall % | destroyed | edited | beyond-N-best pp |")
    print("|---|---:|---:|---:|---:|---:|")
    base = DOCUMENTED[0]
    print("| **%s** | %.4f | %.2f | %d | %d | %+.3f |"
          % (base[0], base[1], base[2], base[3], base[4], base[5]))

    results = []
    for label, path in arms:
        if not path.exists():
            print("| %s | _missing_ | | | | |" % label)
            continue
        m = evaluate(path, label)
        if not m:
            print("| %s | _no restore row_ | | | | |" % label)
            continue
        results.append((label, m))
        print("| %s | %.4f | %.2f | %d | %d | %+.3f |"
              % (label, m["cer"], m["recall"], m["destroyed"], m["edited"], m["beyond"]))

    print()
    if results:
        best = min(results, key=lambda kv: kv[1]["cer"])
        print("best CER among evaluated arms: %s (%.4f%%)" % (best[0], best[1]["cer"]))
        better = [l for l, m in results if m["cer"] < base[1]]
        print("arms beating the deployed baseline (%.4f%%): %s"
              % (base[1], ", ".join(better) if better else "NONE"))


if __name__ == "__main__":
    main()
