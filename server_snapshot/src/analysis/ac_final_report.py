#!/usr/bin/env python
"""One-shot verdict for the AC-COVO run: the report to read when it finishes.

Collapses the whole decision into one command, so the outcome is read off numbers
rather than argued after the fact:

  1. the per-checkpoint AISHELL-dev table, and which checkpoint is best (the
     checkpoint-selection rule);
  2. the one-shot transfer rows for ST-CMDS and THCHS-30;
  3. the six pre-registered gates for ST-CMDS, via ac_gate_report.py;
  4. the section-15 SPECIFICITY test, which had no implementation until now.

On (4): the project constraint is that no change may be dataset-specific.  A gain
that appears only on ST-CMDS is adaptation to the target corpus, not a better
backend, and must be discarded even though the headline number improved.  So the
improvement is required to show up on more than one corpus.

Documented baselines (RESULTS sections 11 and 14):
    deployed V-C          ST-CMDS 4.9356 % recall 95.98 % destroyed 0
    text-only trained arm ST-CMDS 7.3109 % recall 94.99 % destroyed 17
    THCHS-30              baseline 3.1674 %, text-only trained 3.4238 %
    AISHELL dev           baseline ~2.2936 %, text-only trained 2.2984 %
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
PY = "/root/autodl-tmp/great/bin/python"
S = WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test"
T = WS / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
A = WS / "datasets/aishell/data_aishell_sensevoice/hotword/dev"

ROW = re.compile(
    r"^\s*(input|raw|verifier|restore\[deployable\]|restore\[annotation\])\s+"
    r"CER\s+([0-9.]+)%\s*\|\s*recall\s+([0-9.]+)%\s*\((\d+)/(\d+)\)\s*"
    r"destroyed\s+(\d+)\s*\|\s*edited\s+(\d+)\s*\|\s*beyond-N-best\s+([+-][0-9.]+)\s*pp")
HEAD = re.compile(r"^#\s*(.+?)\s*$")


def parse_log(path: Path):
    blocks, cur = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        h = HEAD.match(line.strip())
        if h and "CER" not in line:
            cur = {"label": h.group(1).strip(), "rows": {}}
            blocks.append(cur)
            continue
        if cur is not None:
            m = ROW.match(line)
            if m:
                cur["rows"][m.group(1)] = {
                    "cer_pct": float(m.group(2)), "recall_pct": float(m.group(3)),
                    "hits": int(m.group(4)), "total": int(m.group(5)),
                    "destroyed": int(m.group(6)), "edited": int(m.group(7)),
                    "beyond_nbest_pp": float(m.group(8))}
    return [b for b in blocks if b["rows"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(R / "acoustic_pipeline.log"))
    ap.add_argument("--stcmds-pred", default=str(R / "acoustic_v1/stcmds.predictions.jsonl"))
    args = ap.parse_args()

    blocks = parse_log(Path(args.log))
    if not blocks:
        print("no restore_eval blocks yet in %s" % args.log)
        print("(the run has not reached P4; nothing to report)")
        return

    print("=" * 100)
    print("AC-COVO FINAL REPORT")
    print("=" * 100)

    dev = [b for b in blocks if b["label"].lower().startswith("aishell dev")]
    others = [b for b in blocks if b not in dev]

    print("\n[1] AISHELL dev, per checkpoint (checkpoint selection set)")
    print("    %-34s %9s %9s %9s %7s %8s" % ("checkpoint", "dep_CER", "recall", "destroyed", "edited", "b-nbest"))
    best = None
    for b in dev:
        d = b["rows"].get("restore[deployable]", {})
        if not d:
            continue
        print("    %-34s %9.4f %9.2f %9d %7d %8.3f"
              % (b["label"], d["cer_pct"], d["recall_pct"], d["destroyed"],
                 d["edited"], d["beyond_nbest_pp"]))
        if best is None or d["cer_pct"] < best[1]["cer_pct"]:
            best = (b["label"], d)
    if best:
        print("    -> best checkpoint: %s (dev CER %.4f%%)" % (best[0], best[1]["cer_pct"]))

    print("\n[2] one-shot transfer")
    for b in others:
        d = b["rows"].get("restore[deployable]", {})
        if d:
            print("    %-34s CER %7.4f%%  recall %6.2f%%  destroyed %4d  edited %5d  b-nbest %+0.3f"
                  % (b["label"], d["cer_pct"], d["recall_pct"], d["destroyed"],
                     d["edited"], d["beyond_nbest_pp"]))

    print("\n[3] pre-registered gates for ST-CMDS", flush=True)
    if Path(args.stcmds_pred).exists():
        sys.stdout.flush()          # keep the subprocess output under its own heading
        subprocess.run([PY, str(WS / "src/analysis/ac_gate_report.py"),
                        "--records", args.stcmds_pred, "--label", "AC-COVO",
                        "--aligned", str(S / "aligned.txt"),
                        "--uttid-file", str(S / "uttid")])
    else:
        print("    (no ST-CMDS predictions at %s)" % args.stcmds_pred)

    # ------------------------------------------------------------------ section 15
    print("\n[4] section-15 SPECIFICITY test: the gain must not be ST-CMDS-only")
    base = {"stcmds": 4.9356, "thchs30": 3.1674, "aishell_dev": 2.2936}
    got = {}
    for b in blocks:
        d = b["rows"].get("restore[deployable]", {})
        if not d:
            continue
        low = b["label"].lower()
        if "stcmds" in low or "st-cmds" in low:
            got.setdefault("stcmds", d["cer_pct"])
        elif "thchs" in low:
            got.setdefault("thchs30", d["cer_pct"])
        elif low.startswith("aishell dev") and "final" in low:
            got.setdefault("aishell_dev", d["cer_pct"])

    if not got:
        print("    not enough arms yet to test specificity")
    else:
        improved = []
        for k, v in got.items():
            delta = v - base[k]
            flag = "improved" if delta < 0 else "worse   "
            print("    %-12s baseline %7.4f%%  this arm %7.4f%%  delta %+8.4f pp  %s"
                  % (k, base[k], v, delta, flag))
            if delta < 0:
                improved.append(k)
        print()
        if len(improved) >= 2:
            print("    VERDICT: generalises -- improvement on %s. Complies with the"
                  % ", ".join(improved))
            print("             constraint that the change must not be dataset-specific.")
        elif improved == ["stcmds"]:
            print("    VERDICT: DATASET-SPECIFIC -- only ST-CMDS improved. Under the")
            print("             section-15 constraint this is adaptation to the target")
            print("             corpus, not a better backend. DISCARD, however good the")
            print("             headline ST-CMDS number looks.")
        else:
            print("    VERDICT: no corpus improved; nothing to keep.")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
