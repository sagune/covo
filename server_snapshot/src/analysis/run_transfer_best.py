#!/usr/bin/env python
"""One-shot transfer using the checkpoint that AISHELL dev actually selected.

The objective requires the checkpoint to be chosen on AISHELL dev, but the pipeline's
P5 transfers `$OUT/final` unconditionally -- which is only correct if the final model
also happens to be the best on dev. If the model peaks before the end of the epoch,
the pre-registered protocol wants that earlier checkpoint, and using `final` would
silently report the wrong arm.

This re-runs the transfer with the best-on-dev checkpoint (or `final` when that is
genuinely best), writing to *_best.predictions.jsonl so it never collides with the
pipeline's own outputs.

Run it AFTER P4 has produced dev metrics. Needs the GPU to be free.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
OUT = R / "acoustic_v1"
COVO = WS / "cbwhisper_covo_migration_20260609_tar_extracted/covo"
MODEL = WS / "cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
PY = "/root/autodl-tmp/great/bin/python"
S = WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test"
T = WS / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"

ROW = re.compile(
    r"^\s*restore\[deployable\]\s+CER\s+([0-9.]+)%\s*\|\s*recall\s+([0-9.]+)%\s*"
    r"\((\d+)/(\d+)\)\s*destroyed\s+(\d+)\s*\|\s*edited\s+(\d+)\s*\|\s*"
    r"beyond-N-best\s+([+-][0-9.]+)\s*pp")
HEAD = re.compile(r"^#\s*(AISHELL dev[^\n]*)$")


def best_dev_checkpoint(log: Path, out_dir: Path):
    """-> (label, dir, cer) for the lowest-CER AISHELL-dev arm in the log."""
    cur, found = None, []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        h = HEAD.match(line.strip())
        if h:
            cur = h.group(1).strip()
            continue
        m = ROW.match(line)
        if m and cur:
            found.append((cur, float(m.group(1))))
            cur = None
    if not found:
        return None
    label, cer = min(found, key=lambda kv: kv[1])
    # The pipeline log labels dev arms as "# AISHELL dev checkpoint-2000" -- NO comma --
    # while an earlier run used "# AISHELL dev, checkpoint-2163".  Splitting on the comma
    # therefore returned the WHOLE string as the directory tag under the first format,
    # which is why the best-on-dev transfer reported "adapter or projector missing"
    # instead of running.  Match the tag explicitly instead.
    m = re.search(r"(checkpoint-\d+|final)\s*$", label.strip())
    tag = m.group(1) if m else label.strip().split()[-1]
    d = out_dir / "final" if tag == "final" else out_dir / "final" / tag
    return label, d, cer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(R / "acoustic_pipeline.log"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)

    pick = best_dev_checkpoint(Path(a.log), out_dir)
    if not pick:
        print("no AISHELL-dev metrics in %s yet -- run this after P4" % a.log)
        return 1
    label, ckpt, cer = pick
    proj = ckpt / "acoustic_projector.pt"
    print("best-on-dev checkpoint: %s  (dev CER %.4f%%)" % (label, cer))
    print("  adapter dir : %s" % ckpt)
    print("  projector   : %s" % proj)
    if not (ckpt / "adapter_config.json").exists() or not proj.exists():
        print("  FATAL: adapter or projector missing at that path")
        return 2
    if a.dry_run:
        return 0

    jobs = [
        ("stcmds", R / "e2eSTCMDS_VA.messages.ac.jsonl", S / "aligned.txt", S / "uttid"),
        ("thchs30", OUT / "eval_thchs.messages.ac.jsonl", T / "aligned.txt", T / "uttid"),
    ]
    for name, rows, aligned, uttid in jobs:
        if not rows.exists():
            print("[%s] SKIP: prompt file missing (%s)" % (name, rows))
            continue
        pred = OUT / ("%s_best.predictions.jsonl" % name)
        env = {"PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": "0",
               "PYTHONPATH": str(COVO / "src")}
        import os
        full = dict(os.environ)
        full.update(env)
        print("[%s] running inference with %s ..." % (name, label), flush=True)
        with pred.open("w", encoding="utf-8") as fh:
            rc = subprocess.run(
                [PY, str(COVO / "scripts/infer_acoustic_covo.py"),
                 "--input", str(rows), "--output", str(pred),
                 "--model-name-or-path", str(MODEL),
                 "--adapter-path", str(ckpt), "--projector-path", str(proj),
                 "--acoustic-bin", str(R / ("ac_%s.bin" % ("stcmds" if name == "stcmds" else "thchs30"))),
                 "--acoustic-index", str(R / ("ac_%s.jsonl" % ("stcmds" if name == "stcmds" else "thchs30"))),
                 "--batch-size", "4", "--max-new-tokens", "96", "--disable-thinking"],
                stdout=fh, stderr=subprocess.STDOUT, env=full).returncode
        print("[%s] inference exit=%d rows=%d" % (name, rc, sum(1 for _ in pred.open(encoding="utf-8"))))
        p = subprocess.run([PY, str(WS / ".dsh_checks/restore_eval.py"),
                            "--records", str(pred), "--label", "%s (best-on-dev)" % name,
                            "--aligned", str(aligned), "--uttid-file", str(uttid)],
                           capture_output=True, text=True)
        sys.stdout.write(p.stdout)
        sys.stdout.write(p.stderr)
    print("\nNOTE: these are the protocol-compliant transfer numbers (checkpoint chosen "
          "on AISHELL dev). The pipeline's own stcmds/thchs30 arms used `final`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
