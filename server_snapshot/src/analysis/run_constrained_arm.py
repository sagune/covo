#!/usr/bin/env python
"""Evaluate the constrained-decoding arm. Run after the pipeline, with the GPU free.

The constrained arm has allowed sets only for ST-CMDS, because that is the only
corpus whose per-character acoustic top-k had been extracted. That makes it
ineligible for the section-15 specificity test, which needs more than one corpus --
and constraining is exactly the intervention whose whole point is to suppress the
off-pool hallucination that helped kill the earlier SFT arm.

So this script fills the gap: for any corpus missing its top-k it runs the frozen
extractor and builds the sets, then evaluates the arm.

Cheapest useful probe is ST-CMDS alone, whose sets already exist:

    python run_constrained_arm.py --corpora stcmds --limit 400   # quick look
    python run_constrained_arm.py --corpora stcmds               # full
    python run_constrained_arm.py --corpora stcmds,thchs30,aishell_dev   # for section 15
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path("/root/autodl-tmp")
R = WS / ".dsh_checks/rerank"
OUT = R / "acoustic_v1"
COVO = WS / "cbwhisper_covo_migration_20260609_tar_extracted/covo"
MODEL = WS / "cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B"
PY = "/root/autodl-tmp/great/bin/python"
A = WS / "datasets/aishell/data_aishell_sensevoice/hotword/dev"
T = WS / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S = WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test"

SPEC = {
    "stcmds": {
        "pool": R / "stcmds_va_tok_input.jsonl",
        "manifest": R / "stcmds_manifest.jsonl",
        "uttid_file": None,
        "messages": R / "e2eSTCMDS_VA.messages.ac.jsonl",
        "aligned": S / "aligned.txt", "uttid": S / "uttid",
        "ac_bin": R / "ac_stcmds.bin", "ac_idx": R / "ac_stcmds.jsonl",
    },
    "thchs30": {
        "pool": R / "thchs_learned_v2.jsonl",
        "manifest": R / "thchs30_positional_manifest.jsonl",
        "uttid_file": None,
        "messages": OUT / "eval_thchs.messages.ac.jsonl",
        "aligned": T / "aligned.txt", "uttid": T / "uttid",
        "ac_bin": R / "ac_thchs30.bin", "ac_idx": R / "ac_thchs30.jsonl",
    },
    "aishell_dev": {
        "pool": R / "aishell_relax_learned_v2.jsonl",
        "manifest": R / "aishell_dev_positional_manifest.jsonl",
        "uttid_file": None,
        "messages": OUT / "eval_aishell.messages.ac.jsonl",
        "aligned": A / "aligned.txt", "uttid": A / "uttid",
        "ac_bin": R / "ac_aishell_dev.bin", "ac_idx": R / "ac_aishell_dev.jsonl",
    },
}


def env():
    e = dict(os.environ)
    e.update({"PYTORCH_ALLOC_CONF": "expandable_segments:True",
              "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
              "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(COVO / "src")})
    return e


def ensure_sets(name, spec):
    top5 = R / ("%s_top5.jsonl" % name)
    sets = R / ("%s_allowed_sets.jsonl" % name)
    if not sets.exists():
        if not top5.exists():
            print("[%s] extracting per-character acoustic top-k ..." % name, flush=True)
            cmd = [PY, str(WS / "src/analysis/score_sensevoice_tokstats.py"),
                   "--input", str(spec["pool"]), "--manifest", str(spec["manifest"]),
                   "--output", str(top5), "--token-stats-topk", "1",
                   "--progress-every", "500"]
            if subprocess.run(cmd, env=env()).returncode != 0:
                print("[%s] top-k extraction FAILED" % name)
                return None
        print("[%s] building allowed sets ..." % name, flush=True)
        if subprocess.run([PY, str(WS / "src/analysis/build_allowed_sets.py"),
                           "--top5", str(top5), "--out", str(sets)],
                          env=env()).returncode != 0:
            print("[%s] allowed-set build FAILED" % name)
            return None
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="stcmds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arm", default="acoustic_v1/final",
                    help="adapter dir under .dsh_checks/rerank, or a checkpoint path")
    a = ap.parse_args()

    ckpt = R / a.arm
    proj = ckpt / "acoustic_projector.pt"
    if not proj.exists():
        print("FATAL: no projector at %s" % proj)
        raise SystemExit(2)

    for name in [c.strip() for c in a.corpora.split(",") if c.strip()]:
        spec = SPEC.get(name)
        if spec is None:
            print("[%s] unknown corpus" % name)
            continue
        sets = ensure_sets(name, spec)
        if sets is None or not sets.exists():
            continue
        rows = spec["messages"]
        if not rows.exists():
            print("[%s] SKIP: prompt file missing (%s)" % (name, rows))
            continue
        pred = OUT / ("%s_constrained.predictions.jsonl" % name)
        cmd = [PY, str(COVO / "scripts/infer_acoustic_covo.py"),
               "--input", str(rows), "--output", str(pred),
               "--model-name-or-path", str(MODEL),
               "--adapter-path", str(ckpt), "--projector-path", str(proj),
               "--acoustic-bin", str(spec["ac_bin"]), "--acoustic-index", str(spec["ac_idx"]),
               "--constrained-sets", str(sets),
               "--batch-size", "4", "--max-new-tokens", "96", "--disable-thinking"]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print("[%s] running CONSTRAINED inference ..." % name, flush=True)
        rc = subprocess.run(cmd, env=env()).returncode
        print("[%s] inference exit=%d" % (name, rc))
        if rc != 0 or not pred.exists():
            continue
        p = subprocess.run([PY, str(WS / ".dsh_checks/restore_eval.py"),
                            "--records", str(pred), "--label", "%s constrained" % name,
                            "--aligned", str(spec["aligned"]), "--uttid-file", str(spec["uttid"])],
                           capture_output=True, text=True)
        sys.stdout.write(p.stdout + p.stderr)


if __name__ == "__main__":
    main()
