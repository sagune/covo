#!/usr/bin/env python
"""Within-model ablation: the trained adapter WITH and WITHOUT the acoustic channel.

Every other comparison in this project is between separately trained models, so a
difference could come from the data, the optimiser, or the LoRA. This one holds all
of that fixed: same trained adapter, same prompts, same decoding -- the only change is
whether the projector contributes.

Zeroing the whole projector state dict makes `proj(x) == 0` exactly (LayerNorm with
zeroed weight gives 0, every Linear with zeroed weight and bias gives 0), so the
acoustic term vanishes while the language model is untouched. The CER difference is
therefore the acoustic channel's contribution and nothing else.

This needs no new inference code: the existing script takes --projector-path, so a
zeroed copy is enough.

    python run_acoustic_ablation.py --corpus stcmds
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

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

SPEC = {
    "stcmds": dict(rows=R / "e2eSTCMDS_VA.messages.ac.jsonl", ac_bin=R / "ac_stcmds.bin",
                   ac_idx=R / "ac_stcmds.jsonl", aligned=S / "aligned.txt", uttid=S / "uttid"),
    "thchs30": dict(rows=OUT / "eval_thchs.messages.ac.jsonl", ac_bin=R / "ac_thchs30.bin",
                    ac_idx=R / "ac_thchs30.jsonl", aligned=T / "aligned.txt", uttid=T / "uttid"),
}


def env():
    e = dict(os.environ)
    e.update({"PYTORCH_ALLOC_CONF": "expandable_segments:True", "TRANSFORMERS_OFFLINE": "1",
              "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0",
              "PYTHONPATH": str(COVO / "src")})
    return e


def evaluate(pred, aligned, uttid, label):
    p = subprocess.run([PY, str(WS / ".dsh_checks/restore_eval.py"), "--records", str(pred),
                        "--label", label, "--aligned", str(aligned), "--uttid-file", str(uttid)],
                       capture_output=True, text=True)
    for line in (p.stdout + p.stderr).splitlines():
        m = ROW.match(line)
        if m:
            return {"cer": float(m.group(1)), "recall": float(m.group(2)),
                    "destroyed": int(m.group(5)), "edited": int(m.group(6))}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="stcmds")
    ap.add_argument("--arm", default="final")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    ckpt = OUT / a.arm
    trained = ckpt / "acoustic_projector.pt"
    if not trained.exists():
        print("FATAL: no trained projector at %s (training has not finished)" % trained)
        raise SystemExit(2)

    zero = ckpt / "acoustic_projector_ZERO.pt"
    sd = torch.load(trained, map_location="cpu")
    torch.save({k: torch.zeros_like(v) for k, v in sd.items()}, zero)
    print("wrote zeroed projector -> %s" % zero)
    # sanity: confirm the zeroed projector really is a no-op
    sys.path.insert(0, str(COVO / "src"))
    from acoustic_conditioning import AcousticProjector
    dim_out = int(sd[[k for k in sd if k.endswith("weight")][-1]].shape[0])
    pr = AcousticProjector(512, dim_out, 2)
    pr.load_state_dict(torch.load(zero, map_location="cpu"))
    with torch.no_grad():
        out = pr(torch.randn(4, 512))
    print("zeroed projector max |output| = %.3g (must be exactly 0)" % float(out.abs().max()))
    if float(out.abs().max()) != 0.0:
        print("FATAL: the zeroed projector is not a no-op; ablation would be meaningless")
        raise SystemExit(3)

    spec = SPEC[a.corpus]
    results = {}
    for tag, proj in (("with acoustics", trained), ("without acoustics", zero)):
        pred = OUT / ("%s_ablation_%s.predictions.jsonl"
                      % (a.corpus, "on" if proj is trained else "off"))
        cmd = [PY, str(COVO / "scripts/infer_acoustic_covo.py"),
               "--input", str(spec["rows"]), "--output", str(pred),
               "--model-name-or-path", str(MODEL), "--adapter-path", str(ckpt),
               "--projector-path", str(proj), "--acoustic-bin", str(spec["ac_bin"]),
               "--acoustic-index", str(spec["ac_idx"]),
               "--batch-size", "4", "--max-new-tokens", "96", "--disable-thinking"]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        print("[%s] %s ..." % (a.corpus, tag), flush=True)
        if subprocess.run(cmd, env=env()).returncode != 0:
            print("  inference FAILED")
            continue
        m = evaluate(pred, spec["aligned"], spec["uttid"], "%s %s" % (a.corpus, tag))
        if m:
            results[tag] = m
            print("  %-18s CER %7.4f%%  recall %6.2f%%  destroyed %4d  edited %5d"
                  % (tag, m["cer"], m["recall"], m["destroyed"], m["edited"]))

    if len(results) == 2:
        on, off = results["with acoustics"], results["without acoustics"]
        d = off["cer"] - on["cer"]
        print()
        print("ACOUSTIC CHANNEL CONTRIBUTION on %s: %+.4f pp "
              "(with %.4f%% vs without %.4f%%)" % (a.corpus, d, on["cer"], off["cer"]))
        print("  positive => the channel helps. Everything else (data, LoRA, prompts,")
        print("  decoding) is identical between the two, so this difference is the")
        print("  acoustic channel and nothing else.")


if __name__ == "__main__":
    main()
