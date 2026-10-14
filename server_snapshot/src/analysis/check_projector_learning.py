#!/usr/bin/env python
"""Is the acoustic channel actually earning its influence?

The projector's last layer is zero-initialised on purpose: at step 0 the model is
numerically identical to the text-only model, so training starts at the baseline.
The cost of that choice is a specific failure mode -- LoRA can learn to solve the
task with the channel permanently switched off, leaving the projector at ~0. The
run would then finish, report "no improvement", and look like the acoustic
information does not help, when in fact it was never used.

This script measures the channel's progress from a saved checkpoint, so that
possibility is visible in ~50 minutes instead of after the whole epoch. It needs
no GPU and no model weights, only the projector file and a few acoustic vectors.

Reported:
  * the last layer's weight norm -- exactly 0.0 at initialisation, so any growth
    is the channel turning on;
  * the first layer's weight norm, to show the projection is being shaped;
  * the RMS of the bias the projector actually produces on real acoustic vectors,
    which is the quantity that perturbs the character embeddings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

B = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo")
sys.path.insert(0, str(B / "src"))

from acoustic_conditioning import AcousticProjector, AcousticStore  # noqa: E402


def describe(sd, label):
    last_w = [k for k in sd if k.endswith("net.%d.weight" % (len(sd) // 2))]
    print("  %s" % label)
    for k in sorted(sd):
        if "weight" in k or "bias" in k:
            t = sd[k].float()
            print("    %-18s shape %-16s norm %.6f  mean|.| %.6f"
                  % (k, tuple(t.shape), float(t.norm()), float(t.abs().mean())))
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projector", required=True, help="acoustic_projector.pt")
    ap.add_argument("--bin", default="/root/autodl-tmp/.dsh_checks/rerank/ac_aishell_dev.bin")
    ap.add_argument("--index", default="/root/autodl-tmp/.dsh_checks/rerank/ac_aishell_dev.jsonl")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--embed-rms", type=float, default=0.013466,
                    help="Per-element RMS of Qwen3.5-9B embed_tokens.weight, measured "
                         "from the checkpoint (row norm 0.8524 over 4096 dims). The "
                         "injected bias is added element-wise to those embeddings, so "
                         "this is the scale it has to be comparable to. Note the "
                         "embeddings are TINY: a bias RMS of 1e-3 is already ~7%% of "
                         "the embedding scale, and 0.33 (a randomly-perturbed "
                         "projector) would be 24x too large.")
    a = ap.parse_args()

    sd = torch.load(a.projector, map_location="cpu")
    print("checkpoint projector: %s" % a.projector)

    keys = sorted(k for k in sd if k.endswith(".weight") and "net." in k)
    last_key = keys[-1] if keys else None
    first_key = keys[0] if keys else None
    last_norm = float(sd[last_key].float().norm()) if last_key else float("nan")
    first_norm = float(sd[first_key].float().norm()) if first_key else float("nan")

    print()
    print("=== has the channel switched on? ===")
    print("  first layer weight norm : %.6f   (%s)" % (first_norm, first_key))
    print("  last  layer weight norm : %.6f   (%s)" % (last_norm, last_key))
    print("  -> at initialisation the last layer is EXACTLY 0.0, so the model was")
    print("     numerically the text-only model. Any growth here is the acoustic")
    print("     channel beginning to contribute.")

    # actually push real acoustic vectors through it
    d_model = int(sd[last_key].shape[0]) if last_key else 4096
    proj = AcousticProjector(a.dim, d_model, a.layers)
    proj.load_state_dict(sd)
    proj.eval()

    store = AcousticStore(a.bin, a.index)
    n = min(a.rows, len(store))
    vecs = []
    for i in range(n):
        v = store.vectors(i)
        if v.size:
            vecs.append(torch.from_numpy(v))
    if not vecs:
        print("\nno acoustic vectors available to probe")
        return
    cat = torch.cat(vecs, 0)
    with torch.no_grad():
        bias = proj(cat)
    bias_rms = float(bias.pow(2).mean().sqrt())
    print()
    print("=== the bias the projector actually injects (%d real characters) ===" % cat.size(0))
    print("  acoustic input  RMS : %.6f" % float(cat.pow(2).mean().sqrt()))
    print("  injected bias   RMS : %.6f" % bias_rms)
    print("  injected bias   absmax : %.6f" % float(bias.abs().max()))
    print()
    ratio = bias_rms / a.embed_rms if a.embed_rms else 0.0
    print("  embedding RMS (yardstick) : %.6f" % a.embed_rms)
    print("  bias / embedding scale    : %.4f  (%.2f%% of the embedding scale)"
          % (ratio, 100.0 * ratio))

    print()
    if last_norm == 0.0:
        print("VERDICT: INERT -- the last layer is still exactly zero at this checkpoint.")
        print("         The channel is switched off and the run is, so far, a text-only")
        print("         run wearing a projector. Train the projector alone (--freeze-llm).")
    elif bias_rms < 1e-5:
        print("VERDICT: NEGLIGIBLE -- the channel moved off zero but injects a bias far")
        print("         below any scale that could alter the embeddings.")
    elif ratio < 0.05:
        print("VERDICT: WEAK -- the bias is under 5%% of the embedding scale; it may be")
        print("         present in the weights without meaningfully conditioning generation.")
    else:
        print("VERDICT: ACTIVE -- the bias is >=5% of the embedding scale, so generation is")
        print("         genuinely conditioned on acoustics and the result is interpretable.")


if __name__ == "__main__":
    main()
