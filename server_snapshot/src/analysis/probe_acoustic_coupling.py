#!/usr/bin/env python
"""Checkpoint 0 of AC-COVO: what shape does the coupling component have to bridge?

The projector must turn per-frame acoustic state into soft tokens, but the
alignment frames we measured in RESULTS section 17 live in the CTC matrix, while
the rich representation lives in the encoder output.  Before writing any model
code we need hard answers to:

  * encoder output width D and how many frames T it produces per utterance,
  * whether the CTC matrix frame count equals the encoder frame count
    (if not, every per-frame measurement would need a resampling map),
  * how many of the leading frames are the 4 query tokens (textnorm, language,
    event, emotion) that must be stripped before projecting,
  * measured per-utterance cost, to size the 17,301-row AISHELL export and the
    fp16 storage it needs.

It also checks the L2 norm of the raw hidden states, because the existing
`hs_utils.quantize_hidden_states` path CLAMPS to [-1, 1] and int8-quantises --
fine for the KWS task it was written for, but lossy for a projector, so this
script reports how much mass that clamp would destroy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/.dsh_checks")
from score_sensevoice_tokstats import audio_map, encode_audio  # noqa: E402


def encoder_forward(model, frontend, wav, device, language, textnorm):
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    audio = load_audio_text_image_video(
        wav, fs=frontend.fs, audio_fs=None, data_type="sound",
        tokenizer=None,
    )
    speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
    speech = speech.to(device)
    speech_lengths = speech_lengths.to(device)

    n = speech.size(0)
    language_query = model.embed(
        torch.LongTensor([[model.lid_dict[language] if language in model.lid_dict else 0]]).to(device)
    ).repeat(n, 1, 1)
    textnorm_query = model.embed(
        torch.LongTensor([[model.textnorm_dict[textnorm]]]).to(device)
    ).repeat(n, 1, 1)
    x = torch.cat((textnorm_query, speech), dim=1)
    lengths = speech_lengths + 1
    event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(device)).repeat(n, 1, 1)
    input_query = torch.cat((language_query, event_emo_query), dim=1)
    x = torch.cat((input_query, x), dim=1)
    lengths = lengths + 3

    with torch.no_grad():
        enc, enc_lens = model.encoder(x, lengths)
    if isinstance(enc, tuple):
        enc = enc[0]
    return enc, int(enc_lens[0]), int(speech.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--textnorm", default="withitn")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    from funasr import AutoModel

    wrapper = AutoModel(model=a.model, device="cuda:0", disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = next(model.parameters()).device

    au = audio_map(a.manifest)
    keys = list(au)[: a.n]
    print("probing %d utterances" % len(keys), flush=True)

    recs = []
    t0 = time.monotonic()
    for rid in keys:
        wav = au[rid]
        lp = encode_audio(model, tokenizer, frontend, wav, device, False)
        lpn = lp.detach().float().cpu().numpy()
        enc, enc_len, n_bins = encoder_forward(model, frontend, wav, device, a.language, a.textnorm)
        e = enc[0, :enc_len].detach().float().cpu()
        norms = e.norm(dim=-1)
        recs.append({
            "id": rid,
            "ctc_frames": int(lpn.shape[0]),
            "vocab": int(lpn.shape[1]),
            "enc_frames": int(enc.shape[1]),
            "enc_len": enc_len,
            "enc_dim": int(enc.shape[2]),
            "input_bins": n_bins,
            "norm_mean": float(norms.mean()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            # how much of the representation would the int8 clamp destroy
            "frac_abs_gt_1": float((e.abs() > 1.0).float().mean()),
        })
    elapsed = time.monotonic() - t0

    D = recs[0]["enc_dim"]
    frames = [r["ctc_frames"] for r in recs]
    eframes = [r["enc_len"] for r in recs]
    print()
    print("=== shapes ===")
    print("  encoder_output_size (model.encoder_output_size): %s" % getattr(model, "encoder_output_size", "?"))
    print("  encoder dim D            : %d" % D)
    print("  per-utterance seconds    : %.3f" % (elapsed / len(recs)))
    print("  CTC frames  min/med/max  : %d / %d / %d" % (min(frames), int(np.median(frames)), max(frames)))
    print("  enc_len     min/med/max  : %d / %d / %d" % (min(eframes), int(np.median(eframes)), max(eframes)))
    print()
    print("=== the frame-mapping question (CTC frame <-> encoder frame) ===")
    ok = sum(1 for r in recs if r["ctc_frames"] == r["enc_len"])
    print("  ctc_frames == enc_len for %d / %d utterances" % (ok, len(recs)))
    for r in recs[:4]:
        print("    id=%-24s ctc=%-4d enc_len=%-4d enc_frames=%-4d bins=%-5d"
              % (r["id"], r["ctc_frames"], r["enc_len"], r["enc_frames"], r["input_bins"]))
    print()
    print("=== hidden-state statistics (is the int8 clamp lossy?) ===")
    print("  per-frame L2 norm  mean %.3f  min %.3f  max %.3f"
          % (float(np.mean([r["norm_mean"] for r in recs])),
             float(np.min([r["norm_min"] for r in recs])),
             float(np.max([r["norm_max"] for r in recs]))))
    print("  fraction |value| > 1.0 (what quantize_hidden_states clamps away): %.4f"
          % float(np.mean([r["frac_abs_gt_1"] for r in recs])))
    print()
    est_frames = float(np.median(frames))
    print("=== export sizing for the AISHELL-train set (17,301 rows) ===")
    print("  median frames/utt (in order to project only aligned frames) : %d" % est_frames)
    print("  fp16 full sequence  : %.3f GB" % (17301 * est_frames * D * 2 / 1e9))
    print("  fp16 only CTC-emission frames (~len(text)+1) : %.1f MB"
          % (17301 * 20 * D * 2 / 1e6))
    print("  measured encoder+ctc cost : %.1f min for 17,301 rows"
          % (17301 * (elapsed / len(recs)) / 60))

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"records": recs, "enc_dim": D,
             "seconds_per_utt": elapsed / len(recs),
             "ctc_equals_enc": ok == len(recs)}, indent=2), encoding="utf-8")
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
