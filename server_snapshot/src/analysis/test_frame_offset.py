#!/usr/bin/env python
"""Prove the CTC frame <-> encoder frame correspondence, not merely assert it.

The whole coupling rests on one claim: CTC frame f is the encoder output at index
f + 4, because the encoder is fed four query tokens (textnorm, language, event,
emotion). Everything downstream indexes acoustics with that offset -- the per-character
export, the constrained-decoding sets, and the RESULTS section 17 measurements. So far
it had only been checked by index ARITHMETIC (shapes differ by exactly 4, on all
17,301 utterances), which would also hold if the offset were wrong but the alignment
were shifted relative to the CTC head.

The decisive test is functional: take the encoder state we would use for frame f, push
it back through the CTC head, and check it reproduces the CTC distribution at f. If
the offset were off by even one, the two distributions would disagree.

Runs on CPU -- SenseVoiceSmall is small, so this can execute while training owns the GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

WS = Path("/root/autodl-tmp")
sys.path.insert(0, str(WS / ".dsh_checks"))
sys.path.insert(0, str(WS / "src" / "analysis"))
from score_sensevoice_tokstats import audio_map, encode_audio  # noqa: E402
from export_acoustic_vectors import QUERY_TOKENS, encoder_states  # noqa: E402


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(WS / ".dsh_checks/rerank/stcmds_manifest.jsonl"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    a = ap.parse_args()

    from funasr import AutoModel
    wrapper = AutoModel(model=a.model, device="cpu", disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = torch.device("cpu")
    ctc_lo = model.ctc.ctc_lo

    au = audio_map(a.manifest)
    keys = list(au)[: a.n]

    worst = 0.0
    checked = 0
    # argmax agreement per candidate offset: THE discriminative statistic. A raw max
    # deviation is dominated by a few outlier frames and barely separates the correct
    # offset from the wrong ones, whereas the argmax either matches at ~every frame
    # (right offset) or at almost none (wrong offset).
    agree_by_off = {}
    for rid in keys:
        wav = au[rid]
        lpn = encode_audio(model, tokenizer, frontend, wav, device, False).detach().float().numpy()
        enc = encoder_states(model, frontend, wav, device, "auto", "withitn")

        print("id=%s  ctc_frames=%d  enc_rows_after_strip=%d" % (rid, lpn.shape[0], enc.shape[0]))
        assert enc.shape[0] == lpn.shape[0], "frame counts differ; offset assumption broken"

        ref = lpn.argmax(1)
        checked += lpn.shape[0]
        for off in range(-2, 8):
            idx = np.arange(lpn.shape[0]) + (off - QUERY_TOKENS)
            ok = (idx >= 0) & (idx < enc.shape[0])
            if ok.sum() == 0:
                continue
            with torch.no_grad():
                w = ctc_lo(torch.from_numpy(enc[idx[ok]]).float())
                pred = w.argmax(-1).numpy()
            a, b = agree_by_off.get(off, (0, 0))
            agree_by_off[off] = (a + int((pred == ref[ok]).sum()), b + int(ok.sum()))

        # absolute deviation at the assumed offset, for the record
        with torch.no_grad():
            repro = torch.log_softmax(ctc_lo(torch.from_numpy(enc).float()), dim=-1).numpy()
        worst = max(worst, float(np.abs(repro - lpn).max()))

    print()
    print("=== argmax agreement per candidate offset (f + QUERY_TOKENS + delta) ===")
    for off in sorted(agree_by_off):
        a, b = agree_by_off[off]
        mark = "   <-- assumed" if off == QUERY_TOKENS else ""
        print("  offset f+%-3d  %6d/%-6d = %6.2f%%%s" % (off, a, b, 100.0 * a / b, mark))

    a, b = agree_by_off.get(QUERY_TOKENS, (0, 1))
    rate = 100.0 * a / b
    others = [100.0 * x / y for o, (x, y) in agree_by_off.items() if o != QUERY_TOKENS]
    best_other = max(others) if others else 0.0

    print()
    print("frames checked: %d" % checked)
    print("offset f+%d agreement      : %.2f%%" % (QUERY_TOKENS, rate))
    print("best WRONG-offset agreement: %.2f%%" % best_other)
    print("max |reproduced - recorded| at the assumed offset: %.4f" % worst)
    print("(the absolute deviation is NOT ~0 because the two code paths condition the")
    print(" encoder differently -- encode_audio uses language='zh' and textnorm='woitn',")
    print(" whereas the exporter uses language='auto' and textnorm='withitn'. Only the")
    print(" argmax comparison is offset-sensitive; the deviation is conditioning.)")
    ok = rate > 90.0 and rate > best_other + 20.0
    print()
    print("VERDICT: %s" % ("PASS -- enc[f+4] IS the encoder state behind CTC frame f"
                           if ok else "FAIL -- the offset assumption is wrong"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
