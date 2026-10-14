#!/usr/bin/env python
"""Data-integrity check on the exported acoustic .bin itself.

Every stage around this file is verified -- the frame offset is proven, the span is
located correctly, the pairing is checked -- but the ARTEFACT has never been read back
and compared with a fresh computation. A wrong offset, a wrong stride, or an index/byte
unit confusion in `AcousticStore.vectors` would all produce vectors that are plausible
in shape and magnitude while being attached to the wrong characters.

A first version of this test asserted `stored[p] == enc[frame[p]]` exactly and failed
0/49. That was the TEST being wrong, and the diagnosis is worth recording: the encoder
forward is NOT deterministic -- two identical calls on the same audio and conditioning
differ by up to 1.18 on values of order 20 (the ASR frontend dithers its features).
Exact equality was therefore unachievable even for a perfectly correct artefact.

The robust and actually stronger check is a NEAREST-FRAME test: for each character, the
stored vector must be closer to the expected emission frame than to any other frame in
the utterance. That is insensitive to the dither while still failing loudly if the
offset, the stride or the indexing were wrong.

CPU only, so it runs while training owns the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

WS = Path("/root/autodl-tmp")
sys.path.insert(0, str(WS / ".dsh_checks"))
sys.path.insert(0, str(WS / "src" / "analysis"))
sys.path.insert(0, str(WS / "cbwhisper_covo_migration_20260609_tar_extracted/covo/src"))
from export_acoustic_vectors import QUERY_TOKENS, encoder_states, raw_to_norm_map  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(WS / ".dsh_checks/rerank/ac_aishell_train.bin"))
    ap.add_argument("--index", default=str(WS / ".dsh_checks/rerank/ac_aishell_train.jsonl"))
    ap.add_argument("--manifest", default=str(WS / ".dsh_checks/rerank/aishell_train_manifest.jsonl"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    a = ap.parse_args()

    from score_sensevoice_tokstats import (audio_map, encode_audio, normalize_text,
                                           viterbi_token_logps)
    from acoustic_conditioning import AcousticStore

    store = AcousticStore(a.bin, a.index)
    man = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    audio = audio_map(a.manifest)

    from funasr import AutoModel
    wrapper = AutoModel(model=a.model, device="cpu", disable_update=True)
    model = wrapper.model.eval()
    tok = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = torch.device("cpu")
    print("frontend dither=%s  (non-zero makes the forward random)"
          % getattr(frontend, "dither", "n/a"))
    vocab = int(model.ctc.ctc_lo.out_features)
    blank = int(model.blank_id)
    excl = set(range(1, 16)) | set(range(25000, vocab))

    n_checked = n_ok = 0
    margins = []
    for i in range(min(a.n, len(store))):
        rec = store.row(i)
        uttid = rec["id"]
        if uttid not in audio:
            print("row %d: no audio for %s" % (i, uttid))
            continue
        top1 = rec["top1"]
        stored = store.vectors(i)

        enc = encoder_states(model, frontend, audio[uttid], device, "auto", "withitn")
        lpn = encode_audio(model, tok, frontend, audio[uttid], device, False).detach().float().numpy()
        ids = [int(t) for t in tok.encode(normalize_text(top1))
               if int(t) not in excl and int(t) != blank]
        _, frames = viterbi_token_logps(lpn, ids, blank, with_frames=True)
        r2n = raw_to_norm_map(top1)

        ok = bad = 0
        for p, j in enumerate(r2n):
            if j < 0 or j >= len(frames):
                continue
            f = frames[j]
            if f < 0 or f >= enc.shape[0]:
                continue
            d = np.abs(enc - stored[p]).max(axis=1)      # distance to every frame
            best = int(np.argmin(d))
            order = np.sort(d)
            margins.append(float(order[1] - order[0]))
            n_checked += 1
            if best == f:
                ok += 1
                n_ok += 1
            else:
                bad += 1
        print("row %-3d uttid=%-18s chars=%-3d nearest-frame correct=%d wrong=%d"
              % (i, uttid, len(top1), ok, bad))

    print()
    print("characters checked          : %d" % n_checked)
    print("stored vector nearest to the EXPECTED frame : %d (%.2f%%)"
          % (n_ok, 100.0 * n_ok / max(n_checked, 1)))
    if margins:
        print("margin over the next-best frame: median %.4f" % float(np.median(margins)))
    ok = n_checked > 0 and n_ok == n_checked
    print()
    print("VERDICT: %s" % ("PASS -- every stored vector sits closest to its own emission "
                           "frame, so the offset, stride and indexing are all correct"
                           if ok else "FAIL -- some stored vectors belong to another frame"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
