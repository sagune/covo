#!/usr/bin/env python3
"""Re-score an existing candidate pool and export PER-TOKEN acoustic confidence.

Why.  The front end exports `asr_score` = the PER-TOKEN MEAN log-probability of a candidate.
A mean erases the shape we need: "this sentence is fine except two characters where the model
was very unsure" averages to nothing, and that shape is exactly what "the top-1 is wrong"
looks like.  Results 15 found per-CHARACTER agreement reaches precision 46.7% against a 50%
break-even -- close, but short.  This exports the missing acoustic quantity.

What it adds per candidate: n_tokens and the per-token log-probabilities along the Viterbi
forced-align path (CTC lattice), from which min / low-percentile / fraction-below-threshold
are computed downstream.  Tokenisation is 1:1 with Chinese characters (verified), so token k
is character k of the candidate.

Cost: ONE model forward per utterance, shared by all candidates (same as the existing scorer);
the extra work is a Viterbi pass over an already-computed log-prob matrix.

    score_sensevoice_tokstats.py --input <pool.jsonl> --manifest <manifest.jsonl> \
                                 --output <out.jsonl> [--token-stats-topk 3] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path("/root/autodl-tmp")
sys.path.insert(0, str(REPO_ROOT / "src"))
from analysis.generate_sensevoice_chinesehp import encode_audio, normalize_text   # noqa: E402


def ctc_total(log_probs: torch.Tensor, token_ids, blank_id: int) -> float:
    if not token_ids:
        return -1e9
    dev = log_probs.device
    tgt = torch.tensor(token_ids, dtype=torch.long, device=dev)
    loss = F.ctc_loss(log_probs.unsqueeze(1), tgt,
                      torch.tensor([log_probs.size(0)], dtype=torch.long, device=dev),
                      torch.tensor([len(token_ids)], dtype=torch.long, device=dev),
                      blank=int(blank_id), reduction="none", zero_infinity=False)[0]
    s = -float(loss.detach().cpu())
    return s if math.isfinite(s) else -1e9


def viterbi_token_logps(lp: np.ndarray, token_ids, blank_id: int, with_frames: bool = False):
    """Per-token log-probability along the best CTC path (forced alignment).

    lp: [T, V] log-probabilities.  Returns one max-path log-prob per token, in order, so
    token k corresponds to character k of the candidate text.  With with_frames=True also
    returns the frame index at which each token was emitted, which lets the caller read the
    model's own alternative characters at exactly those positions.
    """
    T, _ = lp.shape
    ext = [int(blank_id)]
    for t in token_ids:
        ext += [int(t), int(blank_id)]
    S = len(ext)
    extarr = np.asarray(ext, dtype=np.int64)
    NEG = -1e30
    a = np.full(S, NEG, dtype=np.float64)
    a[0] = lp[0, extarr[0]]
    if S > 1:
        a[1] = lp[0, extarr[1]]
    bp = np.zeros((T, S), dtype=np.int8)
    allow = extarr != int(blank_id)
    if S > 2:
        allow[2:] &= (extarr[2:] != extarr[:-2])
    ar = np.arange(S)
    for t in range(1, T):
        c1 = np.concatenate(([NEG], a[:-1]))
        c2 = np.concatenate(([NEG, NEG], a[:-2])) if S > 2 else np.full(S, NEG)
        c2 = np.where(allow, c2, NEG)
        stack = np.stack([a, c1, c2])
        idx = stack.argmax(0)
        bp[t] = idx.astype(np.int8)
        a = stack[idx, ar] + lp[t, extarr]
    s = int(np.argmax(a))
    path = np.empty(T, dtype=np.int32)
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        # bp stores the OFFSET (0 = stay, 1 = advance, 2 = skip), not the predecessor state.
        # Treating it as a state number collapsed the whole alignment onto the first state
        # (observed: state histogram {0: 77, 30: 1}) and every token score came out as -inf.
        s -= int(bp[t, s])
        path[t - 1] = s
    out, frames = [], []
    for k in range(len(token_ids)):
        st = 2 * k + 1
        mask = path == st
        vals = lp[mask, extarr[st]]
        if vals.size:
            idx = np.flatnonzero(mask)
            j = int(idx[int(np.argmax(vals))])
            out.append(float(vals.max()))
            frames.append(j)
        else:
            out.append(NEG)
            frames.append(-1)
    return (out, frames) if with_frames else out


def read_jsonl(p):
    for line in Path(p).open(encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def audio_map(manifest):
    m = {}
    for r in read_jsonl(manifest):
        m[str(r["id"])] = str(r["wav"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-nbest", type=int, default=10)
    ap.add_argument("--token-stats-topk", type=int, default=3)
    ap.add_argument("--use-itn", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    a = ap.parse_args()

    from funasr import AutoModel
    wrapper = AutoModel(model=a.model, device=a.device, disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = next(model.parameters()).device
    vocab = int(model.ctc.ctc_lo.out_features)
    blank_id = int(model.blank_id)
    excluded = set(range(1, 16)) | set(range(25000, vocab))

    au = audio_map(a.manifest)
    rows = [r for r in read_jsonl(a.input) if str(r.get("id", "")) in au]
    if a.limit:
        rows = rows[: a.limit]
    print("rows to score: %d" % len(rows), flush=True)

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with out.open("w", encoding="utf-8") as w:
        for n, row in enumerate(rows, 1):
            rid = str(row["id"])
            inp = row.get("input") or {}
            nbest = [str(x).strip() for x in (inp.get("nbest") or []) if str(x).strip()]
            if inp.get("asr_top1"):
                top = str(inp["asr_top1"]).strip()
                if top and top not in nbest:
                    nbest = [top] + nbest
            nbest = nbest[: a.max_nbest]
            if not nbest:
                continue
            lp = encode_audio(model, tokenizer, frontend, au[rid], device, a.use_itn)
            lpn = lp.detach().float().cpu().numpy()
            cmin = lpn.min(axis=0)          # per-vocab-id floor, for the "unemittable" count
            cands = []
            for rank, text in enumerate(nbest, 1):
                # Align on the NORMALISED text.  Punctuation is not in the acoustic model's
                # CTC vocabulary (verified: a period scores -inf at every frame and the
                # Viterbi skips that state, leaving -inf token scores), and every metric in
                # this project is computed on normalised text anyway.
                ntext = normalize_text(str(text))
                ids = [int(t) for t in tokenizer.encode(ntext)
                       if int(t) not in excluded and int(t) != blank_id]
                rec = {"rank": rank, "text": str(text), "norm_text": ntext, "n_tokens": len(ids),
                       "asr_score": ctc_total(lp, ids, blank_id)}
                if ids:
                    mx = [float(lpn[:, i].max()) for i in ids]
                    rec["n_unemittable"] = sum(1 for v in mx if v < -15.0)
                if rank <= a.token_stats_topk and ids:
                    tl, tf = viterbi_token_logps(lpn, ids, blank_id, with_frames=True)
                    # floor the -inf entries so downstream statistics stay finite; the count
                    # of them is kept separately as `n_unemittable`, which is itself a strong
                    # "this candidate is wrong" indicator.
                    rec["tok_logps"] = [round(max(x, -20.0), 5) for x in tl]
                    if rank == 1:
                        # the model's OWN alternative at the frame where each character was
                        # emitted: this is the natural pairing with per-character confidence,
                        # and a far better proposal source than some other candidate's text
                        alts, amarg = [], []
                        for k, f in enumerate(tf):
                            if f < 0:
                                alts.append(None)
                                amarg.append(0.0)
                                continue
                            frame_lp = lpn[f]          # NOT `row` -- that name is the record
                            cur = ids[k]
                            order = np.argsort(frame_lp)[::-1]
                            pick = None
                            for j in order[:12]:
                                j = int(j)
                                if j == int(cur) or j == int(blank_id) or j in excluded:
                                    continue
                                pick = j
                                break
                            alts.append(tokenizer.decode([pick]) if pick is not None else None)
                            amarg.append(round(float(frame_lp[cur] - frame_lp[pick]), 4)
                                        if pick is not None else 0.0)
                        rec["tok_alt"] = alts
                        rec["tok_alt_margin"] = amarg
                cands.append(rec)
            w.write(json.dumps({"id": rid, "reference": normalize_text(str(row.get("reference", ""))),
                                "input": {"asr_top1": nbest[0], "nbest": nbest,
                                          "cbwhisper": {"candidates": cands}}},
                               ensure_ascii=False) + "\n")
            if a.progress_every and n % a.progress_every == 0:
                el = time.monotonic() - t0
                print(json.dumps({"scored": n, "elapsed_s": round(el, 1),
                                  "eta_min": round(el / n * (len(rows) - n) / 60, 1)}), flush=True)
    print("done -> %s" % out, flush=True)


if __name__ == "__main__":
    main()
