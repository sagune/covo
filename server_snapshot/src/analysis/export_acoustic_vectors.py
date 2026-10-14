#!/usr/bin/env python
"""Export per-character acoustic vectors for the AC-COVO coupling component.

For every character of the ASR top-1 hypothesis we store the FROZEN SenseVoice
encoder state at the exact frame where the CTC head emitted that character
(Viterbi forced alignment).  That is the vector the projector will turn into an
additive bias on the corresponding character's input embedding, so the language
model sees, per character, the acoustic evidence that produced it.

Two facts from the checkpoint-0 probe drive the implementation:

  * CTC frame f corresponds to encoder frame f + 4 (the encoder is fed four
    query tokens -- textnorm, language, event, emotion -- so stripping them
    gives a 1:1 frame correspondence and no resampling is needed).
  * `hs_utils.quantize_hidden_states` clamps to [-1, 1] and int8-quantises, but
    the per-frame L2 norm is ~34 and 38.5% of values exceed 1.0, so that format
    would destroy most of the representation.  We store raw fp16 instead.

Alignment is done on the NORMALISED top-1 (punctuation is unemittable in
SenseVoice CTC and would break Viterbi), then mapped back onto the RAW top-1
characters through an explicit index map, because 22.9% of AISHELL-train rows
have raw top-1 != normalise(top-1) and the prompt renders the RAW string.
Characters that do not survive normalisation get a zero vector, so one output
row is always `len(raw_top1)` long and lines up with the hypothesis span.

Output layout (single memmap-able binary + jsonl index):
  <out-bin>    concatenated float16 arrays, each (n_chars, 512)
  <out-index>  {"id", "offset", "n", "top1", "reference", "n_unemittable"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/autodl-tmp")
sys.path.insert(0, str(ROOT / ".dsh_checks"))
from score_sensevoice_tokstats import (  # noqa: E402
    audio_map, encode_audio, normalize_text, viterbi_token_logps,
)

QUERY_TOKENS = 4          # textnorm + language + event + emotion prepended by the encoder


def encoder_states(model, frontend, wav, device, language, textnorm):
    """Encoder output with the 4 query-token frames stripped -> (T_ctc, D)."""
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    audio = load_audio_text_image_video(wav, fs=frontend.fs, audio_fs=None,
                                        data_type="sound", tokenizer=None)
    speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)
    speech = speech.to(device)
    speech_lengths = speech_lengths.to(device)
    n = speech.size(0)

    lang_q = model.embed(torch.LongTensor(
        [[model.lid_dict[language] if language in model.lid_dict else 0]]).to(device)).repeat(n, 1, 1)
    tn_q = model.embed(torch.LongTensor([[model.textnorm_dict[textnorm]]]).to(device)).repeat(n, 1, 1)
    x = torch.cat((tn_q, speech), dim=1)
    lengths = speech_lengths + 1
    ee_q = model.embed(torch.LongTensor([[1, 2]]).to(device)).repeat(n, 1, 1)
    x = torch.cat((lang_q, ee_q, x), dim=1)
    lengths = lengths + 3

    with torch.no_grad():
        enc, _ = model.encoder(x, lengths)
    if isinstance(enc, tuple):
        enc = enc[0]
    return enc[0, QUERY_TOKENS:].detach().float().cpu().numpy()


def raw_to_norm_map(raw: str):
    """For each raw character, the index of the normalised character it produced.

    `normalize_text` is a filter/converter, so normalise(prefix) is monotone in
    the prefix length; the increment tells us how many normalised characters this
    raw character contributed.  -1 means the character did not survive.
    """
    out = []
    prev = 0
    for i in range(len(raw)):
        cur = len(normalize_text(raw[: i + 1]))
        out.append(prev if cur > prev else -1)
        prev = cur
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="evidence jsonl (id, reference, input.asr_top1)")
    ap.add_argument("--manifest", required=True, help="jsonl with id -> wav")
    ap.add_argument("--out-bin", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--textnorm", default="withitn")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument(
        "--uttid-file",
        default="",
        help="When the input's `id` is a POSITIONAL index (AISHELL dev, THCHS-30 and "
             "the ST-CMDS eval messages all use 0,1,2,... rather than uttids), this "
             "file supplies uttid[id] so the row can be resolved to audio.",
    )
    a = ap.parse_args()

    from funasr import AutoModel

    wrapper = AutoModel(model=a.model, device="cuda:0", disable_update=True)
    model = wrapper.model.eval()
    tokenizer = wrapper.kwargs["tokenizer"]
    frontend = wrapper.kwargs["frontend"]
    device = next(model.parameters()).device
    vocab = int(model.ctc.ctc_lo.out_features)
    blank_id = int(model.blank_id)
    excluded = set(range(1, 16)) | set(range(25000, vocab))

    au = audio_map(a.manifest)
    man = [json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = {m["key"]: m for m in man if m.get("key")}
    by_id = {str(m["id"]): m for m in man}

    def resolve(row):
        """-> (uttid, wav) or (None, None).

        AISHELL-train evidence rows carry SHARD-LOCAL ids, so the bare id is
        ambiguous; the manifest's `key` ("split|row") is the exact handle.  The
        eval files instead carry a POSITIONAL index, resolved through
        `--uttid-file`.  Plain uttid ids (ST-CMDS/THCHS token inputs) resolve
        directly.
        """
        rid = str(row.get("id", ""))
        split = row.get("split")
        if split is not None and ("%s|%s" % (split, rid)) in by_key:
            m = by_key["%s|%s" % (split, rid)]
            return str(m["id"]), m["wav"]
        if pos2utt is not None and rid.lstrip("-").isdigit():
            idx = int(rid)
            if 0 <= idx < len(pos2utt):
                u = pos2utt[idx]
                if u in by_id:
                    return str(by_id[u]["id"]), by_id[u]["wav"]
                if u in au:
                    return u, au[u]
        if rid in by_id:
            return str(by_id[rid]["id"]), by_id[rid]["wav"]
        if rid in au:
            return rid, au[rid]
        return None, None

    pos2utt = None
    if a.uttid_file:
        pos2utt = [l.split()[0] for l in Path(a.uttid_file).read_text(
            encoding="utf-8-sig").splitlines() if l.strip()]
        print("uttid file: %d entries" % len(pos2utt), flush=True)

    rows = [json.loads(l) for l in Path(a.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    resolved = []
    for r in rows:
        rid, wav = resolve(r)
        if rid is not None:
            resolved.append((r, rid, wav))
    rows = resolved
    if a.limit:
        rows = rows[: a.limit]
    print("rows to export: %d   (audio map entries: %d)" % (len(rows), len(au)), flush=True)

    bin_path = Path(a.out_bin)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path = Path(a.out_index)
    offset = 0
    n_zero = 0
    t0 = time.monotonic()
    stats = {"chars": 0, "no_frame": 0, "frame_mismatch": 0}

    with bin_path.open("wb") as wb, idx_path.open("w", encoding="utf-8") as wi:
        for n, (row, rid, wav) in enumerate(rows, 1):
            top1 = str(((row.get("input") or {}).get("asr_top1")) or "")
            if not top1:
                continue

            lp = encode_audio(model, tokenizer, frontend, wav, device, False)
            lpn = lp.detach().float().cpu().numpy()
            n_ctc = lpn.shape[0]
            enc = encoder_states(model, frontend, wav, device, a.language, a.textnorm)
            if enc.shape[0] != n_ctc:
                stats["frame_mismatch"] += 1

            norm = normalize_text(top1)
            ids = [int(t) for t in tokenizer.encode(norm)
                   if int(t) not in excluded and int(t) != blank_id]
            if ids:
                _, frames_norm = viterbi_token_logps(lpn, ids, blank_id, with_frames=True)
            else:
                frames_norm = []

            r2n = raw_to_norm_map(top1)
            vec = np.zeros((len(top1), enc.shape[1]), dtype=np.float16)
            n_no = 0
            for i, j in enumerate(r2n):
                if j < 0 or j >= len(frames_norm):
                    n_no += 1
                    continue
                f = frames_norm[j]
                if f < 0 or f >= enc.shape[0]:
                    n_no += 1
                    continue
                vec[i] = enc[f].astype(np.float16)
            stats["chars"] += len(top1)
            stats["no_frame"] += n_no
            n_zero += 1 if n_no else 0

            wb.write(vec.tobytes())
            wi.write(json.dumps({"id": rid, "offset": offset, "n": int(vec.shape[0]),
                                 "dim": int(vec.shape[1]), "top1": top1,
                                 "reference": row.get("reference"),
                                 "n_no_frame": n_no}, ensure_ascii=False) + "\n")
            offset += vec.nbytes

            if a.progress_every and n % a.progress_every == 0:
                el = time.monotonic() - t0
                print(json.dumps({"exported": n, "elapsed_s": round(el, 1),
                                  "eta_min": round(el / n * (len(rows) - n) / 60, 1),
                                  "gb": round(offset / 1e9, 3)}), flush=True)
            if n % 500 == 0:
                wi.flush()
                wb.flush()

    print("done -> %s (%.3f GB) + %s" % (bin_path, offset / 1e9, idx_path), flush=True)
    print(json.dumps({"rows": len(rows), "chars": stats["chars"],
                      "chars_without_frame": stats["no_frame"],
                      "rows_with_any_missing": n_zero,
                      "enc_ctc_frame_mismatch": stats["frame_mismatch"],
                      "bytes": offset}), flush=True)


if __name__ == "__main__":
    main()
