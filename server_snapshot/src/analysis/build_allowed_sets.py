#!/usr/bin/env python
"""Build per-position acoustically-grounded character sets for constrained decoding.

Consumes the output of score_sensevoice_tokstats.py (`tok_top5`, the acoustic
model's own 2nd..6th choices at each character's emission frame) and emits the
`{"id", "top1", "topk"}` form that constrained_decoding.load_allowed_sets expects.

Alignment note that matters: `tok_top5` is indexed by the NORMALISED top-1 (that is
what the Viterbi forced alignment runs on, because punctuation is unemittable in
SenseVoice CTC), whereas both the acoustic vectors and the prompt render the RAW
top-1.  The same `raw_to_norm_map` used by export_acoustic_vectors.py is applied
here, so position p means the same character in the encoder-state channel and in
the constrained-decoding channel.  Getting this wrong would silently apply the
right constraint to the wrong character.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/root/autodl-tmp")
sys.path.insert(0, str(REPO / "src" / "analysis"))
from export_acoustic_vectors import raw_to_norm_map  # noqa: E402

sys.path.insert(0, str(REPO / ".dsh_checks"))
from score_sensevoice_tokstats import normalize_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top5", required=True, help="*_top5.jsonl from score_sensevoice_tokstats.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.top5).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    n_no_alts, n_chars, n_with_alts = 0, 0, 0
    for r in rows:
        inp = r.get("input") or {}
        top1 = str(inp.get("asr_top1") or "")
        cands = ((inp.get("cbwhisper") or {}).get("candidates")) or []
        if not top1 or not cands:
            continue
        c0 = cands[0]
        top5 = c0.get("tok_top5") or []
        norm = c0.get("norm_text") or normalize_text(top1)
        # sanity: tok_top5 must index the normalised text
        if top5 and norm and len(top5) != len(norm):
            n_no_alts += 1
            top5 = []
        r2n = raw_to_norm_map(top1)
        topk = []
        for p, ch in enumerate(top1):
            j = r2n[p]
            alts = []
            if j >= 0 and j < len(top5):
                for c in (top5[j] or [])[: a.top_k]:
                    if c and c != ch and c not in alts:
                        alts.append(c)
            n_chars += 1
            if alts:
                n_with_alts += 1
            topk.append(alts)
        out.append({"id": str(r.get("id")), "top1": top1, "topk": topk})

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out),
                           encoding="utf-8")
    print(json.dumps({
        "rows": len(out),
        "characters": n_chars,
        "characters_with_alternatives": n_with_alts,
        "characters_without_alternatives": n_chars - n_with_alts,
        "rows_dropped_length_mismatch": n_no_alts,
        "out": a.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
