#!/usr/bin/env python
"""Build an id->wav manifest for one AISHELL hotword split (dev/test).

Same convention as the train builder: `wav/<split>/<speaker>/<uttid>.wav` with
speaker = uttid[6:11].  Used so the acoustic exporter can attach encoder states
to the AISHELL-dev checkpoint-selection set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotword-dir", required=True)
    ap.add_argument("--wav-split", required=True, help="subdirectory under wav/, e.g. dev")
    ap.add_argument("--wav-root", default="/root/autodl-tmp/datasets/aishell/data_aishell/wav")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = Path(a.hotword_dir)
    uttids = [l.strip() for l in (d / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    texts = {}
    for line in (d / "text").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            p = line.split(maxsplit=1)
            texts[p[0]] = p[1] if len(p) > 1 else ""

    out, missing = [], 0
    for u in uttids:
        wav = Path(a.wav_root) / a.wav_split / u[6:11] / ("%s.wav" % u)
        if not wav.exists():
            missing += 1
            continue
        out.append({"id": u, "wav": str(wav), "reference": texts.get(u, ""), "split": a.wav_split})

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out),
                           encoding="utf-8")
    print(json.dumps({"uttids": len(uttids), "written": len(out), "missing_wav": missing,
                      "out": a.out}))


if __name__ == "__main__":
    main()
