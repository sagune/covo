#!/usr/bin/env python
"""Build an exact (split, id) -> uttid -> wav manifest for the AISHELL-train evidence.

The evidence file uses SHARD-LOCAL ids (`id` is the row index inside a 1000-row
shard, `split` names the shard), so `id` alone is ambiguous -- a known trap in
this project.  This resolves each row through the shard's own `uttid` file and
also reports how often the raw ASR top-1 differs from its normalised form, which
decides whether acoustic frames can be attached to the prompt's characters
directly or need a raw->normalised index map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp")
HOT = ROOT / "datasets/aishell/data_aishell_sensevoice/hotword"
WAV = ROOT / "datasets/aishell/data_aishell/wav/train"
EVID = ROOT / ".dsh_checks/rerank/train_aishell_evidence.jsonl"
OUT = ROOT / ".dsh_checks/rerank/aishell_train_manifest.jsonl"

sys.path.insert(0, str(ROOT / ".dsh_checks"))
from score_sensevoice_tokstats import normalize_text  # noqa: E402


def main() -> None:
    rows = [json.loads(l) for l in EVID.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("evidence rows: %d" % len(rows))

    shards = {}
    for split in sorted({r["split"] for r in rows}):
        d = HOT / split
        uttids = [l.strip() for l in (d / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        texts = {}
        for line in (d / "text").read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                parts = line.split(maxsplit=1)
                texts[parts[0]] = parts[1] if len(parts) > 1 else ""
        shards[split] = (uttids, texts)
        print("  %-20s %d uttids" % (split, len(uttids)))

    out, missing, mismatch = [], 0, 0
    norm_diff = 0
    for r in rows:
        split, rid = r["split"], int(r["id"])
        uttids, texts = shards[split]
        if rid >= len(uttids):
            missing += 1
            continue
        uttid = uttids[rid]
        text = texts.get(uttid, "")
        if text and text != r.get("reference"):
            mismatch += 1
        wav = WAV / uttid[6:11] / ("%s.wav" % uttid)
        if not wav.exists():
            missing += 1
            continue
        top1 = str(((r.get("input") or {}).get("asr_top1")) or "")
        if top1 and normalize_text(top1) != top1:
            norm_diff += 1
        out.append({"key": "%s|%d" % (split, rid), "id": uttid, "wav": str(wav),
                    "reference": r.get("reference"), "split": split, "row": rid})

    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    print()
    print("manifest rows          : %d" % len(out))
    print("missing uttid or wav   : %d" % missing)
    print("reference != shard text: %d" % mismatch)
    print("raw top-1 != normalise(top-1) : %d  (%.2f%%)"
          % (norm_diff, 100.0 * norm_diff / max(len(out), 1)))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
