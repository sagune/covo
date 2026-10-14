#!/usr/bin/env python
"""Convert a uttid-keyed manifest into a positional one.

Why this is needed: `score_sensevoice_tokstats.py` (which produces the per-character
acoustic top-k) resolves audio by matching the input row's `id` against the
manifest's `id`. The AISHELL-dev and THCHS-30 pools use POSITIONAL ids (0,1,2,...)
while their manifests are keyed by uttid, so the scorer matches nothing and silently
exports zero rows.

The positional mapping is the project-wide convention: row i of the pool corresponds
to line i of that corpus's `uttid` file. This script asserts that convention by
checking the references agree, rather than trusting it.

    python make_positional_manifest.py \
      --manifest .dsh_checks/rerank/thchs30_manifest.jsonl \
      --uttid   datasets/thchs30/.../test/uttid \
      --pool    .dsh_checks/rerank/thchs_learned_v2.jsonl \
      --out     .dsh_checks/rerank/thchs30_positional_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--pool", default="", help="optional: verify references line up")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    man = {}
    for line in Path(a.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            man[str(r["id"])] = r
    uttids = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines()
              if l.strip()]

    rows, missing = [], 0
    for i, u in enumerate(uttids):
        m = man.get(u)
        if m is None:
            missing += 1
            continue
        rows.append({"id": str(i), "wav": m["wav"], "reference": m.get("reference", ""),
                     "uttid": u})

    print(json.dumps({"manifest_entries": len(man), "uttids": len(uttids),
                      "written": len(rows), "missing_wav": missing}))

    if a.pool:
        pool = [json.loads(l) for l in Path(a.pool).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        n = min(len(pool), len(rows))
        agree = sum(1 for k in range(n)
                    if str(pool[k].get("reference") or "") == str(rows[k].get("reference") or ""))
        print(json.dumps({"pool_rows": len(pool), "compared": n,
                          "reference_agreement": agree,
                          "rate": round(agree / max(n, 1), 4)}))
        if n and agree / n < 0.99:
            raise SystemExit("REFUSING: the pool order does not line up with the uttid "
                             "file, so a positional manifest would attach the wrong audio.")

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
