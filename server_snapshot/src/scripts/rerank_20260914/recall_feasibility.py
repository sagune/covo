#!/usr/bin/env python3
"""Is there a CER/recall conflict inside the pool, or is it a labelling artefact?

For a pool this reports:
  * oracle CER unrestricted                      min over the pool
  * oracle CER with hlost == 0                   best CER among candidates that keep
                                                 every designated hotword the reference has
  * designated-hotword recall of both oracles
  * how often the pool minimum-CER candidate loses a designated hotword
  * how often NO candidate keeps all designated hotwords (label infeasible)
If the restricted oracle is close to the unrestricted one, a good ranker can have
both axes and the earlier recall loss was a labelling/optimisation artefact.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--uttid", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    pos2utt = [l.split()[0] for l in Path(a.uttid).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = defaultdict(list)
    for line in Path(a.aligned).read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig[f[1].strip()].append(norm(f[0]))

    chars = 0
    e_orc = e_res = 0
    men = h_orc = h_res = 0
    conf = infeasible = 0
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        idx = int(r["id"])
        uid = pos2utt[idx] if idx < len(pos2utt) else None
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        pool = [norm(t) for t in (inp.get("nbest") or [])]
        pool += [norm(c["text"]) for c in ((inp.get("cbwhisper") or {}).get("candidates") or [])
                 if isinstance(c, dict) and c.get("text")]
        pool = list(dict.fromkeys(pool))
        if not ref or not pool or not uid:
            continue
        keys = desig.get(uid, [])
        chars += len(ref)
        d = [edit_distance(list(ref), list(t)) for t in pool]
        hl = [sum(1 for k in keys if k in ref and k not in t) for t in pool]
        j0 = min(range(len(pool)), key=lambda j: d[j])
        ok = [j for j in range(len(pool)) if hl[j] == 0]
        j1 = min(ok, key=lambda j: d[j]) if ok else j0
        if not ok:
            infeasible += 1
        if hl[j0] > 0:
            conf += 1
        e_orc += d[j0]
        e_res += d[j1]
        for k in keys:
            men += 1
            h_orc += int(k in pool[j0])
            h_res += int(k in pool[j1])

    print("# %s   chars=%d mentions=%d" % (a.label, chars, men))
    print("  pool oracle, unrestricted        CER %.4f%%   recall %.2f%%" % (100 * e_orc / chars, 100 * h_orc / men))
    print("  pool oracle, hlost==0 required   CER %.4f%%   recall %.2f%%" % (100 * e_res / chars, 100 * h_res / men))
    print("  CER-optimal candidate loses a designated hotword in %d rows (%.1f%%); label infeasible in %d rows (%.1f%%)"
          % (conf, 100 * conf / max(1, len(desig)), infeasible, 100 * infeasible / max(1, len(desig))))


if __name__ == "__main__":
    main()
