#!/usr/bin/env python3
"""Why does the rerank rule hurt ST-CMDS? Compare key orders within the preserving subset."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
W = Path("/root/autodl-tmp")

JOBS = [
    ("ST-CMDS", W / ".dsh_checks/stcmds/stcmds_evidence.jsonl",
     R / "stcmds_ctc_scores.jsonl",
     W / "datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid",
     W / "datasets/stcmds/cb_sensevoice_heldout/hotword/test/aligned.txt"),
    ("THCHS-30", W / "cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl",
     R / "thchs30_ctc_scores.jsonl",
     W / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid",
     W / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/aligned.txt"),
]


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def num(c, k, d=0.0):
    try:
        return float(c.get(k, d) or d)
    except (TypeError, ValueError):
        return d


for label, ev_path, ctc_path, uttid_path, align_path in JOBS:
    if not ctc_path.exists():
        print("%s: missing %s" % (label, ctc_path))
        continue
    pos2utt = [l.split()[0] for l in uttid_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    desig = {}
    for line in align_path.read_text(encoding="utf-8-sig").splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            desig.setdefault(f[1].strip(), []).append(norm(f[0]))
    ctc = {}
    for l in ctc_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
        ctc[str(r["id"])] = {norm(c["text"]): num(c, "asr_score", -1e9) for c in cs if c.get("text")}

    prep = []
    for l in ev_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        uid = pos2utt[int(r["id"])]
        inp = r.get("input") or {}
        old = norm(inp.get("asr_top1") or "")
        sc = ctc.get(uid) or {}
        pool = []
        for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
            if not isinstance(c, dict) or not c.get("text"):
                continue
            t = norm(c["text"])
            if t not in sc:
                continue
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), exact_raw=num(c, "exact_score"),
                             ctc=sc[t], search=num(c, "search_score"), rank=int(num(c, "rank", 99))))
        if not pool:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        ref = norm(r.get("reference") or "")
        prep.append(dict(ref=ref, old=old, sub=sub, desig=[norm(k) for k in desig.get(uid, [])]))

    TC = sum(len(p["ref"]) for p in prep)
    MEN = sum(len(p["desig"]) for p in prep)
    print("# %s  rows=%d  mention=%d  mean subset=%.2f" % (label, len(prep), MEN,
          sum(len(p["sub"]) for p in prep) / len(prep)))

    def ev(fn, name):
        err = hits = 0
        for p in prep:
            t = fn(p)["text"]
            err += edit_distance(list(p["ref"]), list(t))
            for k in p["desig"]:
                hits += k in t
        print("   %-34s CER %7.4f%% | recall %6.2f%% (%d)" % (name, 100 * err / TC, 100 * hits / MEN, hits))

    ev(lambda p: dict(text=p["old"]), "shipped reranker (reference)")
    ev(lambda p: max(p["sub"], key=lambda c: (c["exact"], c["ctc"])), "(exact, ctc)  [used]")
    ev(lambda p: max(p["sub"], key=lambda c: (c["ctc"], c["exact"])), "(ctc, exact)")
    ev(lambda p: max(p["sub"], key=lambda c: c["ctc"]), "ctc only")
    ev(lambda p: max(p["sub"], key=lambda c: c["exact"]), "exact only")
    ev(lambda p: max(p["sub"], key=lambda c: c["search"]), "search_score only")
    ev(lambda p: max(p["sub"], key=lambda c: -c["rank"]), "shipped rank (rank=1 preferred)")
    ev(lambda p: min(p["sub"], key=lambda c: edit_distance(list(p["ref"]), list(c["text"]))), "ORACLE in subset")
    print()
