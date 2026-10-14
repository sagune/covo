#!/usr/bin/env python3
"""Can the rerank switch be restricted so it helps THCHS-30 without hurting ST-CMDS?

Principle: the front-end reranker should only change the shipped top-1 when the
change is about hotwords (it strictly buys hotword coverage, or every changed
span is hotword-local). Arbitrary content changes stay with the shipped choice.
"""
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
W = Path("/root/autodl-tmp")
JOBS = [
    ("ST-CMDS", W / ".dsh_checks/stcmds/stcmds_evidence.jsonl", R / "stcmds_ctc_scores.jsonl",
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


def spans(text, terms):
    z = []
    for t in terms:
        s = 0
        while True:
            p = text.find(t, s)
            if p < 0:
                break
            z.append((p, p + len(t)))
            s = p + 1
    return z


def hotword_local(old, new, protect):
    zo, zn = spans(old, protect), spans(new, protect)
    ops = [o for o in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if o[0] != "equal"]
    for _, i, j, k, l in ops:
        if any(i < z1 and j > z0 for z0, z1 in zo):
            continue
        if any(k < z1 and l > z0 for z0, z1 in zn):
            continue
        return False
    return True


for label, ev_path, ctc_path, uttid_path, align_path in JOBS:
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
            pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=sc[t]))
        if not pool:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
        old_cov = max([c["exact"] for c in pool if c["text"] == old] or [0.0])
        prep.append(dict(uid=uid, ref=norm(r.get("reference") or ""), old=old, sub=sub, protect=protect,
                         old_cov=old_cov, desig=[norm(k) for k in desig.get(uid, [])]))

    TC = sum(len(p["ref"]) for p in prep)
    MEN = sum(len(p["desig"]) for p in prep)
    print("# %s  rows=%d  mention=%d" % (label, len(prep), MEN))

    def ev(name, decide):
        err = hits = sw = 0
        for p in prep:
            prop = max(p["sub"], key=lambda c: (c["exact"], c["ctc"]))
            t = decide(p, prop)
            sw += t != p["old"]
            err += edit_distance(list(p["ref"]), list(t))
            for k in p["desig"]:
                hits += k in t
        print("   %-40s CER %7.4f%% | recall %6.2f%% (%d) | switched %d" % (
            name, 100 * err / TC, 100 * hits / MEN, hits, sw))

    ev("shipped (no switch)", lambda p, prop: p["old"])
    ev("proposal (exact, ctc) always", lambda p, prop: prop["text"])
    ev("switch only if coverage strictly gains",
       lambda p, prop: prop["text"] if prop["exact"] > p["old_cov"] else p["old"])
    ev("switch only if hotword-local", lambda p, prop: prop["text"] if hotword_local(p["old"], prop["text"], p["protect"]) else p["old"])
    ev("switch if hotword-local AND coverage >= old",
       lambda p, prop: prop["text"] if (hotword_local(p["old"], prop["text"], p["protect"]) and prop["exact"] >= p["old_cov"]) else p["old"])
    ev("ORACLE", lambda p, prop: min(p["sub"], key=lambda c: edit_distance(list(p["ref"]), list(c["text"])))["text"])
    print()
