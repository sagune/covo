#!/usr/bin/env python3
"""P1: replay alternative ranking rules over the logged candidate features.

No model, no training. Uses only the per-candidate fields already recorded in the
CB-SenseVoice evidence, so every rule is auditable and reproducible.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SEL = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

rows = [json.loads(l) for l in (SEL / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print("rows:", len(rows))
print()
print("=== full candidate field set (row 0) ===")
c0 = ((rows[0]["input"].get("cbwhisper") or {}).get("candidates") or [])
print(sorted(c0[0].keys()) if c0 else "(no candidates)")
print()


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


def num(c, key, default=0.0):
    try:
        v = c.get(key, default)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


prep = []
rank_mismatch = 0
for r in rows:
    inp = r.get("input") or {}
    ref = norm(r.get("reference") or "")
    if not ref:
        continue
    uid = pos2utt[int(r["id"])]
    cands = []
    for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        cands.append(dict(text=norm(c["text"]), asr=num(c, "asr_score"), search=num(c, "search_score"),
                          hot=num(c, "hotword_score"), exact=num(c, "exact_weighted_score"),
                          phon=num(c, "phonetic_score"), cons=num(c, "consensus_score"),
                          total=num(c, "total_score"), rank=int(num(c, "rank", 999))))
    if not cands:
        continue
    top = norm(inp.get("asr_top1") or "")
    if top and max(cands, key=lambda c: c["total"])["text"] != top:
        rank_mismatch += 1
    preserve = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in top]
    prep.append(dict(uid=uid, ref=ref, top=top, cands=cands, preserve=preserve,
                     desig=[norm(k) for k in desig.get(uid, [])],
                     ref_in_pool=any(c["text"] == ref for c in cands)))

print("rows usable: %d | rows where argmax(total_score) != asr_top1: %d" % (len(prep), rank_mismatch))
print()


def pick(rule, p):
    c = p["cands"]
    if rule == "current_total":
        return max(c, key=lambda x: x["total"])
    if rule == "acoustic":
        return max(c, key=lambda x: x["asr"])
    if rule == "search_score":
        return max(c, key=lambda x: x["search"])
    if rule == "hotword":
        return max(c, key=lambda x: x["hot"])
    if rule == "exact":
        return max(c, key=lambda x: x["exact"])
    if rule == "consensus":
        return max(c, key=lambda x: x["cons"])
    if rule == "preserve_then_acoustic":
        keep = [x for x in c if all(t in x["text"] for t in p["preserve"])] or c
        return max(keep, key=lambda x: x["asr"])
    if rule == "preserve_then_total":
        keep = [x for x in c if all(t in x["text"] for t in p["preserve"])] or c
        return max(keep, key=lambda x: x["total"])
    if rule == "oracle":
        return min(c, key=lambda x: edit_distance(list(p["ref"]), list(x["text"])))
    if rule == "input_top1":
        return dict(text=p["top"])
    raise ValueError(rule)


RULES = ["input_top1", "current_total", "acoustic", "search_score", "hotword", "exact",
         "consensus", "preserve_then_acoustic", "preserve_then_total", "oracle"]

total_chars = sum(len(p["ref"]) for p in prep)
mentions = sum(len(p["desig"]) for p in prep)
inp = [p for p in prep if p["ref_in_pool"]]
inp_chars = sum(len(p["ref"]) for p in inp)
print("%-24s %9s %9s %16s %14s %9s %9s" % (
    "rule", "CER", "in-pool CER", "designated recall", "destroyed", "preserve%", "top1==in%"))
for rule in RULES:
    err = ierr = hits = destroyed = pres_ok = top_ok = 0
    for p in prep:
        c = pick(rule, p)
        t = c["text"]
        d = edit_distance(list(p["ref"]), list(t))
        err += d
        if p["ref_in_pool"]:
            ierr += d
        for k in p["desig"]:
            hits += k in t
            destroyed += (k in p["top"] and k not in t)
        if p["preserve"]:
            pres_ok += all(x in t for x in p["preserve"])
        top_ok += t == p["top"]
    pres_den = sum(1 for p in prep if p["preserve"])
    print("%-24s %8.4f%% %8.4f%% %8.2f%% (%4d) %13d %8.1f%% %8.1f%%" % (
        rule, 100 * err / total_chars, 100 * ierr / inp_chars,
        100 * hits / mentions, hits, destroyed,
        100 * pres_ok / max(pres_den, 1), 100 * top_ok / len(prep)))

print()
print("in-pool rows: %d (%.1f%% of chars); oracle on those rows is 0%% by construction." % (
    len(inp), 100 * inp_chars / total_chars))
print("pool oracle CER (all rows) = %.4f%%" % (
    100 * sum(edit_distance(list(p["ref"]), list(min(p["cands"], key=lambda x: edit_distance(list(p["ref"]), list(x["text"]))))) for p in prep) / total_chars))
