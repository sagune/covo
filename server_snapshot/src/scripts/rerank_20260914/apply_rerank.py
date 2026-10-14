#!/usr/bin/env python3
"""P3': apply the new ranking rule to the AISHELL dev evidence.

Rule (reference-blind):
  protect = prompt_hotwords/hotwords present in the current asr_top1
  subset  = candidates containing every protected term (fallback: whole pool)
  top-1   = argmax (exact_weighted_score, forced-CTC log-likelihood)

Writes a reranked evidence file (asr_top1 + nbest reordered, cbwhisper untouched)
and verifies the new front-end CER / designated recall before any GPU run.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

SRC = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908")
R = Path("/root/autodl-tmp/.dsh_checks/rerank")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")
OUT = R / "dev_reranked.jsonl"

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))

ctc = {}
for l in (R / "ctc_scores.jsonl").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
    ctc[str(r["id"])] = {norm(c["text"]): float(c.get("asr_score") or -1e9) for c in cs if c.get("text")}


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


rows = [json.loads(l) for l in (SRC / "dev.evidence.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
out_rows = []
changed = 0
chars = men = 0
err_before = err_after = 0
hits_before = hits_after = 0
no_ctc = 0

for r in rows:
    uid = pos2utt[int(r["id"])]
    inp = r["input"]
    old_top = norm(inp.get("asr_top1") or "")
    scores = ctc.get(uid) or {}
    pool = []
    for c in (inp.get("cbwhisper") or {}).get("candidates") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        t = norm(c["text"])
        if t not in scores:
            continue
        pool.append(dict(text=t, exact=num(c, "exact_weighted_score"), ctc=scores[t]))
    if not pool:
        no_ctc += 1
        out_rows.append(r)
        continue

    protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old_top]
    sub = [c for c in pool if all(p in c["text"] for p in protect)] or pool
    best = max(sub, key=lambda c: (c["exact"], c["ctc"]))

    ref = norm(r.get("reference") or "")
    if ref:
        chars += len(ref)
        err_before += edit_distance(list(ref), list(old_top))
        err_after += edit_distance(list(ref), list(best["text"]))
        for k in desig.get(uid, []):
            men += 1
            hits_before += k in old_top
            hits_after += k in best["text"]

    if best["text"] != old_top:
        changed += 1
    new = json.loads(json.dumps(r, ensure_ascii=False))
    new["input"]["asr_top1"] = best["text"]
    nb = [best["text"]] + [t for t in (inp.get("nbest") or []) if norm(t) != best["text"]]
    new["input"]["nbest"] = nb
    out_rows.append(new)

OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows), encoding="utf-8")
print("rows: %d | top-1 changed: %d (%.1f%%) | rows without forced-CTC coverage: %d" % (
    len(out_rows), changed, 100 * changed / len(out_rows), no_ctc))
print("front-end CER   : %.4f%% -> %.4f%%  (%+.4f pp)" % (
    100 * err_before / chars, 100 * err_after / chars, 100 * (err_before - err_after) / chars))
print("designated recall: %.2f%% (%d/%d) -> %.2f%% (%d/%d)" % (
    100 * hits_before / men, hits_before, men, 100 * hits_after / men, hits_after, men))
print("wrote %s" % OUT)
