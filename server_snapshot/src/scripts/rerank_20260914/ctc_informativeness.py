#!/usr/bin/env python3
"""Is forced-CTC informative about correctness, per dataset?

For each row, inside the preserving subset: does the forced-CTC score rank the
best-CER candidate first? If not, no acoustic-only rerank can work there.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
W = Path("/root/autodl-tmp")
JOBS = [
    ("AISHELL dev", W / "src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl", R / "ctc_scores.jsonl",
     W / "datasets/aishell/data_aishell_sensevoice/hotword/dev/uttid"),
    ("THCHS-30", W / "cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl",
     R / "thchs30_ctc_scores.jsonl", W / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/uttid"),
    ("ST-CMDS", W / ".dsh_checks/stcmds/stcmds_evidence.jsonl", R / "stcmds_ctc_scores.jsonl",
     W / "datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid"),
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


for label, ev_path, ctc_path, uttid_path in JOBS:
    if not ctc_path.exists():
        print("%s: missing scores" % label)
        continue
    pos2utt = [l.split()[0] for l in uttid_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    ctc = {}
    for l in ctc_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        cs = ((r.get("input") or {}).get("cbwhisper") or {}).get("candidates") or []
        ctc[str(r["id"])] = {norm(c["text"]): num(c, "asr_score", -1e9) for c in cs if c.get("text")}

    hit_top = hit_thirds = n = 0
    pct_sum = 0.0
    delta_sum = 0.0
    for l in ev_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        uid = pos2utt[int(r["id"])]
        inp = r.get("input") or {}
        old = norm(inp.get("asr_top1") or "")
        sc = ctc.get(uid) or {}
        pool = [(norm(c["text"]), sc[norm(c["text"])]) for c in ((inp.get("cbwhisper") or {}).get("candidates") or [])
                if isinstance(c, dict) and c.get("text") and norm(c["text"]) in sc]
        if len(pool) < 2:
            continue
        protect = [t for t in texts_of(inp.get("prompt_hotwords")) or texts_of(inp.get("hotwords")) if t in old]
        sub = [(t, s) for t, s in pool if all(p in t for p in protect)] or pool
        ref = norm(r.get("reference") or "")
        if not ref:
            continue
        d = [(edit_distance(list(ref), list(t)), t, s) for t, s in sub]
        d.sort(key=lambda x: x[0])
        best_err = d[0][0]
        # where does the forced-CTC argmax land?
        arg = max(d, key=lambda x: x[2])
        n += 1
        hit_top += arg[0] == best_err
        rank = sorted(d, key=lambda x: -x[2]).index(arg)
        pct = 100.0 * (1 - rank / max(len(d) - 1, 1))
        pct_sum += pct
        orc = [x for x in d if x[0] == best_err]
        delta_sum += sum(x[2] for x in orc) / len(orc) - sum(x[2] for x in d) / len(d)

    print("%-14s rows=%5d | forced-CTC argmax IS best-CER: %5.1f%% | oracle mean ctc minus mean ctc: %+8.4f" % (
        label, n, 100 * hit_top / n, delta_sum / n))
