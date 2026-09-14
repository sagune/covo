#!/usr/bin/env python3
"""Does sampling + self-consistency actually help the rows where the LLM must hard-carry?

Damaged rows: greedy destroyed a designated hotword the front-end had.
Control rows: greedy kept it.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

WORK = Path("/root/autodl-tmp/.dsh_checks/sampling")
BASE = Path("/root/autodl-tmp/.dsh_checks/aishell_selector")
K = 5

manifest = json.loads((WORK / "manifest.json").read_text(encoding="utf-8"))
greedy = {json.loads(l)["id"]: json.loads(l) for l in
          (BASE / "aishell_9b_aishell_adapter.predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
samples = []
for k in range(1, K + 1):
    p = WORK / ("sample_%d.predictions.jsonl" % k)
    samples.append({json.loads(l)["id"]: norm(json.loads(l).get("prediction") or "") for l in
                    p.read_text(encoding="utf-8").splitlines() if l.strip()})

print("samples loaded:", [len(s) for s in samples])
print()

for group in ("damaged", "control"):
    rows = [m for m in manifest if m["group"] == group]
    pairs = sum(len(m["hotwords"]) for m in rows)
    hits_per_k = [0] * K
    any_hit = major = all_lost = 0
    uniq_total = 0
    same_as_greedy = [0] * K
    escaped = 0
    examples = []
    for m in rows:
        rid = m["id"]
        g = norm(greedy[rid].get("prediction") or "")
        preds = [s[rid] for s in samples]
        hs = [norm(h) for h in m["hotwords"]]
        for k in range(K):
            hits_per_k[k] += sum(h in preds[k] for h in hs)
            same_as_greedy[k] += preds[k] == g
        n_any = sum(any(h in p for h in hs) for p in preds)
        if n_any >= 1:
            any_hit += 1
        if n_any >= (K // 2 + 1):
            major += 1
        if n_any == 0:
            all_lost += 1
        uniq_total += len(set(preds))
        if any(p != g for p in preds):
            escaped += 1
        if group == "damaged" and 0 < n_any < K and len(examples) < 4:
            k_hit = [i for i, p in enumerate(preds) if any(h in p for h in hs)]
            examples.append((rid, m["hotwords"], g, preds[k_hit[0]]))

    print("### %s: %d rows, %d designated hotwords" % (group, len(rows), pairs))
    if group == "damaged":
        print("  per-sample recovery of the destroyed hotword: %s" % 
              ["%d/%d" % (h, pairs) for h in hits_per_k])
        print("  recovered in >=1 of %d samples : %d/%d rows (%.1f%%)" % (K, any_hit, len(rows), 100 * any_hit / len(rows)))
        print("  recovered by majority (>=%d)   : %d/%d rows (%.1f%%)" % (K // 2 + 1, major, len(rows), 100 * major / len(rows)))
        print("  lost in ALL %d samples          : %d/%d rows (%.1f%%)  <- confidently wrong" % (K, all_lost, len(rows), 100 * all_lost / len(rows)))
    else:
        print("  per-sample preservation of the correct hotword: %s" % 
              ["%d/%d" % (h, pairs) for h in hits_per_k])
        print("  broken in >=1 of %d samples    : %d/%d rows (%.1f%%)" % (K, len(rows) - all_lost, len(rows), 100 * (len(rows) - all_lost) / len(rows)))
        print("  preserved in ALL %d samples     : %d/%d rows (%.1f%%)" % (K, all_lost, len(rows), 100 * all_lost / len(rows)))
    print("  samples identical to greedy     : %s" % ["%d" % s for s in same_as_greedy])
    print("  mean distinct predictions/row   : %.2f of %d" % (uniq_total / len(rows), K))
    print("  rows where any sample differs from greedy: %d/%d (%.1f%%)" % (escaped, len(rows), 100 * escaped / len(rows)))
    if examples:
        print("  --- damaged rows where sampling produced the correct entity in some samples ---")
        for rid, hs, g, hit in examples:
            print("    id=%s hotword=%s" % (rid, hs))
            print("      greedy :", g)
            print("      sample :", hit)
    print()
