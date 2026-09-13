#!/usr/bin/env python3
"""Where is the pool ceiling set: KWS retrieval, or the beam decoder?

For every designated hotword mention, classify:
  in_pool   - appears in at least one CB candidate
  retrieved - appears in the KWS-retrieved hotword list (input.hotwords)
  injected  - was injected into the decoding prompt (input.prompt_hotwords)

If most missing mentions were never retrieved, the lever is the KWS threshold /
lexicon admission. If they were retrieved but no candidate contains them, the
lever is the decode (hotword bonus weight, beam size, injection cap).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm

SRC = Path("/root/autodl-tmp/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl")
DATA = Path("/root/autodl-tmp/datasets/aishell/data_aishell_sensevoice/hotword/dev")

pos2utt = [l.split()[0] for l in (DATA / "uttid").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
desig = {}
for line in (DATA / "aligned.txt").read_text(encoding="utf-8-sig").splitlines():
    f = line.split("\t")
    if len(f) >= 2:
        desig.setdefault(f[1].strip(), []).append(norm(f[0]))
lexicon = [norm(x) for x in (DATA / "hotword.txt").read_text(encoding="utf-8-sig").split() if x.strip()]

rows = {}
for l in SRC.read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    r = json.loads(l)
    rows[pos2utt[int(r["id"])]] = r


def texts_of(v):
    o = []
    for x in v or []:
        if isinstance(x, str):
            o.append(norm(x))
        elif isinstance(x, dict) and x.get("text"):
            o.append(norm(x["text"]))
    return [t for t in o if t]


cls = Counter()
missing_examples = []
retr_total = retr_in_pool = 0
inj_total = inj_in_pool = 0

for uid, kws in desig.items():
    r = rows.get(uid)
    if not r:
        continue
    inp = r.get("input") or {}
    pool = texts_of((inp.get("cbwhisper") or {}).get("candidates")) or texts_of(inp.get("nbest"))
    hw = set(texts_of(inp.get("hotwords")))
    ph = set(texts_of(inp.get("prompt_hotwords")))
    for k in kws:
        inpool = any(k in c for c in pool)
        retr = any(k == h or k in h for h in hw)
        inj = any(k == h or k in h for h in ph)
        cls[(inpool, retr, inj)] += 1
        if not inpool and len(missing_examples) < 8:
            missing_examples.append((uid, k, retr, inj, sorted(hw)[:5]))
    for h in hw:
        retr_total += 1
        retr_in_pool += any(h in c for c in pool)
    for h in ph:
        inj_total += 1
        inj_in_pool += any(h in c for c in pool)

total = sum(cls.values())
print("lexicon size: %d words | designated mentions: %d" % (len(lexicon), total))
print()
print("%-10s %-11s %-10s %6s %7s" % ("in_pool", "retrieved", "injected", "count", "share"))
for (a, b, c), n in sorted(cls.items(), key=lambda kv: -kv[1]):
    print("%-10s %-11s %-10s %6d %6.1f%%" % (a, b, c, n, 100 * n / total))

nopool = sum(n for (a, b, c), n in cls.items() if not a)
nopool_noretr = sum(n for (a, b, c), n in cls.items() if not a and not b)
nopool_retr = sum(n for (a, b, c), n in cls.items() if not a and b)
print()
print("mentions NOT in any candidate: %d" % nopool)
print("  ...never retrieved by KWS   : %d (%.0f%%)  <- retrieval/lexicon limitation" % (
    nopool_noretr, 100 * nopool_noretr / max(nopool, 1)))
print("  ...retrieved but no candidate contains it: %d (%.0f%%)  <- decode limitation" % (
    nopool_retr, 100 * nopool_retr / max(nopool, 1)))
print()
print("KWS-retrieved hotwords: %d | appear in >=1 candidate: %d (%.1f%%)" % (
    retr_total, retr_in_pool, 100 * retr_in_pool / max(retr_total, 1)))
print("prompt-injected hotwords: %d | appear in >=1 candidate: %d (%.1f%%)" % (
    inj_total, inj_in_pool, 100 * inj_in_pool / max(inj_total, 1)))
print()
print("examples of mentions missing from the pool:")
for uid, k, retr, inj, hw in missing_examples:
    print("  %-18s %-10s retrieved=%-5s injected=%-5s  KWS gave: %s" % (uid, k, retr, inj, hw))
