#!/usr/bin/env python3
"""Build the backend SFT set from the AISHELL-train CB evidence.

Source: src/logs/cb_sensevoice_w14_train_evidence_full.jsonl.gz (17,301 rows, every
one carrying a 16-candidate scored pool plus a reference) -- so no decode and no
forced-CTC scoring are needed for this interface.

Interface: rendered by the *same* bridge and the *same* flags the inference arms use
(--rank-mode transmitted), so train and test prompts are identical in form.  That
matters because the adapter we currently ship was trained on a much older, shorter
prompt (mean 371 tokens vs 1015 here).

Target: the normalised reference.  This single target trains both abilities at once,
partitioned by whether the reference happens to be in the pool:
    ref in pool   -> the target is a pool member      => selection
    ref not in poo-> the target is text the pool lacks => editing / extra correction

Row identity: `id` is positional within a shard (the file concatenates shards and the
ids restart), so rows are deduplicated on (split, id).
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stats", default="")
    a = ap.parse_args()

    def opener(p):
        return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")

    rows, seen = [], set()
    dup = noobs = 0
    n_ref_in_pool = n_pool_better = 0
    with opener(a.evidence) as fh:
        for l in fh:
            if not l.strip():
                continue
            r = json.loads(l)
            key = (str(r.get("split")), str(r.get("id")))
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            inp = r.get("input") or {}
            old = norm(inp.get("asr_top1") or "")
            ref = norm(r.get("reference") or "")
            if not old or not ref:
                noobs += 1
                continue
            pool = [norm(c["text"]) for c in (inp.get("cbwhisper") or {}).get("candidates") or []
                    if isinstance(c, dict) and c.get("text")]
            keep = dict(r)
            keep["input"] = {"asr_top1": inp.get("asr_top1"), "nbest": inp.get("nbest"),
                             "hotwords": inp.get("hotwords"), "prompt_hotwords": inp.get("prompt_hotwords"),
                             "cbwhisper": inp.get("cbwhisper")}
            rows.append(keep)
            if ref in pool:
                n_ref_in_pool += 1
            if pool:
                best = min(pool, key=lambda t: edit_distance(list(ref), list(t)))
                if edit_distance(list(ref), list(best)) < edit_distance(list(ref), list(old)):
                    n_pool_better += 1
            if a.limit and len(rows) >= a.limit:
                break

    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                           encoding="utf-8")
    msg = ("rows kept %d | duplicates dropped %d | missing old/ref %d | ref in pool %d (%.1f%%) | "
           "pool candidate beats top-1 %d (%.1f%%)" % (
               len(rows), dup, noobs, n_ref_in_pool, 100.0 * n_ref_in_pool / max(1, len(rows)),
               n_pool_better, 100.0 * n_pool_better / max(1, len(rows))))
    print(msg)
    if a.stats:
        Path(a.stats).write_text(msg + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
