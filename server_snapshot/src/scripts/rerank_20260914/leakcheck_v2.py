#!/usr/bin/env python3
"""Independent leak / provenance verification for the backend training set.

The goal makes this a red line: the training data must contain no ST-CMDS or THCHS-30
uttid, and a leak invalidates the whole result.  RESULTS §9.2 verified exact-reference
overlap on the EVIDENCE file, but the trainer actually reads train_sft.jsonl, and that
file carries only {messages, id, reference} - no dataset/split.  So the provenance chain
evidence -> SFT was asserted, never shown.  This script closes both gaps:

  A. provenance   the SFT rows must be a subset of the evidence rows whose dataset=aishell
  B. composition  the evidence file must contain only aishell / train_covo_shard*
  C. exact leak   no SFT reference equals any ST-CMDS or THCHS-30 test reference
  D. fuzzy leak   no shared n-gram (6/8/10/12) between any AISHELL training candidate text
                  (reference + every visible nbest candidate) and any test reference
  E. uttid        no AISHELL uttid appears among the test uttids

CPU only; streams the evidence file so memory stays bounded.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TAR = Path("/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted")
sys.path.insert(0, str(TAR / "covo/src"))
from covo.text import normalize_chinese_text as norm   # noqa: E402

R = Path("/root/autodl-tmp/.dsh_checks/rerank")
WS = Path("/root/autodl-tmp")
EV = R / "train_aishell_evidence.jsonl"
SFT = R / "train_aishell_v1" / "train_sft.jsonl"
TESTS = {
    "ST-CMDS": WS / "datasets/stcmds/cb_sensevoice_heldout/hotword/test",
    "THCHS-30": WS / "datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test",
}
LEVELS = (6, 8, 10, 12)


def read_uttid(p):
    """'uttid transcript' lines -> (list of uttids, list of normalized refs)."""
    uids, refs = [], []
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        f = line.split(maxsplit=1)
        uids.append(f[0])
        refs.append(norm(f[1]) if len(f) > 1 else "")
    return uids, refs


def ngrams(s, n):
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def main():
    print("=" * 100)
    print("A/B. composition and provenance")
    print("=" * 100)
    ds, sp = Counter(), Counter()
    ev_refs = set()
    ev_rows = 0
    for line in EV.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ev_rows += 1
        ds[r.get("dataset")] += 1
        sp[r.get("split")] += 1
        ref = norm(r.get("reference") or "")
        if ref:
            ev_refs.add(ref)
    print("  evidence rows        : %d" % ev_rows)
    print("  dataset values       : %s" % dict(ds))
    print("  split values         : %s" % ("ALL train_covo_shard*" if all(
        str(k).startswith("train_covo_shard") for k in sp) else dict(sp.most_common(8))))
    non_aishell = [k for k in ds if k != "aishell"]
    print("  non-aishell datasets : %s" % (non_aishell or "NONE  <-- red line OK"))

    sft_refs, sft_ids = [], []
    for line in SFT.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        sft_refs.append(norm(r.get("reference") or ""))
        sft_ids.append(r.get("id"))
    sft_set = set(sft_refs)
    print("  SFT rows             : %d   distinct refs %d" % (len(sft_refs), len(sft_set)))
    print("  SFT ids sample       : %s   distinct %d" % (sft_ids[:5], len(set(map(str, sft_ids)))))
    missing = sft_set - ev_refs
    print("  SFT refs NOT in evidence-aishell : %d  %s  <-- provenance %s" % (
        len(missing), list(missing)[:3], "OK" if not missing else "BROKEN"))

    print()
    print("=" * 100)
    print("C/D/E. leak tests against the held-out sets")
    print("=" * 100)
    for name, d in TESTS.items():
        uids, refs = read_uttid(d / "uttid")
        rset = set(r for r in refs if r)
        print("-" * 100)
        print("%s : %d rows, %d distinct references" % (name, len(refs), len(rset)))
        print("  C. exact reference overlap with the SFT training set : %d  <-- %s" % (
            len(rset & sft_set), "red line OK" if not (rset & sft_set) else "LEAK"))
        print("  E. uttid overlap with the training ids               : %d" % (
            len(set(uids) & set(map(str, sft_ids)))))

        # D. fuzzy: index the test n-grams, then stream the training candidate texts
        idx = {n: defaultdict(list) for n in LEVELS}
        for u, r in zip(uids, refs):
            if not r:
                continue
            for n in LEVELS:
                for g in ngrams(r, n):
                    idx[n][g].append(u)
        hits = {n: Counter() for n in LEVELS}
        examples = {n: [] for n in LEVELS}
        ntext = 0
        for line in EV.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            inp = r.get("input") or {}
            texts = [norm(r.get("reference") or "")]
            for c in (inp.get("nbest") or []):
                texts.append(norm(c if isinstance(c, str) else (c or {}).get("text") or ""))
            # hotwords are prompt-side evidence the model also reads, so scan them too
            for key in ("prompt_hotwords", "hotwords"):
                for c in (inp.get(key) or []):
                    texts.append(norm(c if isinstance(c, str) else (c or {}).get("text") or ""))
            for t in texts:
                if not t:
                    continue
                ntext += 1
                for n in LEVELS:
                    seen = set()
                    for g in ngrams(t, n):
                        for u in idx[n].get(g, ()):
                            if u not in seen:
                                seen.add(u)
                                hits[n][u] += 1
                                if len(examples[n]) < 3:
                                    examples[n].append((u, g, t[:40]))
        print("  D. training candidate texts scanned : %d" % ntext)
        for n in LEVELS:
            print("     shared %2d-grams: %d rows affected  %s" % (
                n, len(hits[n]),
                ("examples " + str(examples[n][:2])) if hits[n] else ""))
        worst = max(LEVELS, key=lambda n: len(hits[n]))
        print("  D. verdict: longest shared n-gram level with any hit = %d  <-- %s" % (
            worst if hits[worst] else 0,
            "no overlap at 6+ chars" if not hits[6] else
            ("benign if 6 only (common phrases)" if not hits[8] else "INVESTIGATE")))
    print()
    print("expected: C=0, E=0, and D clean at 10-12.  AISHELL and ST-CMDS/THCHS-30 are")
    print("different corpora, so any 10-gram hit would be a genuine duplicate sentence.")


if __name__ == "__main__":
    main()
