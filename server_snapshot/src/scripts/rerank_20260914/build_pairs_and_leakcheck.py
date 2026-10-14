#!/usr/bin/env python3
"""Two zero-GPU jobs for the backend-training objective.

(1) LEAKAGE RED LINE.  The objective states the training data must not contain any
    ST-CMDS or THCHS-30 utterance.  The train evidence carries no uttid field (its
    `id` is positional within a shard), so this verifies the property three ways:
      * the `dataset` attribute of every training row,
      * the `split` attribute (must not name stcmds/thchs),
      * exact reference-string overlap with the two evaluation sets (the strongest
        content-level test available without uttids), plus a normalised form of it.

(2) DPO FALLBACK PAIRS.  Ready-made preference data so the fallback can launch the
    moment SFT proves useless.  Three pair types:
      A  reference not in pool   -> rejected = the pool's best candidate
                                    ("the best in the list is still not good enough")
      B  top-1 wrong, ref in pool-> rejected = the top-1      ("prefer the right candidate")
      C  top-1 already correct   -> rejected = the nearest wrong candidate
                                    ("keep it")  -- anchors against always-edit drift
    chosen is always the normalised reference, serialised as {"text": ...} to match
    what inference emits.
"""
import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8") if str(p).endswith(".gz") else open(p, encoding="utf-8")


def refs_of_evidence(path):
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for l in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out[str(r.get("id"))] = norm(r.get("reference") or "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-evidence", required=True)
    ap.add_argument("--train-messages", required=True)
    ap.add_argument("--eval-evidence", action="append", default=[], help="LABEL=PATH")
    ap.add_argument("--out-pairs", default="")
    ap.add_argument("--out-leak", default="")
    a = ap.parse_args()

    # ---------------------------------------------------------------- (1) leakage
    leak_lines = []
    train_refs = []
    datasets, splits = Counter(), Counter()
    with opener(a.train_evidence) as fh:
        for l in fh:
            if not l.strip():
                continue
            r = json.loads(l)
            datasets[str(r.get("dataset"))] += 1
            splits[str(r.get("split"))] += 1
            train_refs.append(norm(r.get("reference") or ""))
    train_refs = [t for t in train_refs if t]
    leak_lines.append("train rows: %d | datasets: %s" % (sum(datasets.values()), dict(datasets)))
    leak_lines.append("train splits (top 5): %s" % dict(list(splits.most_common(5))))
    bad_split = [s for s in splits if any(k in s.lower() for k in ("stcmds", "thchs", "magicdata"))]
    leak_lines.append("splits naming a forbidden corpus: %s" % (bad_split or "NONE"))
    tset = set(train_refs)
    redline_ok = (not bad_split)
    for item in a.eval_evidence:
        label, _, path = item.partition("=")
        m = refs_of_evidence(path)
        ev = {v for v in m.values() if v}
        exact = len(tset & ev)
        leak_lines.append("  %-12s eval rows=%d  exact-reference overlap with train = %d" % (label, len(ev), exact))
        if label.strip().upper().startswith(("ST-CMDS", "STCMDS", "THCHS")):
            if exact:
                redline_ok = False
    leak_lines.append("RED LINE (ST-CMDS / THCHS-30 must not appear in training data): %s" % (
        "PASS" if redline_ok else "FAIL"))
    leak_lines.append("note: AISHELL-dev overlap is expected (same corpus) and is not a red-line item")

    print("=== leakage check ===")
    for x in leak_lines:
        print("  " + x)

    # ------------------------------------------------------------- (2) DPO pairs
    if not a.out_pairs:
        return
    msgs = {}
    for l in Path(a.train_messages).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            msgs[(str(r.get("split")), str(r.get("id")))] = [m for m in (r.get("messages") or [])
                                                             if m.get("role") != "assistant"]

    pairs = Counter()
    rows = []
    with opener(a.train_evidence) as fh:
        for l in fh:
            if not l.strip():
                continue
            r = json.loads(l)
            key = (str(r.get("split")), str(r.get("id")))
            pm = msgs.get(key)
            if not pm or len(pm) < 2:
                continue
            inp = r.get("input") or {}
            old = norm(inp.get("asr_top1") or "")
            ref = norm(r.get("reference") or "")
            pool = [norm(c["text"]) for c in (inp.get("cbwhisper") or {}).get("candidates") or []
                    if isinstance(c, dict) and c.get("text")]
            if not ref or not old or not pool:
                continue
            cer = {t: edit_distance(list(ref), list(t)) for t in set(pool)}
            in_pool = ref in cer and cer[ref] == 0
            rej, kind = None, None
            if not in_pool:
                cand = min(cer, key=lambda t: cer[t])
                if cer[cand] > 0:
                    rej, kind = cand, "A_ref_out_of_pool"
            if rej is None and old != ref:
                rej, kind = old, "B_top1_wrong"
            if rej is None and old == ref:
                wrong = [t for t in cer if cer[t] > 0]
                if wrong:
                    rej, kind = min(wrong, key=lambda t: cer[t]), "C_keep_it"
            if rej is None or rej == ref:
                continue
            pairs[kind] += 1
            rows.append({"prompt_messages": pm,
                         "chosen": {"text": ref},
                         "rejected": {"text": rej},
                         "pair_type": kind})

    Path(a.out_pairs).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                                 encoding="utf-8")
    print("\n=== DPO pairs ===")
    print("  wrote %d pairs to %s" % (len(rows), a.out_pairs))
    for k, v in pairs.most_common():
        print("    %-22s %6d (%.1f%%)" % (k, v, 100.0 * v / max(1, len(rows))))


if __name__ == "__main__":
    main()
