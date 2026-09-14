#!/usr/bin/env python3
"""JSON parse-failure rate: the one gate metric the plan declares but never baselined.

RESULTS 9.5 lists four anti-self-deception indicators: edited rows, JSON parse failure
rate, the beyond-N-best imp/wor partition, and the same-file control.  restore_eval.py
reports three of them; the parse-failure rate was only asserted ("failures show up as
predictions equal to the input").  The predictions files carry the raw generation
(`model_output`), the parsed text (`raw_prediction`) and `parse_warnings`, so it can be
measured exactly - and the trained adapter needs a baseline to be judged against.

    parse_failures.py --records <predictions.jsonl> [--label ...]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

sys_path = "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src"
import sys   # noqa: E402
sys.path.insert(0, sys_path)
from covo.text import normalize_chinese_text as norm   # noqa: E402


def classify(model_output):
    """Return one of: ok | empty | not_json | no_text_key | text_not_str | text_empty."""
    s = (model_output or "").strip()
    if not s:
        return "empty"
    try:
        obj = json.loads(s)
    except Exception:
        # the generator sometimes wraps the object in prose; try the outermost braces
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i:j + 1])
            except Exception:
                return "not_json"
        else:
            return "not_json"
    if not isinstance(obj, dict):
        return "no_text_key"
    if "text" not in obj:
        return "no_text_key"
    if not isinstance(obj["text"], str):
        return "text_not_str"
    return "ok" if obj["text"].strip() else "text_empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    kinds, warns, n = Counter(), Counter(), 0
    silent_fallback = 0
    rel = Counter()
    for line in Path(a.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        k = classify(r.get("model_output"))
        kinds[k] += 1
        for w in (r.get("parse_warnings") or []):
            warns[str(w)[:60]] += 1
        inp = r.get("input") or {}
        top = norm(inp.get("asr_top1") or "")
        pred = norm(r.get("prediction") or "")
        raw = norm(r.get("raw_prediction") or "")
        # a parse failure silently reverting to the input shows up as: final == top-1
        # but the raw generation parsed to something else
        if k != "ok" and pred == top and raw and raw != top:
            silent_fallback += 1
        # what did the model DO, before any protected-span restore
        def texts_of(v):
            o = []
            for x in v or []:
                if isinstance(x, str):
                    o.append(norm(x))
                elif isinstance(x, dict) and x.get("text"):
                    o.append(norm(x["text"]))
            return [t for t in o if t]
        vis = set(texts_of(inp.get("nbest")))
        pool = vis | set(texts_of((inp.get("cbwhisper") or {}).get("candidates")))
        ref = norm(r.get("reference") or "")
        if raw == top:
            rel["raw == top-1 (kept the input)"] += 1
        elif raw in vis:
            rel["raw == a VISIBLE n-best candidate"] += 1
        elif raw in pool:
            rel["raw == a pool-only candidate"] += 1
        else:
            rel["raw NOT in the pool (generated/edited)"] += 1
        if ref and raw == ref:
            rel["raw == the reference (exactly right)"] += 1

    bad = n - kinds["ok"]
    print("# %s   rows=%d" % (a.label or a.records, n))
    print("  parse outcome: %s" % dict(kinds.most_common()))
    print("  FAILURE RATE: %d/%d = %.3f%%" % (bad, n, 100.0 * bad / max(1, n)))
    if warns:
        print("  parse_warnings seen: %s" % dict(warns.most_common(5)))
    else:
        print("  parse_warnings: none recorded")
    print("  silent fallbacks (unparsed AND final==top-1 AND raw!=top-1): %d" % silent_fallback)
    print("  behaviour fingerprint (raw generation vs the evidence, before restore):")
    for k2, c2 in rel.most_common():
        if k2.startswith("raw ==") and "reference" in k2:
            continue
        print("     %-44s %5d  (%.1f%%)" % (k2, c2, 100.0 * c2 / max(1, n)))
    for k2, c2 in rel.most_common():
        if "reference" in k2:
            print("     %-44s %5d  (%.1f%%)" % (k2, c2, 100.0 * c2 / max(1, n)))


if __name__ == "__main__":
    main()
