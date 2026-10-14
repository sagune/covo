#!/usr/bin/env python3
"""Sanity numbers for the selector ablation: what is the best a selector could do?

Reports, on the same rows:
  top-1 CER        the front end COVO receives
  visible oracle   best CER among the n-best list the prompt actually shows
  pool oracle      best CER among all cbwhisper candidates (what a wider list would allow)
This bounds the selector variant and quantifies what `--max-nbest 8` costs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/src")
from covo.text import normalize_chinese_text as norm
from covo.metrics import edit_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    chars = e_top = e_vis = e_pool = 0
    n = 0
    vis_sizes, pool_sizes = [], []
    for l in Path(a.evidence).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        inp = r.get("input") or {}
        ref = norm(r.get("reference") or "")
        top = norm(inp.get("asr_top1") or "")
        vis = [norm(t) for t in (inp.get("nbest") or []) if t]
        pool = [norm(c["text"]) for c in (inp.get("cbwhisper") or {}).get("candidates") or []
                if isinstance(c, dict) and c.get("text")]
        if not ref or not top or not vis:
            continue
        n += 1
        chars += len(ref)
        vis_sizes.append(len(set(vis)))
        pool_sizes.append(len(set(pool)))
        e_top += edit_distance(list(ref), list(top))
        e_vis += min(edit_distance(list(ref), list(t)) for t in vis)
        e_pool += min(edit_distance(list(ref), list(t)) for t in (pool or [top]))

    print("# %s   rows=%d chars=%d" % (a.label, n, chars))
    print("  top-1 CER              %.4f%%   (%.3f edits/utterance)" % (100 * e_top / chars, e_top / n))
    print("  visible-list oracle    %.4f%%   (%.3f edits/utterance)  mean list %.1f" % (
        100 * e_vis / chars, e_vis / n, sum(vis_sizes) / n))
    print("  full-pool oracle       %.4f%%   (%.3f edits/utterance)  mean pool %.1f" % (
        100 * e_pool / chars, e_pool / n, sum(pool_sizes) / n))
    print("  => a perfect SELECTOR over the visible list would reach %.4f%%;" % (100 * e_vis / chars))
    print("     showing the whole pool instead would reach %.4f%%." % (100 * e_pool / chars))


if __name__ == "__main__":
    main()
