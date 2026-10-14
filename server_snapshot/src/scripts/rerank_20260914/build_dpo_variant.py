#!/usr/bin/env python3
"""Build a keep-it-weighted DPO variant, for the failure mode RESULTS 10.13 predicts.

The training prior says "change the input" on 75.4% of rows while ST-CMDS only needs it on
35.7%, and 10.12 shows this model's edit precision collapses from 75% to ~50% as the edit
rate rises - below the ~54% break-even, more editing LOSES. So if the trained adapter
misses the target by OVER-editing, the remedy is not more DPO but DPO with the "keep it"
anchors strengthened.

dpo_pairs_aishell.jsonl is 63.4% B_top1_wrong / 24.6% C_keep_it / 12.0% A_ref_out_of_pool.
This oversamples C by repeating it until it reaches --target-share of the file, leaving
B and A untouched.  Repeating rows (rather than synthesising new ones) keeps every pair
traceable to the verified data and is ordinary class balancing.

    build_dpo_variant.py --out <file> [--target-share 0.5]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

R = Path("/root/autodl-tmp/.dsh_checks/rerank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(R / "dpo_pairs_aishell.jsonl"))
    ap.add_argument("--out", default=str(R / "dpo_pairs_aishell_keepit50.jsonl"))
    ap.add_argument("--target-share", type=float, default=0.5,
                    help="desired share of C_keep_it pairs in the output")
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.src).open(encoding="utf-8") if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r.get("pair_type", "?"), []).append(r)
    n = len(rows)
    b, c, other = len(by.get("B_top1_wrong", [])), len(by.get("C_keep_it", [])), 0
    other = n - b - c
    print("source: %d rows  B=%d (%.1f%%)  C=%d (%.1f%%)  A/other=%d (%.1f%%)"
          % (n, b, 100.0 * b / n, c, 100.0 * c / n, other, 100.0 * other / n))

    # solve  (c*k) / (b + other + c*k) = share   ->   k = share*(b+other) / (c*(1-share))
    fixed = b + other
    share = min(max(a.target_share, 0.0), 0.95)
    k = share * fixed / (c * (1 - share)) if c else 1.0
    reps = max(1, int(round(k)))
    print("target C share %.2f  =>  repeat C x%d" % (share, reps))

    out = list(rows)
    for _ in range(reps - 1):
        out.extend(by.get("C_keep_it", []))
    rng = random.Random(a.seed)
    rng.shuffle(out)

    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                           encoding="utf-8")
    cnt = Counter(r.get("pair_type") for r in out)
    print("wrote %s  rows=%d  %s" % (a.out, len(out),
          {k2: "%d (%.1f%%)" % (v, 100.0 * v / len(out)) for k2, v in cnt.most_common()}))


if __name__ == "__main__":
    main()
