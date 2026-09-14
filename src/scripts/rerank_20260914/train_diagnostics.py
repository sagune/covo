#!/usr/bin/env python3
"""Is the training run actually the one we documented?

Everything downstream assumes: linear LR decay from 2e-5 with 3% warmup, batch 4 x accum 2,
1 epoch over 17,301 rows, bf16 + gradient checkpointing.  The hyper-parameters were checked
before launch; this checks the OBSERVED trajectory against them, plus gradient stability,
from the trainer's own log.  A constant or mis-scheduled LR would silently change what the
run means.

    train_diagnostics.py [--log ...]
"""
import argparse
import json
import re
from pathlib import Path

LOG_DEFAULT = "/root/autodl-tmp/.dsh_checks/rerank/train_out.log"
TOTAL_STEPS = 2163
PEAK_LR = 2e-5
WARMUP_RATIO = 0.03
WARMUP_STEPS = round(TOTAL_STEPS * WARMUP_RATIO)


def expected_lr(step):
    # the scheduler is 0-based: observed LR at step s is (s-1)/warmup * peak during warmup
    # (verified: step 20 -> 19/65 * 2e-5 = 5.846e-6, step 40 -> 39/65 * 2e-5 = 1.200e-5).
    # Using `step` here instead produced a spurious 5% deviation at the boundary.
    if step <= WARMUP_STEPS:
        return PEAK_LR * (step - 1) / max(1, WARMUP_STEPS)
    frac = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return PEAK_LR * max(0.0, 1.0 - frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=LOG_DEFAULT)
    a = ap.parse_args()

    rows = []
    txt = Path(a.log).read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r'\{"trainer_log": (\{.*?\})\}', txt, re.S):
        try:
            d = json.loads(m.group(1))
        except Exception:
            continue
        rows.append(d)

    print("trainer_log entries: %d" % len(rows))
    if not rows:
        return
    steps = [int(r.get("step", 0)) for r in rows]
    lrs = [float(r.get("learning_rate", 0)) for r in rows]
    losses = [float(r.get("loss", 0)) for r in rows]
    gns = [float(r.get("grad_norm", 0)) for r in rows]

    print("steps %d..%d   loss %.4f -> %.4f (min %.4f, max %.4f)"
          % (steps[0], steps[-1], losses[0], losses[-1], min(losses), max(losses)))
    print()
    print("LR schedule check (expected: warmup %d steps to %.2e, then linear to 0 at %d)"
          % (WARMUP_STEPS, PEAK_LR, TOTAL_STEPS))
    worst = 0.0
    for s, lr in list(zip(steps, lrs))[:3] + list(zip(steps, lrs))[-5:]:
        exp = expected_lr(s)
        rel = abs(lr - exp) / max(exp, 1e-12)
        worst = max(worst, rel)
        print("  step %5d  lr %.6e   expected %.6e   rel err %.2f%%" % (s, lr, exp, 100 * rel))
    for s, lr in zip(steps, lrs):
        exp = expected_lr(s)
        worst = max(worst, abs(lr - exp) / max(exp, 1e-12))
    print("  worst relative deviation across all logged steps: %.2f%%" % (100 * worst))
    print("  VERDICT: %s" % ("schedule matches" if worst < 0.05 else
                             "DEViates - investigate"))
    print()
    print("gradient stability")
    print("  grad_norm: max %.4f  mean %.4f   (a spike >10 would suggest instability)"
          % (max(gns), sum(gns) / len(gns)))
    print("  non-finite losses: %d    non-finite grad_norms: %d"
          % (sum(1 for x in losses if x != x or x in (float("inf"), float("-inf"))),
             sum(1 for x in gns if x != x or x in (float("inf"), float("-inf")))))
    rises = [(s, losses[i - 1], losses[i]) for i, s in enumerate(steps)
             if i and losses[i] > losses[i - 1] + 0.5]
    print("  loss jumps > +0.5 between logs: %d %s" % (len(rises), rises[:3]))
    print()
    print("epoch progress: last logged epoch %.4f (of 1.0)" % float(rows[-1].get("epoch", 0)))


if __name__ == "__main__":
    main()
