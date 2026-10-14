#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp
D=src/scripts/rerank_20260914
S=.dsh_checks
cp -f "$S/apply_rerank_v2.py" "$D/apply_rerank_v2.py"
cp -f "$S/rerank_policy_sweep.py" "$D/rerank_policy_sweep.py"
cp -f "$S/rerank_policy_sweep2.py" "$D/rerank_policy_sweep2.py"
cp -f "$S/rerank_policy_sweep4.py" "$D/rerank_policy_sweep4.py"
cp -f "$S/rerank_policy_sweep6.py" "$D/rerank_policy_sweep6.py"
cp -f "$S/ctc_length_bias.py" "$D/ctc_length_bias.py"
cp -f "$S/run_policy_sweep6.sh" "$D/run_policy_sweep6.sh"
cp -f "$S/run_v2_confirm.sh" "$D/run_v2_confirm.sh"
cp -f "$S/run_fixed_rerank_e2e.sh" "$D/run_fixed_rerank_e2e.sh"
cp -f "$S/run_stcmds_orig_fixed_e2e.sh" "$D/run_stcmds_orig_fixed_e2e.sh"
cp -f "$S/run_stcmds_breadth.sh" "$D/run_stcmds_breadth.sh"
git add "$D" src/RESULTS_RERANK_20260914.md
git status --short "$D" src/RESULTS_RERANK_20260914.md
git -c user.name=agent -c user.email=agent@local commit -q -m "fix(rerank): remove the forced-CTC length bias and forbid shortening overrides

The shipped front-end reranker orders pool candidates by
(exact_weighted_score, forced-CTC log-likelihood).  Two defects:

* ctc_sequence_score returns a *total* log-likelihood, so its argmax prefers
  short candidates.  Measured r(len, score) is -0.44 on ST-CMDS (about -1.19
  nats/token) but ~0 on AISHELL/THCHS-30, which is why the defect was invisible
  in-domain.
* nothing prevented the override from deleting content the front-end was already
  confident about: on ST-CMDS the harmed rows lose 0.81 characters on average.

ST-CMDS held-out, original admission (5130 rows / 1718 mentions):
  shipped reranker      5.7921%% -> 6.5417%%  (+0.7496 pp CER)   [regression]
  length-normalised     5.7921%% -> 6.1250%%  (-0.417 pp)
  + no-shorten floor    5.7921%% -> 5.7867%%  (-0.005 pp)        [fixed]

apply_rerank_v2.py keeps exact_weighted_score as the primary key (dropping it
costs ~1.3 pp designated recall on THCHS-30) and adds a per-token normalisation
plus a reference-free constraint that a candidate may never be shorter than the
front-end top-1.  At the relaxed admission the fixed reranker improves CER on
AISHELL (-0.119 pp), THCHS-30 (-0.112 pp) and ST-CMDS (-0.005 pp) with no recall
loss.  Scripts: the full key x floor grid, the length-bias measurement and the
end-to-end runners.  RESULTS_RERANK_20260914.md gains section 7.

Artifacts (evidence, predictions, logs) are deliberately not committed."
git log --oneline -2
