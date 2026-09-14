#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp
cp -f .dsh_checks/exact_key_informativeness.py src/scripts/rerank_20260914/exact_key_informativeness.py
cp -f /root/autodl-tmp/src/RESULTS_RERANK_20260914.md .dsh_checks/_doc_check.md 2>/dev/null || true
git add src/scripts/rerank_20260914/exact_key_informativeness.py src/RESULTS_RERANK_20260914.md
git -c user.name=agent -c user.email=agent@local commit -q -m "docs(rerank): measure the primary key before blaming it

exact_weighted_score is a hotword-coverage statistic over {0, 0.5, 1}, not an
inert field, and the front-end top-1 already attains the pool maximum in
98.0-99.5%% of rows on every pool.  So the shipped key is a hotword-coverage
guard, and the actual selector among equal-coverage candidates is the acoustic
term -- which is where the length bias acts.  Section 7.2/7.3 restated
accordingly, with the measured per-pool key informativeness table."
git log --oneline -1
