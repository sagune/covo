# Status — 2026-09-23

Implementation prepared locally; not deployed and no new training launched.

Latest user override: use old AISHELL dev-CER-selected checkpoint30022, not625. experiment.py and PLAN.md now implement three NEW30022 training arms plus its unchanged pure-text baseline. The625 A group cannot be reused. Afull connector remains shared A/B initialization but its625 pretraining is an explicit transfer limitation. No remote code or weights changed.

Read-only SSH inspection succeeded initially: NVIDIA RTX4080SUPER32760MiB idle; approximately48GiB free disk. Existing checkpoint625 config: rank16, alpha32, dropout0.05, targets q/k/v/o_proj and gate/up/down_proj. Confirmed prior shardMLP data/schedule. Later the server closed the active SSH connection; two SCP attempts and a fresh SSH attempt failed during key exchange with connection reset.

Local Python syntax compilation passed for experiment.py and test_experiment.py. CPU unit tests, immutable protocol preparation, GPU smoke tests and checkpoint round-trip checks have NOT run. No experimental success or runtime is established.

Next after connectivity returns:
1. Recheck remote GPU processes, free disk and experiment destination (never overwrite an existing run).
2. Upload the three source/plan files to /root/autodl-tmp/src/experiments/joint_lora_20260923.
3. Verify checkpoint30022 identity/config/weights. Using /root/autodl-tmp/great/bin/python, run test_experiment.py, then experiment.py prepare. Preparation verifies historical data/cache provenance (not reusable625 controls), locks30022 weights and audits every sequence length.
4. Start experiment.py pipeline in a persistent background process with a dedicated log. It performs A/B/C smoke+reload tests and unchanged30022 text baseline before formal training. Observe smoke results and first real training updates before reporting training started.
5. Report measured ETA, monitor same locked scope, then collect full1334dev audited results. No old frontend/full-data queue may be resumed.

Local code has not been runtime-tested; review and fix implementation defects before protocol lock. Do not change locked protocol or legacy sources to conceal failures.
