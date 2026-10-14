# AISHELL full-data connector comparison (2026-09-22)

User authorized full train then full official dev; subsequently chose extension of the existing CBSense shard contexts, not a new 20,000-word frontend.

## Dataset boundary

- Available train waves: 120,418; official labelled: 120,098.
- Remove 1,068 train utterances with normalized reference text also present in full dev, following inherited strict deduplication. Full eligible train: **119,030**. No correctness/error sampling.
- Available dev waves: 14,331; labelled: **14,326**, **205,341** normalized reference characters. The 5 unlabelled dev and 320 unlabelled train waves are explicitly listed in exclusions.
- The historical contextual dev contains only 1,334 utterances. It is not the full official dev.
- No test prediction or test metric is read. Existing adapter/Afull history is inherited; results are diagnostic, not a fresh held-out claim. The old Afull *manifest* contains 1,009 normalized-reference matches to the expanded dev; this does not establish that all those examples were actually trained on.

## Frontend

Keep old train utterances' original shard and cached evidence (17,255 eligible). Assign missing train utterances to `sha256("917|" + utterance_id) mod 18`. No reference-based assignment. Dev uses its fixed 600-word dictionary for all 14,326 utterances, retaining the old 1,334 inputs.

Before reusing outputs, two training examples per train context and two existing dev examples are decoded with the current frontend and compared on the exact whitelisted model evidence. This is a compatibility diagnostic, not a metric-based parameter search. Any mismatch stops the queue; no silent regeneration or substitution. Such checks are not proof of universal numerical identity across environments, so reuse remains an explicitly recorded provenance difference.

Preserve KWS int8 quantization and CPU similarity/resize computation. Do not confuse normalized `withitn` KWS features with unnormalized `woitn` continuous connector features. No reference, oracle selection, or gold keyword routing is supplied to inference. All output IDs and original references are checked on merge.

## Training and evaluation

Run both connectors on the exact same 119,030 rows, same seeded order, one full epoch, accumulation 8, **14,879** optimizer steps, LR 5e-5, frozen 9B and checkpoint-625 adapter. Last step uses the actual remaining 6 examples, never discards them. Save connector/optimizer/scheduler/RNG every 512 steps; preserve logs and stop on failure.

Residual starts at zero acoustic residual; MLP starts from the historical Afull checkpoint. This compares the proposed residual design with the existing warm-started system, **not equal pretraining histories**. No claim of architecture-only causation.

Select the final epoch checkpoint in advance. Each architecture runs all 14,326 dev utterances under text, true, off, zero, mismatch conditions. This totals 143,260 generation rows for the two architectures, including repeated text/off controls. No output protection/restoration rules.

Primary metric: corpus CER with the inherited `restore_eval` normalization (NFKC; remove punctuation/whitespace; **no OpenCC**). Report S/D/I, changes from input, correct-input damage, contextual recall and paired bootstrap confidence intervals. Recall uses unique (utterance, term) presence pairs from the fixed 600-word vocabulary across the full dev. It is explicitly **not** the old 599 annotated-mention denominator.

## Execution

Required interpreter: `/root/autodl-tmp/great/bin/python`.

1. `prepare_full.py audit` — immutable complete split manifests and exclusions.
2. `prepare_full.py export` — check/reuse all train caches; export missing dev audio caches.
3. `queue_full.py --export-pid PID` — wait for exporter, then frontend, prepare/smoke/train both connectors, evaluate both.

All stages are serial. GPU lock and process checks reject concurrent experiments. Missing rows, empty ASR hypotheses, mismatched source hashes, OOM, nonfinite training, malformed output files or failed compatibility stop the queue. No automatic subsampling, protocol change, or new hyperparameter search.

`queue_status.json` identifies actual child PID and log. A queue being alive does not mean training has started. `status.json` records data-preparation progress. Per-architecture `runs/*/status.json` records training/evaluation progress.
