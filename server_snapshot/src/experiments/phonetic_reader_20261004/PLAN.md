# Local pronunciation reader v1 — 2026-10-04

User authorized implementation of a connector that makes9B understand pronunciation evidence. Scope: one fixed diagnostic experiment, all17294 existing shard train and1334 historical contextual dev, not official full-dev/test. No new frontend, protection, posthoc candidate scores, reference-aware inference routing, or repeated tuning.

## Representation and learning

Initialize frozen language weights from historical joint B/text final2162, previously CER2.9476% on this repeatedly examined dev. Never update base9B or that old LoRA. Add a separate rank8 evidence LoRA active ONLY on acoustic routes, LR5e-6. Inspected installed PEFT0.19.1 additive LoraModel adapter API is unit-tested; freeze all old parameters after activation.

Use the probe's predeclared primary linear epoch10 phone head, trained on these train rows only, not whichever probe wins dev. Freeze it and the train-only399-syllable vocabulary. Read all existing512D raw cached frames. Audio-only posterior argmax runs, capped16frames, define time-local spans; retain every frame, including blank spans. For each span, soft probabilities weight old9B embeddings of spaced pinyin spellings (blank contributes0). Keep uncertainty/blank mass rather than exporting hard pinyin as an answer. Mix this frozen linguistic anchor with a trainable LN512→256→4096 acoustic residual (last layer zero-initialized) and trainable sigmoid pronunciation gain initial0.10. No Afull acoustic-only connector initialization.

This first version uses ordered local acoustic segments and explicit pronunciation semantics. It does NOT implement explicit input-text-token forced time alignment and must not be described as such.9B attention learns text/audio correspondence under correction training. Preserve continuous acoustic residual rather than using pinyin as a complete bottleneck; tone is not directly supervised in this first version.

Main objective: correction JSON CE on all17294 examples once. Auxiliary:0.2 local audio-only pinyin JSON CE on a predeclared hash-selected train quarter. Each selected window has24frames (or whole shorter audio), hash-chosen independent of reference. Reference/dictionary phones and frozen CTC Viterbi midpoints provide weak TRAIN labels only. Empty local labels retained. No reference, full reference phones, oracle words or timestamps enter correction/auxiliary inference input. Local boundary labels are approximate and may split sounds; not listened gold.

Train only new evidence LoRA and acoustic residual/gain. Batch accumulation8,2162updates (final6examples), seed917,connector LR5e-5,AdamWwd.01,warmup32,linear decay,clip1. All old dropout disabled, new LoRA dropout0. Fixed final2162 checkpoint, no dev selection. Resume from hash-verified complete checkpoints with optimizer/scheduler andPython/NumPy/CPU/CUDA RNG. Keep partial/old outputs; fail rather than shrink data. Save every256updates and final. CPU conservative sequence audit includes all train/dev, max4096.

## Checks and serial workflow

CPU tests: all-frame segmentation incl blank, soft-vs-hard semantics, gradient flow, frozen phone head, repeat-phone CTC alignment, impossible/nonfinite rejection, additive PEFT preserving old route. GPU smoke uses train first/longest/auxiliary examples, confirms losses and gradients, new LoRA changes but frozen parameter versions do not; text-route generation before/after identical. Separate fresh-process reload test verifies exact adapter/connector state, optimizer steps, true generation, text identity and resumed optimization. Smoke weights never initialize formal training.

Serial GPU guard and pipeline lock. Runtime environment `/root/autodl-tmp/great/bin/python`. New remote directory `/root/autodl-tmp/src/experiments/phonetic_reader_20261004`; no modification of old results/checkpoints. Stop on source/cache/hash changes, split mismatch, nonfinite/OOM, unexpected trainable/changed frozen params, failed smoke/reload, or incomplete/misordered evaluation. Structural bugs may be repaired before formal training and protocol re-lock, preserving failed preparations/logs; no result-driven retuning.

## Locked evaluation

ALL1334 historical contextualdev per condition,21102 reference characters,599mentions; six conditions/8004predictions:

- text: exact old B/text prompt, no evidence LoRA or connector. Regenerate and require raw outputs equal to historical B/text, or diagnose mismatch instead of hiding it.
- true: learned connector + new evidence LoRA, correct audio.
- zero: erase raw cached frames, then recompute same interface. Not silence re-encoding; different segmentation counts are a confound.
- mismatch: inherited duration-imperfect permutation.
- no_phonetic: remove soft pinyin semantic values but retain acoustic residual AND phone-derived segmentation. This is partial value ablation, not complete phonetic removal.
- frozen_reader: same learned connector/true audio, disable new evidence LoRA; old9B/oldLoRA still active. Inference ablation, not separately trained matched arm.

Use existing actual evaluator: NFKC,strip whitespace/Unicode punctuation,NOOpenCC. Raw corpus CER/hotword recall/correct-input corruption/S-D-I, all malformed generations scored empty, paired95% bootstrap. No protected scores, fallback or restoration. Data/cache/protocol/checkpoint/prediction hashes and ordered IDs saved. Report all arms and failures, not just winning numbers. No novel heldout/official full-dev claim or CER guarantee. Multi-component experiment cannot establish isolated auxiliary-loss/architecture causality. Prior probe posthoc35dev cases must not be training negatives or routing rules.

After smoke, one fixed training and full-dev suite run automatically, summary saved, then stop. No additional training/search when results are disappointing. Download/audit result artifacts on completion through the authorized monitor, if configured.
