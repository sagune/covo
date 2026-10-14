# AC-COVO run — handover (2026-09-15 23:55)

## What is running right now

Unattended pipeline, **pid 13610**, survives ssh/laptop disconnect:

    bash /root/autodl-tmp/.dsh_checks/run_acoustic_pipeline.sh

Logs / markers (all under `/root/autodl-tmp/.dsh_checks/rerank/`):

| file | meaning |
|---|---|
| `acoustic_pipeline.log` | the pipeline's own step-by-step log |
| `AC_P0_DONE` … `AC_P5_DONE` | phase completion markers |
| `ACOUSTIC_PIPELINE_DONE` | everything finished |
| `ACOUSTIC_FAILED` | **something aborted; read the log** |
| `ac_train.log` | the actual training log (`trainer_log` lines per 20 steps) |
| `acoustic_v1/final/checkpoint-*` | saved every 250 steps (2163 steps = 1 epoch) |
| `acoustic_v1/dev_metrics.tsv` | per-checkpoint AISHELL-dev CER |

Check status any time with:

    ssh -p 27081 root@connect.westb.seetacloud.com
    tail -20 /root/autodl-tmp/.dsh_checks/rerank/acoustic_pipeline.log
    grep -a trainer_log /root/autodl-tmp/.dsh_checks/rerank/ac_train.log | tail -5

## Progress at hand-off

- P0 link: **PASS** — 17,301/17,301 rows paired (punctuation-stripped match 1.0000)
- P1 zero-init integration check: **PASS** — `max_logit_absdiff = 0.0`, i.e. at
  initialisation the acoustic model is *numerically identical* to the text-only
  model, and projector gradients are non-zero (4.62)
- P2 pilot: 20 steps in 345 s (**17.3 s/step** under contention) → 1 epoch
  (2163 steps) ≈ **8–10.5 h**
- P3 full training: **started 23:51:33**, GPU 94 %, 40.8 GB

Expected: training done ~**09:00–10:30**, then P4 (~45 min) and P5 (~1 h) →
everything finished ~**11:00–12:30** on 2026-09-16.

## What was built (all new this session)

| file | role |
|---|---|
| `covo/src/acoustic_conditioning.py` | projector + injection + **shared** span location |
| `covo/scripts/train_acoustic_covo.py` | acoustic-conditioned SFT (baseline trainer + one channel) |
| `covo/scripts/infer_acoustic_covo.py` | acoustic-conditioned generation |
| `src/analysis/export_acoustic_vectors.py` | frozen-encoder per-character frame states |
| `src/analysis/link_acoustic_sft.py` | pairs SFT rows to acoustics and **verifies** the pairing |
| `src/analysis/acoustic_topk_ceiling.py` | RESULTS §17: the acoustic side contains the fix |
| `src/analysis/acoustic_gate_ceiling.py` | RESULTS §17: the gate is the ceiling |
| `src/analysis/probe_acoustic_coupling.py` | checkpoint 0: shapes and frame mapping |
| `DESIGN_acoustic_coupling_20260915.md` | the design and its justification |
| `.dsh_checks/run_acoustic_pipeline.sh` | the unattended orchestrator |

### Facts established by measurement this session

- Encoder width **D = 512**; **CTC frame f == encoder frame f+4** (the 4 query
  tokens), so per-character alignment is 1:1 with **no resampling** — held for
  **all 17,301** training utterances (`enc_ctc_frame_mismatch: 0`).
- `hs_utils.quantize_hidden_states` clamps to [-1,1] but per-frame L2 norm is
  ~34 and **38.5 %** of values exceed 1.0 → that path is unusable here; raw fp16.
- At top-1 error positions the reference character is in the model's own top-5
  (argmax excluded) for **87.50 %**, and is the immediate runner-up for
  **58.82 %** (break-even for this project is ~54 %).
- Best gate over ANY single-character acoustic feature gains only **+48** chars
  (needs +434): content present, criterion present, **gate absent**.
- Per-frame top-5 oracle CER **1.1611 %** vs **2.3254 %** for the whole 24-best
  sentence pool ⇒ inject the per-frame distribution, **not** the N-best list.
- Leak check: training uttids disjoint from ST-CMDS / THCHS-30 / AISHELL-dev (0/0/0).

## Pre-registered gates and the falsification clause

Edit precision ≥ 70 %, net edit chars > 0, beyond-N-best imp not degraded,
hotword recall ≥ 95.98 %, protected-span destructions ≤ 0, ST-CMDS CER < 4.9356 %.

**If the joint gate's precision lands near the ~54 % break-even again, that is the
pre-declared falsification of the joint-gate hypothesis: record the negative
result and stop. Do not add further patches.** If it improves only on ST-CMDS and
not on AISHELL-dev/THCHS, it is dataset adaptation, violates the §15 constraint,
and is to be discarded.

## Known cosmetic defect

The pipeline's pilot-throughput regex expects `"train_steps_per_second": 0.058`
but the log has single quotes (`'0.058'`), so it logs
`WARN: pilot did not report throughput`. Harmless — the pipeline continued to P3
correctly; the real rate is 17.3 s/step.

## Three latent bugs in the inference path (found and fixed 2026-09-16 00:1x)

`infer_acoustic_covo.py` had never been executed. Reviewing it against the real
prediction format found three defects, each of which would have made **every**
P4/P5 number meaningless — and, worse, would have looked like "the acoustics did
not help" rather than like a bug. All three are fixed and covered by CPU-only
tests that run without the GPU.

1. **Right padding for generation.** `collate_acoustic` always padded on the
   right, but batched decoder-only generation requires LEFT padding; with right
   padding the model continues from pad tokens and every output is garbage. Fixed
   with `pad_side="left"` for inference (training still uses right). Critically,
   left padding shifts the sequence, so `acoustic_start` must shift by the pad
   amount too — otherwise every character gets its neighbour's acoustics.
2. **Wrong output slice.** With `inputs_embeds`, HF's `generate` returns ONLY the
   newly generated tokens (there are no prompt token ids to prepend). The code
   sliced off `prompt_len` tokens unconditionally, which discarded the entire
   output. Now discriminated on the returned length.
3. **Raw JSON written as the prediction.** `restore_eval.py` compares `prediction`
   against the reference character by character. The code stored the JSON wrapper
   `{"text":"..."}` instead of the sentence inside it, which would have inflated
   CER by the length of the wrapper on every row. Now parsed via
   `parse_prediction()`, with a fallback to the ASR top-1.

A fourth, smaller defect was fixed in the same parser: the brace matcher counted
braces inside string literals, so a `}` in the corrected sentence aborted the
object early and silently fell back to the raw output.

### Verification (both runnable while training owns the GPU)

    # span location + left-pad shift, real tokenizer, no model
    CUDA_VISIBLE_DEVICES="" python src/analysis/test_acoustic_alignment.py \
      --rows .dsh_checks/rerank/acoustic_v1/eval_aishell.messages.ac.jsonl \
      --bin  .dsh_checks/rerank/ac_aishell_dev.bin \
      --index .dsh_checks/rerank/ac_aishell_dev.jsonl \
      --model <MODEL> --limit 200

    # prediction parser
    CUDA_VISIBLE_DEVICES="" python src/analysis/test_prediction_parser.py

Results: alignment **PASS** — 200/200 spans located, 200/200 spans decode to the
hypothesis, left-pad shift exact (382 -> 457 with pad 75), longest row untouched.
Parser **6/6**.

## The box powers itself off when the result exists

Four processes are chained, all detached from any ssh session:

| pid | process | role |
|---|---|---|
| 31481 | `run_acoustic_pipeline.sh` | P3 training → P4 per-checkpoint dev → P5 transfer |
| 31487 | `train_acoustic_covo.py` | the training itself |
| 55530 | `post_watchdog.sh` | on `ACOUSTIC_PIPELINE_DONE`: pre-flight → final report → ablation → transfer with the checkpoint dev selected → writes **`AC_COVO_VERDICT.txt`** |
| 55800 | `shutdown_watcher.sh` | on completion (or failure): writes `AC_COVO_FINAL_STATE.txt`, then `/sbin/shutdown -h now` |

The shutdown is **gated on real artefacts existing**, not merely on processes exiting — a
machine that powers off with nothing to show is worse than one left running. If neither
the verdict nor the final projector is present, it leaves the box up for inspection
instead of shutting down. It follows this project's existing precedent
(`src/scripts/shutdown_after_stcmds_train_20260820.sh`), including the grace period.

## Look here tomorrow

| file | what it is |
|---|---|
| **`AC_COVO_VERDICT.txt`** | the answer: gates, ablation, best-on-dev transfer |
| `AC_COVO_FINAL_STATE.txt` | what existed when the box powered off |
| `acoustic_pipeline.log` | the pipeline's step-by-step log |
| `dev_metrics_rebuilt.tsv` | per-checkpoint AISHELL-dev CER |
| `acoustic_v1/*.predictions.jsonl` | raw per-arm predictions |
| `ACOUSTIC_FAILED` | present **only** if something aborted |

## After the pipeline finishes: one command

    bash .dsh_checks/run_all_post.sh

Runs, in an order where each step can be read before the next is trusted:

1. **pre-flight** — every input present and complete (stops everything if not);
2. **final report** — per-checkpoint dev table, the six pre-registered gates, and the
   section-15 specificity test; plus a rebuilt `dev_metrics_rebuilt.tsv`;
3. **within-model ablation** — the same trained adapter with and without the acoustic
   channel, the only control in this project that holds the training fixed;
4. **transfer with the checkpoint AISHELL dev actually selected** — the pipeline used
   `final` unconditionally, which deviates from the pre-registered protocol;
5. **constrained arm** — left manual, because it is only worth reading once 2–4 are in.

Nothing is destructive: each step writes to its own `*_best` / `*_ablation` /
`*_constrained` files, so the pipeline's own outputs survive for comparison.

## Frame offset: PROVEN, and a conditioning discrepancy to declare

`src/analysis/test_frame_offset.py` proves the claim everything rests on — that
`enc[f+4]` is the encoder state behind CTC frame f — **functionally** rather than by
shape arithmetic: it pushes the encoder states back through the CTC head and compares
the argmax with the recorded distribution at each candidate offset.

| candidate offset | argmax agreement |
|---|---:|
| f−2 … f+3 | 61–66 % |
| **f+4 (assumed)** | **97.99 %** |
| f+5 … f+7 | 60–63 % |

A 32-point gap between the correct offset and the best wrong one. (A raw max deviation
does *not* discriminate — it is 25.09 at the correct offset because of the conditioning
difference below — so only the argmax comparison is offset-sensitive.)

**Conditioning discrepancy, declared.** `encode_audio` (which produced every CTC
statistic in RESULTS, including the section 17 figures and the constrained-decoding
top-k) conditions the encoder with `language="zh"` and `textnorm=("withitn" if use_itn
else "woitn")` — and the scorer runs WITHOUT `--use-itn`, hence `"woitn"`. The exporter
used `language="auto"`, `textnorm="withitn"`. The query-token *order* is identical; only
the *values* differ.

Consequences, stated plainly:

* the injected representation and the section-17-measured representation are **closely
  related but not identical**, agreeing on **97.99 %** of frames;
* this is **not** a train/serve mismatch — training and inference both use the exporter's
  conditioning, so the coupling is internally consistent and the experiment is valid;
* but the claim "87.5 % of errors have the answer inside the injected representation" is
  verified for a *near-identical* representation, not for the byte-identical one. That
  caveat belongs in the paper.
* the exporter already exposes `--language` / `--textnorm`; a future run wanting exact
  parity with the on-disk CTC statistics should pass `--language zh --textnorm woitn`.
  It must NOT be changed on the current run, because the trained projector is matched to
  the current vectors and swapping them would create a real train/serve mismatch.

## Within-model ablation: the strongest control available

Every other comparison here is between SEPARATELY TRAINED models, so a difference
could come from the data, the optimiser or the LoRA. This one holds all of that fixed:

    python src/analysis/run_acoustic_ablation.py --corpus stcmds

It zeroes the projector and re-runs inference with the SAME trained adapter, prompts
and decoding. Zeroing every parameter makes `proj(x) == 0` exactly -- verified:
zeroed output `max|.| = 0`, rms 0 -- so the acoustic term vanishes while the language
model is untouched. The CER difference is therefore the acoustic channel's
contribution and nothing else.

The script refuses to proceed if the zeroed projector is not an exact no-op, since a
non-zero "off" arm would make the ablation meaningless.

## Evaluating the constrained-decoding arm (after training, GPU free)

The constrained arm has allowed sets only for ST-CMDS, because that is the only
corpus whose per-character acoustic top-k had been extracted. That makes it
ineligible for the section-15 specificity test, which needs more than one corpus.

`run_constrained_arm.py` fills the gap: for any corpus missing its top-k it runs the
frozen extractor and builds the sets, then evaluates the arm.

    python src/analysis/run_constrained_arm.py --corpora stcmds --limit 400   # quick
    python src/analysis/run_constrained_arm.py --corpora stcmds,thchs30,aishell_dev

Supporting fix: the scorer resolves audio by matching the row `id` against the
manifest `id`, but the AISHELL-dev and THCHS-30 pools use POSITIONAL ids while their
manifests are keyed by uttid -- so it matched nothing and exported zero rows.
`make_positional_manifest.py` builds the positional manifests, and asserted the
convention rather than trusting it: **reference agreement 1334/1334 and 2495/2495**
(rate 1.0), so row i of each pool really is line i of that corpus's uttid file.

The script refuses to run without a projector rather than silently generating an
unconditioned arm.

## Protocol deviation in P5, and the fix

The objective requires the checkpoint to be **selected on AISHELL dev**, but the
pipeline's P5 transfers `$OUT/final` unconditionally. That is only correct if the
final model also happens to be the best on dev; if the model peaks before the end of
the epoch, the pre-registered protocol wants that earlier checkpoint and P5 would
report the wrong arm.

**Run this after P4** (needs the GPU free) to produce the protocol-compliant
transfer, writing to `*_best.predictions.jsonl` so it never collides with the
pipeline's own outputs:

    python src/analysis/run_transfer_best.py

It parses the dev metrics out of the pipeline log, picks the lowest-CER checkpoint,
maps it to its directory, verifies the adapter and projector are present, then runs
both one-shot transfers with it.

Selection logic validated against the previous run's log: it correctly identifies
`checkpoint-2163` at 2.2984% and maps it to `final/checkpoint-2163`.

## Pre-flight before P4/P5

    bash .dsh_checks/ac_preflight.sh

Checks presence AND row counts for every input the evaluation needs, because a
missing or TRUNCATED input degrades P4/P5 silently rather than loudly. That is not
hypothetical: the THCHS link was once written as 216 rows instead of 2495, and every
row of the truncated file still matched perfectly.

State before training finished: **21 present, 1 missing** — the final projector, which
appears when training completes. The script exits non-zero until that count is 0.

## Span location and truncation on the REAL training set

Verified on the actual 17,301-row set, not a synthetic sample:

| check | result |
|---|---|
| span not found, trainer's own 500-row sample | **0/500** (`span_fail_rate: 0.0`) |
| span not found, independent 400-row check | **0/400** |
| prompt tokens min / median / mean / max | 873 / **1239** / 1221 / **1547** |
| rows reaching `max_length` 2048 | **0 (0.00%)** |

The longest prompt is 1547 tokens against a 2048 limit, so no training row is
truncated and nothing can cut the hypothesis span the acoustics attach to. Combined
with the 200/200 alignment check on dev and the 300/300 tokenisation-identity check,
the training-side alignment is no longer an open risk.

## Training configuration actually in force

The trainer continues from an existing adapter, so the LoRA hyper-parameters come from
`adapter_config.json` on disk rather than from this run's arguments. Verified they match
the documented plan:

| field | value |
|---|---|
| r / lora_alpha | **16 / 32** (ratio 2.0) |
| lora_dropout | 0.05 |
| bias / task_type | none / CAUSAL_LM |
| target_modules | **7**: q, k, v, o, gate, up, down |
| base model | ../models/Qwen3.5-9B |

The projector is the only newly initialised component, and it gets its own 3e-4 rate
against the LoRA's 2e-5 (256 LLM tensors vs 6 projector tensors).

## Reading the mid-course AISHELL-dev numbers

A dev CER alone says nothing; these are the references to compare against. All are
`restore[deployable]` on the same dev set.

| arm | dev CER % |
|---|---:|
| deployed baseline (before any backend training) | ~2.2936 |
| text-only trained, checkpoint-500 | 2.4500 |
| **text-only trained, checkpoint-2163 / final** | **2.2984** |

How to read the acoustic arm's first dev number:

* **essentially identical to 2.2984** — the acoustic channel changed nothing, which
  points at the channel rather than at the idea (check the projector's injected bias
  against the 0.013466 embedding RMS before concluding anything);
* **materially different in either direction** — the channel is doing something, and
  the ST-CMDS transfer is then worth reading closely;
* note that dev is an *in-domain* set for this training data, so it is the least likely
  place for an acoustic-channel gain to show up; the primary read is ST-CMDS.

Also remember: training loss is NOT a checkpoint-selection signal here. It falls to
~0.31 by step 80 because the target is close to the input, so copying scores well.
Only dev CER selects.

## Is the acoustic channel actually learning? (mid-course check)

Zero-initialising the projector buys a clean controlled comparison, but it carries a
specific failure mode: LoRA can learn the task with the channel permanently switched
off, leaving the projector at ~0. The run would then finish and look like "the
acoustic information does not help", when in fact it was never used.

Check it in ~50 minutes instead of after the whole epoch:

    python src/analysis/check_projector_learning.py \
      --projector .dsh_checks/rerank/acoustic_v1/final/checkpoint-250/acoustic_projector.pt

It needs no GPU and no model weights. Validated on two fixtures: a fresh projector
reports **INERT** (injected bias RMS exactly 0.000000), a perturbed one reports
**ACTIVE** (RMS 0.328766 on real AISHELL-dev acoustic vectors, absmax 1.657).

* INERT at step 250 means the channel never engaged; the primary hypothesis was not
  actually tested and the appropriate response is to train the projector alone first
  (`--freeze-llm`), not to read the outcome as evidence about acoustic coupling.
* ACTIVE means generation is genuinely conditioned on acoustics and the result is
  interpretable.

**Calibration caveat.** Step 250 is only ~11% of the epoch, so INERT there is a
caution flag rather than a verdict -- LoRA may still be adjusting and the channel
may engage later. The decisive read is a later checkpoint (step 1000+). Only if it is
still exactly zero then is the channel genuinely unused.

### The --freeze-llm contingency is verified

If projector-only alignment is needed, that path must already be known-good.
`test_freeze_llm.py` passes **6/6** on a tiny GPT-2, covering the configuration most
likely to fail silently:

* no base/adapter parameter requires grad, all 6 projector tensors do;
* one optimiser step changes all 6 projector tensors and leaves every LoRA
  parameter **bit-identical**;
* forward+backward still runs under **gradient checkpointing with the whole base
  frozen** — the risky case, since a checkpointed block needs at least one input
  that requires grad, and here that input is the projected acoustics.

## SEAL-style constrained decoding (built and verified; opt-in)

The second half of the SEAL recipe, ready to run without touching the pipeline
path (`--constrained-sets` is empty by default, so P4/P5 are unaffected).

| file | role |
|---|---|
| `covo/src/constrained_decoding.py` | the logits processor + set loaders |
| `src/analysis/build_allowed_sets.py` | `tok_top5` -> per-position allowed sets |
| `src/analysis/verify_allowed_sets.py` | proves both channels index the same characters |
| `src/analysis/test_constrained_decoding.py` | 12/12 CPU unit test |

Mechanism: inside the reply's text value, output position p may only emit a
character from `{hypothesis[p]} u {CTC top-k alternatives at p}`, and p advances
with each character emitted. Substitution is free inside the acoustically plausible
set and impossible outside it, so the off-pool hallucination that helped kill the
earlier SFT arm (15.3% of rows) becomes structurally impossible.

**Verification.** 12/12 unit checks, explicitly covering the RELEASE conditions
that keep the constraint from trapping the model: outside the value, after the
value closes, past the end of the hypothesis, and — the one that matters —
positions containing a multi-token character.

**Two findings worth recording:**

* `build_allowed_sets.py` re-indexes `tok_top5` from the NORMALISED hypothesis onto
  the RAW one, the same `raw_to_norm_map` the encoder-state export uses.
  Recomputing coverage from the produced sets gives **87.28%**, against section 17's
  **87.50%** — proof that the encoder-state channel and the constrained-decoding
  channel index the same character at position p. A silent mis-index here would
  have applied the right constraint to the wrong character.
* **530 of 4,943 distinct characters (10.7%) encode to more than one token.** The
  mask can only admit a character's first token, so at such a position the emitted
  token decodes to an incomplete character, the position never advances, and the
  model is masked into re-emitting a token it cannot complete. The processor now
  *releases* the constraint at any position whose allowed set contains a
  multi-token character. On the real ST-CMDS sets this releases only **4.39%** of
  positions, leaving **95.61% (53,508 / 55,965)** constrained; the hypothesis
  character itself is multi-token at just **71** positions.

Run it with:

    --constrained-sets .dsh_checks/rerank/stcmds_allowed_sets.jsonl

## Mechanism verified on a tiny model (CPU, no GPU needed)

`test_generate_mechanism.py` builds a randomly-initialised 2-layer GPT-2 from
config and puts the REAL `AcousticConditionedLM` through the real generation path.
Weights are irrelevant to the questions being asked; the code paths are the real
ones. **5/5 pass:**

| check | result |
|---|---|
| zero-init projector does not change logits | max diff **0** |
| `generate(inputs_embeds=...)` return length | prompt_len 12 → **returned 6** |
| the discriminator picks the right branch | new-token branch = 6, wrong branch = **0** |
| a non-zero projector CHANGES generation | identical: **False** |
| left-padded unequal-length batch | runs, shape (2, 6) |

Two conclusions worth recording:

* **The `inputs_embeds` slice bug was real and fatal.** HF returns ONLY the newly
  generated tokens, so the original `gen[:, prompt_len:]` would have produced an
  empty continuation on every row — empty predictions, meaningless metrics, and it
  would have looked like the acoustics failing rather than a bug.
* **The acoustic channel is genuinely wired into generation.** A non-zero
  projector changes the generated tokens, so the injection is not being silently
  ignored by the wrapper.

## Text interface identity (controlled comparison)

`test_interface_identity.py` compares this trainer's tokenisation against the
baseline `train_lora_sft.py` rendering (`apply_chat_template(..., tokenize=False,
add_generation_prompt=False, enable_thinking=False)` followed by the tokenizer's
default `add_special_tokens=True`, versus this module's explicit `False`).
**300/300 rows tokenise identically**, so the acoustic run differs from the
text-only run in exactly one respect — the acoustic channel — and any change in
behaviour is attributable to it.

## Calibrated schedule (from the previous run's real inference timings)

| step | measured cost |
|---|---|
| AISHELL-dev inference, 1334 rows | **~9 min** per checkpoint |
| ST-CMDS, 5130 rows | ~31 min |
| THCHS-30, 2495 rows | ~22 min |

Training runs at **12.46 s/it → 7 h 25 m** for 2163 steps (started 23:51:33).

    training ends      ~07:17
    P4 (9 checkpoints) ~08:40
    P5 (ST-CMDS + THCHS) ~09:30
    ALL DONE           ~09:30-10:30

This is close to full fine-tuning cost, so the schedule is dominated by training
and cannot be shortened by inspecting fewer checkpoints.

## P4 scrape defect (cosmetic, workaround provided)

P4 reads the CER with `grep -oE "CER=[0-9.]+"`, but `restore_eval.py` prints
`CER  7.3109%` with no `=`. `dev_metrics.tsv` will therefore be full of `none`.
Nothing is lost — the full table is appended to `acoustic_pipeline.log`. Rebuild
the table at any time with:

    python src/analysis/extract_dev_metrics.py \
      --log .dsh_checks/rerank/acoustic_pipeline.log \
      --out .dsh_checks/rerank/dev_metrics_rebuilt.tsv

Validated against the previous run's log, where it reproduces the documented
AISHELL-dev result for checkpoint-2163 (2.2984 %).

## Bug found and repaired during the run (2026-09-16 00:0x)

**Symptom.** `acoustic_v1/eval_thchs.messages.ac.jsonl` had **216 rows** instead
of 2495. THCHS-30 — one of the two required one-shot transfer targets — would
have been evaluated on **8.7 %** of its utterances.

**Cause.** P0 waits for the eval-set exports with a non-empty test:

    [[ -s "$R/ac_aishell_dev.jsonl" && -s "$R/ac_stcmds.jsonl" && -s "$R/ac_thchs30.jsonl" ]]

`-s` is true as soon as the file exists with any bytes, so the THCHS link ran
while its export was still being appended to. The linker then compared only the
overlapping prefix, found a **1.0000 match rate**, and wrote a file truncated to
216 rows — silently, with no warning, because 216 correct rows look exactly like
216 correct rows.

**Repair.** Re-ran the linker against the now-complete index: 2495/2495 rows,
punctuation-stripped match 1.0000. All three eval links verified complete:

| file | rows | expected |
|---|---:|---:|
| `acoustic_v1/eval_aishell.messages.ac.jsonl` | 1334 | 1334 |
| `acoustic_v1/eval_thchs.messages.ac.jsonl` | 2495 | 2495 |
| `e2eSTCMDS_VA.messages.ac.jsonl` | 5130 | 5130 |

**Root-cause fix.** `link_acoustic_sft.py` now REFUSES to write unless the
acoustic index row count equals the row-file count, with a message naming this
exact failure. Verified: feeding it a 100-row index against the 2495-row THCHS
file refuses and writes nothing (previously it wrote a 100-row file reporting
1.0000).

**Still to fix after the pipeline exits** (do NOT edit `run_acoustic_pipeline.sh`
while bash is executing it — bash reads scripts incrementally and an edit can
corrupt a running job): change that P0 wait to compare row counts, e.g. require
`ac_aishell_dev.jsonl` ≥ 1334, `ac_stcmds.jsonl` ≥ 5130, `ac_thchs30.jsonl` ≥ 2495.

## Pre-registered gate report

`src/analysis/ac_gate_report.py` applies all six thresholds mechanically and
applies the falsification clause. Validated against the known-failed text-only
arm, where it reproduces every documented number and returns 0/6:

    python src/analysis/ac_gate_report.py \
      --records <predictions.jsonl> --label <arm> \
      --aligned datasets/stcmds/cb_sensevoice_heldout/hotword/test/aligned.txt \
      --uttid-file datasets/stcmds/cb_sensevoice_heldout/hotword/test/uttid

It distinguishes "precision lands in the 45–58 % break-even band" (falsification)
from "precision is simply low" (as in the text-only arm's 27.1 %).

## Unused file

`covo/scripts/train_lora_sft_acoustic.py` is an 884-line alternative written by a
subagent and **interrupted before verification**. It is not referenced by the
pipeline. Kept only for reference; delete if it causes confusion.
