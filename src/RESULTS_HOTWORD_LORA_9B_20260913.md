# AISHELL 9B hotword-preservation LoRA — dev-only protocol (2026-09-08 .. 2026-09-13)

This document records the AISHELL continuation-LoRA line that followed the
Qwen3.5-9B scale-up. It exists because that line previously lived only as
git-ignored artifacts under `src/logs/hotword_lora_9b_20260908/`, with no
committed script, protocol or result record.

## Scope and classification

| Item | Value |
|---|---|
| Scope | AISHELL `dev`, full 1334 contextual utterances |
| Hotword metric | 599 designated hotword mentions from `dev/aligned.txt` |
| Input evidence | cached CB-SenseVoice w14 evidence (no acoustic stage rerun) |
| Starting adapter | `qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022` |
| Base model | `models/Qwen3.5-9B` |
| Classification | **dev exploratory / diagnostic — NOT a held-out result** |

The dev split used here was already used for AISHELL checkpoint selection
(`covo/docs/experiment_log.md`, 2026-08-17), so every number below is
optimistic by construction and cannot support a paper claim.

## Metric definitions

`src/logs/hotword_lora_9b_20260908/workflow.py::evaluate`

- `cer` — corpus CER of the model prediction; `baseline_cer` — corpus CER of the
  input `asr_top1`. Both share one reference set and 21,102 reference characters.
- `recall` — designated hotword mentions present in the prediction;
  `baseline_recall` — the same mentions present in the input `asr_top1`.
- Normalization — OpenCC `t2s`, NFKC, whitespace and punctuation removal.
- `hotwords_lost` / `hotwords_gained` — mentions that flipped between input and
  output in either direction.

## The problem this line is about

| System | CER | Designated recall | Lost | Gained |
|---|---:|---:|---:|---:|
| CB-SenseVoice w14 input (`asr_top1`) | 3.8053% | 541/599 = **90.32%** | — | — |
| Qwen3.5-9B checkpoint-30022 output | 3.3362% | 446/599 = **74.46%** | 111 | 16 |

The 9B adapter buys 0.47 points of CER and pays 15.86 points of designated
hotword recall. Neither side dominates, and no configuration in this line
reached the 0.90 recall floor that `workflow.py::rank` uses as its gate.

Two independent checks establish where the loss comes from:

1. **It is inherited, not caused by the continuation.** The untouched starting
   adapter already loses 110 mentions. Of the 111 lost by the continued adapter,
   **99 are lost by both**; the 1-epoch continuation newly loses 12 and recovers
   11. Tuning the continuation cannot fix a defect that is already in the
   starting checkpoint.
2. **It is not a retrieval problem.** **All 111 lost mentions were present in
   the prompt** under the literal line
   `Protected hotwords that must be preserved exactly:`, in utterances where the
   input top-1 was already correct and the N-best candidates agreed with it.
   The observed failure mode is over-correction of correct proper nouns into
   more frequent homophones:

   | Reference = input top-1 | 9B output |
   |---|---|
   | 本报记者**乔加伟**上海报道记者获悉 | 本报记者**乔佳伟**上海报道记者获悉 |
   | 担任中国队翻译的**何潇**也来到了赛场 | 担任中国队翻译的**何霄**也来到了赛场 |
   | **侯冰**一家租住在镇江市润州区 | **侯斌**一家租住在镇江市润州区 |
   | 其次是京投银泰琨御府和**秦禾**北京院子 | 其次是京投银泰琨御府和**亲和**北京院子 |

The same pattern reproduces on AISHELL-NE 808 through the bridge-A ablation
below (input CER 3.7951%, 878 mention hits → 9B output CER 3.9193%, 774 hits).

## 12-hour queue: continued-LoRA hyperparameter sweep

5000 train rows (repair 652 / preserve 2500 / distractor 1000 / fill 848,
dev/test overlap and duplicate references excluded), 1 epoch, assistant-only
loss, base frozen, all runs continued from the same starting adapter.

| Run | CER | Designated recall | Lost | Gained |
|---|---:|---:|---:|---:|
| old_adapter_old_format | 3.4499% | 446/599 (74.46%) | 110 | 15 |
| old_adapter_unified | 3.4689% | 446/599 (74.46%) | 110 | 15 |
| initial_lr2e-5_seed20260908 | 3.4025% | 443/599 (73.96%) | 115 | 17 |
| low_lr1e-5 | 3.4357% | 442/599 (73.79%) | 116 | 17 |
| seed20260909 | 3.3646% | 441/599 (73.62%) | 117 | 17 |
| two_epochs | 3.3646% | 446/599 (74.46%) | 113 | 18 |
| **high_lr3e-5 (selected)** | **3.3362%** | 446/599 (74.46%) | 111 | 16 |
| context_keep_0.5_old | 3.5305% | 441/599 (73.62%) | 115 | 15 |
| context_keep_0.5_selected | 3.4025% | 443/599 (73.96%) | 113 | 15 |

Seven distinct hyperparameter directions move recall by at most 0.84 points and
cluster near the starting adapter's 74.46%. The `context_keep_0.5` rows only
delete KWS/prompt hotwords from the correction input while CB candidates stay
cached, so they are not a no-hotword front-end rerun.

## Bridge prompt ablation (2026-09-08) — negative

A / B / B+ are three prompt-structure variants of the same bridge, evaluated on
AISHELL-NE 808 with the frozen starting adapter.

| Variant | CER | Mention hits (of 942) |
|---|---:|---:|
| Input CB-SenseVoice w14 | 3.7951% | 878 |
| A (original bridge) | 3.9193% | 774 |
| B | 4.0279% | 764 |
| B+ | 4.0185% | 768 |

Prompt restructuring does not recover preservation; all three variants degrade.

## Dual-view ablation (2026-09-09) — negative

Neutral-view evidence added to the correction prompt, three ways.

| Variant | CER | Designated recall |
|---|---:|---:|
| existing best (`high_lr3e-5`) | 3.3362% | 74.46% |
| `cb_only` | 3.8338% | 80.63% |
| `dual` | 4.0138% | 69.12% |
| `delta` | 4.2176% | 66.44% |

Adding a second recognition view raises recall only when it also wrecks CER, and
the explicit change-list variant is the worst of the three. Recorded in
`src/logs/hotword_lora_9b_20260908/dual_ablation_20260909/RESULTS.json`.

## 2026-09-13 diagnostics

### 1. Qwen-likelihood candidate rescoring

Reference-blind candidate set (cached prediction, CB top-1, and each
char-aligned single edit reverted to CB); selection by maximum sum log
probability of the canonical JSON completion plus EOS.

| System | CER | Designated recall | Objective met |
|---|---:|---:|---|
| cached best | 3.3362% | 74.46% | — |
| rescored | **3.2224%** | 75.29% | yes |

The selector is weak: of 2,294 scored candidates, only **21** were both
lower-CER than the original and scored above it. Most available gains are not
reachable by model likelihood alone.

### 2. Token-region attribution (post-hoc)

283 missed-better-candidate pairs, 566 forward passes, split into token-aligned
prefix / middle / suffix regions. The middle region carries the edits, but it
also contains JSON-boundary effects and must not be described as entity-only
without text alignment. Reference used only to identify the missed candidates.

### 3. Acoustic arbitration

Revert the prediction to the neutral top-1 when the forced-CTC acoustic gap
between candidate 1 and candidate 0 is at least `delta`. The gap is computed
from `sensevoice_forced_ctc` scores, where `ctc_hotword_score` is hard-coded to
0, so it carries no hotword preference.

| delta | swaps | CER | Designated recall |
|---:|---:|---:|---:|
| 0 | 1131 | 3.7011% | 82.47% |
| 1 | 293 | 3.5494% | 81.64% |
| 3 | 193 | 3.3220% | 79.97% |
| **5** | **142** | **3.2130%** | **78.80%** |
| 8 | 89 | 3.1608% | 76.96% |
| 15 | 38 | 3.2414% | 74.96% |

At `delta=5`: 142 swaps = 71 fixed / 47 broken / 24 error-neutral, net 26
characters and net +26 hotword mentions. Both numbers are thin.

**Robustness (independent re-check, 400 random half-splits of dev):**

| Protocol | CER improved | Recall improved |
|---|---:|---:|
| fixed `delta=5` | 88.2% of halves (mean +0.1232 pp, range −0.1592 .. +0.4124) | 99.8% (mean +4.34 pp) |
| threshold tuned on one half, evaluated on the other | **48.5%** (mean −0.0325 pp) | 99.8%-class (mean +5.45 pp) |

The recall gain is robust; **the CER reduction is not** — under a nested
protocol it is a coin flip. The rule is also a confidence gate, which
`EXPERIMENT_CONTEXT.md` excludes from the main method, so this stays a
diagnostic.

## Relation to the 2026-08-17 4B/9B decision

`covo/docs/experiment_log.md` had already recorded, on the CB-SenseVoice
combination route: historical 4B CER 0.018336 with 198 worsened utterances vs
9B checkpoint-30022 CER 0.018460 with 504 worsened, concluding *"retain
Qwen3.5-4B as the best CB-SenseVoice combination model on this route"* and
noting the 9B adapter is *"much more willing to edit CB-SenseVoice output
… without additional conservatism or route-specific training"*. The same-length
and anchor-digit fallbacks were measured there and do not close the gap.

This line therefore did not discover a new phenomenon; it built a sharper
instrument for an already-recorded one. The 8/17 decision was made on CER and
worsened counts; the 599-mention designated-recall protocol quantifies the same
defect as a 15.86-point recall loss, which is why it looks far more severe here.

## Conclusions

1. The 9B adapter's ceiling on this protocol is set by **over-correction of
   already-correct proper nouns**, not by KWS retrieval, candidate pool size or
   prompt formatting. All three of those were tested and none is the bottleneck.
2. Neither 9/13 patch is a main-method candidate: both are post-hoc guards, both
   remain below the 0.90 recall floor, and the arbitration rule's CER gain does
   not survive a nested split.
3. The open route is the one the 8/17 decision named: **route-specific
   conservatism training** on CB-SenseVoice-format inputs with explicit
   hotword-preservation supervision, or reverting to 4B for the CB-SenseVoice
   combination.
4. **Never done:** no Qwen3.5-4B adapter has been evaluated on this
   dev-1334 / 599-mention protocol. Since 4B preserves designated hotwords far
   better on AISHELL-NE 808 (90.50% at 3.1354% CER), a single inference run would
   settle whether the 9B scale-up is worth repairing at all.

## Reproduction

```bash
cd /root/autodl-tmp/src/logs/hotword_lora_9b_20260908
# data preparation (writes train.unified.jsonl, dev.*.jsonl; refuses to overwrite)
/root/autodl-tmp/great/bin/python prepare.py
# continued LoRA, 1 epoch, assistant-only loss
/root/autodl-tmp/great/bin/python train.py --run-dir <new-run-dir> --learning-rate 3e-5
# dev evaluation / diagnostics
/root/autodl-tmp/great/bin/python score_diagnostic_20260913.py
/root/autodl-tmp/great/bin/python attribute_scores_20260913.py
```

## Artifacts (git-ignored, on the experiment host only)

```text
src/logs/hotword_lora_9b_20260908/REPORT.md                    queue report
src/logs/hotword_lora_9b_20260908/QUEUE_*.json                 queue plan/results
src/logs/hotword_lora_9b_20260908/scheduled/high_lr3e-5/       selected adapter + predictions
src/logs/hotword_lora_9b_20260908/dual_ablation_20260909/      dual-view ablation
src/logs/hotword_lora_9b_20260908/score_diagnostic_20260913/   rescoring + attribution
src/logs/hotword_lora_9b_20260908/acoustic_arbitration_20260913/ arbitration sweep
```
