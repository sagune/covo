# CB-Whisper Experiment Context

## Current Task

We are running metric-improvement experiments on the original CB-Whisper codebase. The main research direction is to improve CB-Whisper itself, especially contextual prompting, decoding, n-best rescoring, phonetic matching, candidate selection, repair, and evaluation logic.

The current work is not focused on retraining or replacing the KWS model. KWS is treated as a controlled component unless an experiment is explicitly designed to study KWS.

## Environment

Use the conda environment at:

```text
/root/autodl-tmp/great
```

Do not run experiments in the base environment. Use this Python when launching code:

```text
/root/autodl-tmp/great/bin/python
```

The main experiment config is:

```text
src/configs/cb-whisper-aishell.yaml
```

The Whisper large-v3 adaptation config is:

```text
src/configs/cb-whisper-aishell-v3.yaml
```

Use the v3 config for a separate v2-vs-v3 comparison. Do not overwrite the v2 config, because the accepted nested long-hotword promotion result is currently tied to the large-v2 baseline.

Typical test command shape:

```bash
cd /root/autodl-tmp/src
TRANSFORMERS_VERBOSITY=error \
CBW_DEBUG_LOG=logs/runtime_probe.jsonl \
CBW_DEBUG_MAX_SAMPLES=100 \
CBW_DEBUG_CLEAR_ON_START=1 \
CBW_METRICS_OUT=logs/test_metrics.csv \
/root/autodl-tmp/great/bin/python cb-whisper.py test --config configs/cb-whisper-aishell.yaml
```

## Fixed KWS Checkpoint

Keep this KWS checkpoint fixed unless the experiment is explicitly about KWS modification:

```text
/root/autodl-tmp/src/mlruns/641753688314575260/d9b9fafe87d64bc98400705d2c58525c/checkpoints/f1G-epoch=7-step=48040.ckpt
```

This checkpoint is currently considered the best available KWS model from prior local experiments. It already improves over the original CB-Whisper KWS setup, so changing it would confound CB-Whisper-side experiments.

For large-v3 experiments, keep this KWS checkpoint and the KWS encoder setting fixed. The v3 adaptation should initially change only the ASR generator/processor checkpoint from `openai/whisper-large-v2` to `openai/whisper-large-v3`.

## Research Constraints

CB-Whisper is intended to be a lightweight improvement over Whisper. New methods should preserve that spirit.

Acceptable directions:

- Lightweight contextual prompting changes.
- Lightweight n-best reranking or rescoring.
- Phonetic or lexical scoring that is explainable and cheap.
- Clean candidate selection rules with defensible motivation.
- Paper-friendly methods that can be described as a general algorithm.

Avoid:

- Much longer recognition passes as a final method.
- Many repeated ASR calls per utterance.
- Heavy external models.
- Case-specific patches tuned only to observed errors.
- Overly engineered post-processing that is hard to justify in a paper.

## Main Goal

Primary goal:

```text
Push KWS/entity recall toward 0.90.
```

Secondary goal:

```text
Keep CER as low as possible.
```

Entity Recall is the primary metric for the hotword/contextual-biasing goal. CER, Hotword Only CER, Hotword Sentence CER, and WER are guardrail metrics. A recall gain is not automatically acceptable if CER or WER degrades too much.

## Current Best Accepted Result

The current accepted CB-Whisper-side improvement is nested long-hotword promotion.

Approximate accepted metrics:

| Metric | Value |
| --- | ---: |
| Entity Recall | 0.8486 |
| CER | 0.0820 |
| Hotword Only CER | 0.1081 |
| WER | 0.5149 |

This is only a small gain over baseline, but it improves recall and guardrail metrics at the same time, so it is kept for now.

## Useful Baseline Reference

Baseline metrics before accepted improvements:

| Metric | Value |
| --- | ---: |
| Entity Recall | 0.8475 |
| CER | 0.0822 |
| Hotword Only CER | 0.1087 |
| WER | 0.5161 |

Oracle n-best diagnostics showed limited headroom in the original candidate pool, so large recall gains may require improving candidate availability or changing selection in a way that does not harm CER.

## Experiment Bookkeeping Rules

After every code or config change:

```text
Commit the change to git.
```

After every experiment:

```text
Record the result in README.md and compare it with the previous run and the best accepted run.
```

If a route performs poorly:

1. Try a reasonable fix if the result suggests the direction still has value.
2. If the route remains poor or violates constraints, record the result.
3. Revert the experimental code/config changes.
4. Keep the logs and README record so the failed route remains documented.

## Important Log Files

Main result files:

```text
src/logs/test_metrics.csv
src/logs/runtime_probe.jsonl
src/logs/oracle_nbest_detail_aishell.csv
src/logs/oracle_nbest_summary_aishell.csv
```

Experiment stdout logs are kept under:

```text
src/logs/experiment_*_stdout.log
```

## Lessons From Recent Experiments

- Simple static rerank weight changes did not improve metrics.
- Broader prompt or rescore keyword budgets can hurt CER and recall.
- Decoder hotword bias can improve recall slightly, but tends to damage CER/WER if not tightly constrained.
- Prefix-only hotword bias improved some guardrails but lost recall gains.
- Hard ASR-plausibility guards were too conservative and collapsed recall.
- Candidate exact repair was inactive for current candidate pools.
- Nested long-hotword promotion is useful because it promotes longer competing bias phrases without deleting short true entities.
- Expanding generation candidates to 24 raised recall to about 0.8552 and oracle recall to about 0.8778, but worsened CER and WER and increased runtime. This is useful diagnostically, but not accepted as a final lightweight method in its current form.

## Working Principle

The target is not just a higher number from an ad hoc patch. The target is a defensible CB-Whisper improvement that can be explained in a paper, keeps the KWS model fixed, stays lightweight, improves entity recall, and does not casually sacrifice CER.

## 2026-06-12 Current Context

Current best AISHELL true large-v3 CB-Whisper result:

| Metric | Value |
| --- | ---: |
| Entity Recall | 0.9238 |
| CER | 0.0661 |
| Hotword Only CER | 0.0467 |
| WER | 0.4641 |

This result uses the true large-v3 KWS checkpoint:

```text
outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt
```

The main accepted diagnosis is that KWS retrieval is no longer the dominant AISHELL bottleneck. The useful direction is making Whisper or the downstream corrector use already-available hotword evidence more effectively.

CB-Whisper + covo bridge status:

- `CBW_EVIDENCE_OUT` in `src/model/cb_whisper.py` exports CB-Whisper evidence JSONL.
- `src/analysis/cbwhisper_covo_bridge.py` converts evidence into Qwen/covo messages and can run the migrated LoRA corrector.
- The bridge injects prompt-side CB-Whisper hotwords by default with `--hotword-source prompt`.
- Broad all-KWS hotword injection is available with `--hotword-source all`, but Pilot100 showed it is noisy.
- Do not add a post-hoc gate/selector for this route. The user explicitly rejected gate-style filtering and prefers retraining/fine-tuning if this route continues.

Pilot100 covo observations:

| Variant | COVO Eval CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| No hotword bridge, accept all Qwen rewrites | 0.09073 | 23 | 28 | 49 |
| All KWS hotwords injected | 0.09856 | 22 | 34 | 44 |
| Prompt-side hotwords injected | 0.08486 | 23 | 27 | 50 |

The prompt-side hotword result is the only useful no-gate signal so far, but it was evaluated with a covo LoRA that was not trained on CB-Whisper hotword evidence.

Hotword-aware SFT probe:

- Added `src/analysis/prepare_covo_hotword_sft.py`.
- It converts existing covo ChineseHP/AISHELL rewrite data into the same prompt style used by the CB-Whisper bridge.
- It creates synthetic prompt-side positive hotword evidence from the reference/local edit span and optional confusable KWS-like negatives from N-best.
- Generated local probe files:

```text
cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_cbwhisper_hotword_probe.qwen.jsonl  # 2000 rows
cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_cbwhisper_hotword_probe.qwen.jsonl    # 500 rows
```

Training probe attempt:

```text
output dir: cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_hotword_probe_120steps
log:        cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/logs/qwen35_cbwhisper_hotword_probe_120steps_train.log
base:       cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B
adapter:    cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
```

The first run did not start training because QLoRA loading failed:

```text
ImportError: Using `bitsandbytes` 4-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>=0.46.1`
```

Retried without `--qlora` using bf16 LoRA on the 32 GB RTX 5090. This completed normally:

```text
output dir: cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_hotword_probe_120steps_bf16
log:        cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/logs/qwen35_cbwhisper_hotword_probe_120steps_bf16_train.log
final eval_loss: 0.1956
train_loss:      0.2459
```

Pilot100 evaluation with this continued adapter:

| Variant | COVO Eval CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| Prompt-side hotword bridge, original hardneg LoRA | 0.08486 | 23 | 27 | 50 |
| Prompt-side hotword bridge, 120-step synthetic-hotword SFT | 0.08877 | 23 | 29 | 48 |

Decision: do not scale this exact synthetic-hotword SFT construction. It did not break formatting and it still changed 61/100 samples with 47 exact-reference predictions, but the aggregate CER regressed to the CB-Whisper baseline under the covo evaluator. If this route continues, use real CB-Whisper evidence on a non-test split or improve negative/positive hotword construction before running a longer fine-tune.

Real CB-Whisper evidence probe:

- Exported 200 AISHELL dev samples to `src/logs/cbwhisper_covo_evidence_dev200.jsonl`.
- CB-Whisper dev200 metrics: Entity Recall `0.9369`, CER `0.0725`, Hotword Only CER `0.0601`, WER `0.4550`.
- Full evidence prompt training OOMed after 2 steps because prompt/candidate context was too long.
- Regenerated compact Qwen messages with `max_nbest=4`, `max_pinyin=3`, `max_hotwords=6`, `max_prompt_hotwords=4`, `max_candidates_with_scores=0`.
- Continued `qwen35_text_rewrite_hardneg_dropout_lora_2epoch` for 80 bf16 LoRA steps with gradient checkpointing.
- Output adapter:

```text
cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_real_dev200_compact_80steps_bf16
```

Pilot100 compact-prompt comparison:

| Variant | COVO Eval CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| Compact prompt, original hardneg LoRA | 0.08877 | 25 | 25 | 50 |
| Compact prompt, real-dev200 SFT LoRA | 0.08486 | 23 | 25 | 52 |

This is the first useful no-gate fine-tuning signal. It matches the best previous prompt-hotword Pilot100 CER while using a shorter compact prompt, and it beats the compact-prompt original LoRA baseline. The next sensible scale-up is more real non-test CB-Whisper evidence, not synthetic hotword spans.

Scaled real-evidence probe:

- Added `CBW_EVIDENCE_ONLY=1` to `src/model/cb_whisper.py` so evidence export can skip slow bootstrap metrics.
- Exported 600 AISHELL dev samples to `src/logs/cbwhisper_covo_evidence_dev600.jsonl`.
- Prepared compact bridge messages and randomly split into 500 train / 99 dev rows.
- Continued the hard-negative LoRA for 160 bf16 steps with gradient checkpointing.
- Output adapter:

```text
cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_real_dev600_compact_160steps_bf16
```

Pilot100 compact-prompt comparison:

| Variant | COVO Eval CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| Compact prompt, original hardneg LoRA | 0.08877 | 25 | 25 | 50 |
| Compact prompt, real-dev200 SFT LoRA | 0.08486 | 23 | 25 | 52 |
| Compact prompt, real-dev600 SFT LoRA | 0.08094 | 24 | 23 | 53 |

This is the current best no-gate covo pilot result. It suggests that the useful training signal is real CB-Whisper evidence distribution matching, not synthetic hotword construction. The next scale-up should use more dev evidence or full non-test evidence and then evaluate on the full AISHELL test set, not only Pilot100.

Keep this route lightweight and paper-clean: fine-tune the downstream corrector to use predicted hotword evidence, rather than adding hand-written gates.

Full AISHELL test result, 2026-06-12:

- Evidence: `src/logs/cbwhisper_covo_evidence_test_full.jsonl` with 808 test samples.
- Adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_real_dev600_compact_160steps_bf16`.
- Prediction file: `src/logs/cbwhisper_covo_predictions_test_full_realdev600_compact_ft160.jsonl`.
- COVO evaluator CER: `0.05650`.
- COVO evaluator baseline CER from bridge `input.asr_top1`: `0.10043`. This is the COVO evidence baseline, not the earlier standalone CB-Whisper AISHELL CER table (`~0.0820`).
- Improved / worsened / unchanged: `296 / 96 / 416`.
- Entity Recall check: CB-Whisper input `0.9028`, covo output `0.8541`.

Interpretation: this meets the user's short-term CER target under the covo correction-evaluator normalization, but it is not yet a hotword-recall win. The likely next research step is not a gate; it is training the corrector with an objective or data construction that preserves hotword mentions while still correcting generic ASR errors.

Hotword-preservation oversampling result, 2026-06-12:

- Added `src/analysis/build_covo_preserve_sft_split.py`.
- Input training evidence: `dev600_real_cbwhisper_hotword_compact.qwen.jsonl`.
- Split strategy: split original rows first, then oversample train rows where ASR top-1 already contains at least one true hotword mention.
- Train base rows: `500`.
- Oversampled hotword-preserved rows: `461`.
- Final train rows after repeat=2: `1422`.
- Continued adapter: `qwen35_cbwhisper_real_dev600_compact_160steps_bf16`.
- Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_real_dev600_preserve2_120steps_bf16`.
- Training: 120 bf16 LoRA steps, lr `5e-5`, grad accumulation `8`, final eval_loss `0.2139`, train_loss `0.2068`.

Pilot100 comparison:

| Variant | COVO Eval CER | Entity Recall | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| real-dev600 compact 160-step | 0.08094 | 0.7477 | 24 | 23 | 53 |
| preserve2 continued 120-step | 0.03916 | 0.8037 | 32 | 14 | 54 |

Full AISHELL test comparison:

| Variant | COVO Eval CER | Entity Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real-dev600 compact 160-step | 0.05650 | 0.8541 | 89 | 35 | 296 | 96 | 416 |
| preserve2 continued 120-step | 0.04400 | 0.8785 | 65 | 36 | 309 | 62 | 437 |

Interpretation: this is the best no-gate route so far. It beats the user's CER<6 target by a wide margin under the covo evaluator and partially repairs the hotword recall drop. The remaining gap is that covo output Entity Recall `0.8785` is still below the CB-Whisper input recall `0.9028`; future work should continue along training/data-objective lines, not post-hoc gates.

Preserve4 follow-up, 2026-06-12:

- Built `dev600_real_cbwhisper_hotword_compact_preserve4_*` with preserve repeat `4`.
- Continued from the preserve2 adapter for 80 bf16 steps at lr `3e-5`.
- Final eval_loss `0.2142`, train_loss `0.1689`.
- Pilot100 CER was `0.03916`, same as preserve2.
- Pilot100 Entity Recall was `0.8037`, also same as preserve2.
- Improved / worsened / unchanged was `31 / 13 / 56`, compared with preserve2 `32 / 14 / 54`.

Decision: do not promote preserve4 or run full test unless there is a reason to optimize the improved/worsened count instead of recall/CER. Preserve2 remains the current best because it already has full-test validation.

Full AISHELL-train hotword no-op SFT, 2026-06-14:

- Added `src/analysis/build_aishell_train_hotword_noop_sft.py`.
- Source annotations:
  - `datasets/aishell/train/aligned.txt`
  - `datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt`
- Built `aishell_train_hotword_noop.qwen.jsonl` with `17301` no-op preservation rows.
- Mixed this with `dev600_real_cbwhisper_hotword_compact_preserve2_train.jsonl`, producing `18723` rows.
- Continued from `qwen35_cbwhisper_real_dev600_preserve2_120steps_bf16`.
- Final adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`.
- Training config: one full epoch, batch size `7`, gradient accumulation `4`, lr `2e-5`, bf16, gradient checkpointing.
- Training runtime: about `62` minutes on RTX 5090.
- Final eval_loss `0.2208`, train_loss `0.1704`.

Pilot100 comparison:

| Variant | COVO Eval CER | Entity Recall | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| preserve2 continued 120-step | 0.03916 | 0.8037 | 32 | 14 | 54 |
| + full AISHELL train no-op SFT | 0.03721 | 0.8318 | 31 | 10 | 59 |

Full AISHELL test comparison:

| Variant | COVO Eval CER | Entity Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| preserve2 continued 120-step | 0.04400 | 0.8785 | 65 | 36 | 309 | 62 | 437 |
| + full AISHELL train no-op SFT | 0.04354 | 0.9039 | 39 | 35 | 300 | 41 | 467 |

Interpretation: using the complete AISHELL training hotword alignments worked. It gives the current best no-gate result, with Entity Recall just above the CB-Whisper input recall (`0.9039` vs `0.9028`) and CER still far below 6% under the COVO evaluator. This is stronger than dev600-only preservation because it teaches the corrector a broader prior over real AISHELL hotword surface forms without using test references.

Train-side real-error COVO probe, 2026-06-14:

- Added `src/analysis/build_aishell_hotword_train_split.py`.
- `AishellHotwordDataset` and `DatabaseLite` now accept generated train-style split names such as `train_probe1000` as long as the split folder exists.
- Built full AISHELL `hotword/train` from:
  - `datasets/aishell/train/aligned.txt`
  - `datasets/aishell/train/keywords.txt`
  - `datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt`
  - existing KWS hidden states under `datasets/aishell/data_aishell/kws`
- Full train materialization summary:
  - `17301` usable utterances
  - `20000` keywords
  - keyword hidden states reused by directory symlink
- Built `train_probe1000` for the first practical run:
  - `1000` utterances
  - `2000` keywords
  - per-keyword hidden-state symlinks
  - wav split symlink: `data_aishell/wav/train_probe1000 -> train`
- Export command shape:

```bash
cd /root/autodl-tmp/src
TRANSFORMERS_VERBOSITY=error \
CBW_EVIDENCE_ONLY=1 \
CBW_EVIDENCE_OUT=logs/cbwhisper_covo_evidence_train_probe200.jsonl \
/root/autodl-tmp/great/bin/python cb-whisper.py test \
  --config configs/cb-whisper-aishell-v3-kws.yaml \
  --trainer.limit_test_batches=200 \
  --data.init_args.test_split=train_probe1000 \
  --model.init_args.split=train_probe1000 \
  --model.init_args.oracle_nbest_diagnostic=false
```

- Runtime for 200 evidence rows: about `9m41s`.
- Raw train-probe CB-Whisper top1 statistics:
  - rows: `200`
  - CER: `0.1252`
  - exact matches: `73`
  - keyword recall: `0.8686` (`615 / 708`)
- Converted to compact Qwen/COVO messages:

```text
cbwhisper_covo_migration_20260609_tar_extracted/covo/data/train_probe200_real_cbwhisper_hotword_compact.qwen.jsonl
```

- Short continued training:
  - base adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - output adapter: `qwen35_cbwhisper_train_probe200_realerr_40steps_bf16`
  - max steps: `40`
  - lr: `1e-5`
  - final train_loss: `0.2294`

Pilot100 comparison:

| Variant | COVO Eval CER | Keyword Recall | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.03721 | 0.8136 | 31 | 10 | 59 |
| + train-probe200 real-error SFT | 0.04178 | 0.7542 | 32 | 14 | 54 |

Decision: the clean route is still to train COVO on real CB-Whisper/Whisper errors from train, but 200 rows is too small and overfits, damaging hotword preservation. Do not promote the probe200 adapter. The next run should scale the evidence set substantially and mix it with no-op/preservation rows rather than replacing the training distribution with a tiny real-error set.

Full train-side real-error COVO SFT, 2026-06-15:

- Completed all AISHELL train evidence shards:
  - `train_full_shard00` to `train_full_shard16`: `1000` rows each
  - `train_full_shard17`: `301` rows
  - merged evidence: `src/logs/cbwhisper_covo_evidence_train_full.jsonl`
  - total rows: `17301`
- The shard runner was updated to force local cache loading:
  - commit: `222e304 Use offline cache for AISHELL evidence shards`
  - reason: shard11 previously failed when `hf-mirror.com` refused the processor HEAD request.
- Raw train evidence statistics:
  - CB-Whisper top1 CER: `0.1155`
  - exact match rate: `0.4041`
  - keyword recall: `0.9044` (`56343 / 62296`)
  - average n-best size: `5.42`
  - n-best distribution: `{1: 498, 2: 1435, 3: 1918, 4: 2492, 5: 2281, 6: 2211, 7: 1778, 8: 4688}`
- Converted full evidence to compact COVO messages:

```text
cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_full_real_cbwhisper_hotword_compact.qwen.jsonl
```

- Mixed training set:
  - full train real-error evidence: `17301`
  - full AISHELL train no-op preservation: `17301`
  - dev600 preserve2 rows: `1422`
  - total: `36024`
  - file: `data/processed/chinesehp_aishell1/train_full_real_plus_noop_preserve2.jsonl`
- Continued training:
  - base adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - output adapter: `qwen35_cbwhisper_train_full_real_noop_preserve2_1epoch_bf16_bs7`
  - one epoch, lr `1e-5`, batch size `7`, grad accumulation `4`, bf16
  - runtime: about `2h20m`
  - final eval loss: `0.2133`
  - train loss: `0.2005`

Pilot100 comparison:

| Variant | COVO Eval CER | Keyword Recall | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.03721 | 0.8136 | 31 | 10 | 59 |
| + full train real-error/no-op/preserve2 SFT | 0.04634 | 0.7034 | 30 | 18 | 52 |

Full AISHELL test comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.04354 | 0.9057 | 39 | 35 | 300 | 41 | 467 |
| + full train real-error/no-op/preserve2 SFT | 0.04680 | 0.8316 | 108 | 34 | 312 | 95 | 401 |

Decision: do not promote the full real-error adapter. The real-error data is useful diagnostically, but a full epoch with roughly equal real-error/no-op weighting makes the corrector too aggressive and hurts hotword preservation badly. Future variants should use a much smaller real-error sampling weight, fewer steps from the best no-op adapter, or a preservation-balanced curriculum where no-op/hotword-preserved examples dominate late training.

Hotword-anchored real-error COVO SFT, 2026-06-15:

- Motivation: the full real-error SFT damaged hotword preservation, so this variant only used real-error examples where all true keyword mentions already appeared in ASR top-1. The intended behavior was "fix non-hotword errors while keeping already-hit hotwords".
- Full train evidence filter statistics:
  - rows: `17301`
  - ASR exact rows: `6991`
  - ASR-different rows: `10310`
  - all keyword mentions already in ASR top-1: `13399`
  - any keyword mention in ASR top-1: `16926`
  - all keyword mentions in ASR top-1 and ASR still differs from reference: `6408`
- Mixed training set:
  - full AISHELL train no-op preservation: `17301`
  - dev600 preserve2 rows: `1422`
  - train real-error hotword-anchored rows: `6408`
  - total: `25131`
  - file: `data/processed/chinesehp_aishell1/train_hotword_anchored_real_plus_noop_preserve2.jsonl`
- Continued training:
  - base adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - output adapter: `qwen35_cbwhisper_hotword_anchored_real_300steps_bf16_bs7`
  - max steps: `300`
  - lr: `5e-6`
  - batch size: `7`
  - grad accumulation: `4`
  - bf16 + gradient checkpointing
  - runtime: about `32m26s`
  - final eval loss: `0.2176`
  - train loss: `0.1948`

Pilot100 comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.03721 | 0.8136 | 13 | 4 | 31 | 10 | 59 |
| + hotword-anchored real-error 300 steps | 0.04504 | 0.7373 | 22 | 4 | 30 | 16 | 54 |

Additional strict-prompt probe:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords |
| --- | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.03721 | 0.8136 | 13 | 4 |
| best no-op adapter + stronger hotword preservation prompt | 0.03786 | 0.8136 | 13 | 4 |

Error pattern:

- The model still rewrites already-present rare proper nouns or organization names into more common homophones/variants.
- Observed examples include:
  - `许玮甯 -> 许玮宁`
  - `今久 -> 金九`
  - `宋芳 -> 孙芳` / `颂芳`
  - `刘澄 -> 刘成`
  - `杨锋 -> 杨峰`
  - `布赖恩克尔扎尼奇 -> 布莱恩克尔扎尼基`

Decision: do not promote this adapter and do not run full-test evaluation for it. The current best remains `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`. The diagnosis is now stronger: generic real-error SFT, even filtered to rows where hotwords are already present, teaches the corrector a language-prior rewrite behavior that conflicts with CB-Whisper's hotword evidence. Next attempts should use contrastive hotword-preservation targets, much lower real-error sampling weight, or training examples that explicitly penalize homophone replacement of evidence hotwords.

Contrastive hotword-preservation DPO probe, 2026-06-16:

- Added `src/analysis/build_covo_hotword_dpo_pairs.py`.
- Data construction:
  - source: `train_full_real_cbwhisper_hotword_compact.qwen.jsonl`
  - keep prompt-side hotwords only
  - chosen response: reference text
  - rejected response: an n-best candidate that is close to reference but drops at least one prompt-side hotword mention
  - output pair file: `data/processed/chinesehp_aishell1/train_hotword_preserve_dpo_pairs.jsonl`
  - total pairs: `2583`
- Example pattern:
  - prompt evidence contains `金泉`
  - chosen: `她们是来自韩国金泉大学的留学生`
  - rejected: `他们是来自韩国金权大学的留学生`
- Training details:
  - base adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - DPO script: `covo/scripts/train_lora_dpo_text.py`
  - output adapter: `qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16`
  - max steps: `30`
  - lr: `5e-6`
  - beta: `0.08`
  - sft_weight: `0.03`
  - max_length: `1400`
  - max_prompt_length: `1152`
  - grad accumulation: `4`
  - bf16 + gradient checkpointing
- A first `120`-step attempt was stopped early because DPO with policy/ref models was too slow for a quick probe. A `15`-step weak probe was also trained but not full-tested because Pilot100 did not improve CER.

Pilot100 comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.03721 | 0.8136 | 13 | 4 | 31 | 10 | 59 |
| DPO 15 steps | 0.03786 | 0.8305 | 11 | 4 | 30 | 10 | 60 |
| DPO 30 steps | 0.03460 | 0.8475 | 9 | 4 | 29 | 8 | 63 |

Full AISHELL test comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.04354 | 0.9057 | 39 | 35 | 300 | 41 | 467 |
| DPO 30 steps | 0.04501 | 0.9227 | 22 | 34 | 272 | 22 | 514 |

Observed fixes on Pilot100:

- `宋芳` preserved instead of `颂芳`
- `杨锋` preserved instead of `杨峰`
- `今久` restored in several sentences where the best no-op model output `金九`

Decision: contrastive DPO clearly works for hotword preservation and gives the best recall-oriented COVO variant so far, but it is not the CER-best model. Do not replace `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7` as the main result. Keep `qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16` as a useful ablation/high-recall variant. The next route should add CER-preserving preferences or use lower DPO strength so the full-test recall gain does not cost CER.

Full-pass contrastive DPO, 2026-06-16:

- Goal: test whether the positive 30-step DPO signal scales to the full preference set.
- Pair set:
  - `train_hotword_preserve_dpo_pairs.jsonl`
  - `2583` real n-best preference pairs
- Training:
  - base adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - output adapter: `qwen35_cbwhisper_hotword_preserve_dpo_full_lr2e6_beta003_bf16`
  - max steps: `650`, approximately one full pass with grad accumulation `4`
  - lr: `2e-6`
  - beta: `0.03`
  - sft_weight: `0.03`
  - max_length: `1400`
  - max_prompt_length: `1152`
  - final logged step: `650`

Full AISHELL test comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.04354 | 0.9057 | 39 | 35 | 300 | 41 | 467 |
| DPO 30 steps | 0.04501 | 0.9227 | 22 | 34 | 272 | 22 | 514 |
| full-pass DPO, lr2e-6 beta0.03 | 0.05658 | 0.9089 | 32 | 31 | 244 | 58 | 506 |

Decision: do not promote full-pass DPO. It does not preserve the 30-step recall gain and badly hurts CER. The likely cause is over-optimizing on hotword-preservation pairs without enough CER-preserving preferences. The short 30-step DPO remains useful as a high-recall ablation, but the main result remains the no-op preservation adapter.

Mixed DPO probes, 2026-06-16:

- Added `src/analysis/build_covo_mixed_dpo_pairs.py`.
- Pair types:
  - `hotword_preserve_candidate_dpo`: chosen reference, rejected close n-best candidate that drops a prompt hotword
  - `cer_candidate_dpo`: chosen lower-CER n-best candidate, rejected higher-CER n-best candidate
  - `noop_conservative_dpo`: chosen ASR top-1 when it is already close to reference, rejected worse n-best candidate
- Balanced three-way set:
  - file: `train_mixed_hotword_cer_noop_dpo_pairs.jsonl`
  - counts: `2583` hotword + `2583` CER + `2583` no-op = `7749`
- Hotword+CER set:
  - file: `train_hotword_cer_dpo_pairs.jsonl`
  - counts: `2583` hotword + `2583` CER = `5166`

Full AISHELL test comparison:

| Variant | COVO Eval CER | Keyword Recall | Lost hotwords | Gained hotwords | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best no-op adapter | 0.04354 | 0.9057 | 39 | 35 | 300 | 41 | 467 |
| hotword-only DPO 30 steps | 0.04501 | 0.9227 | 22 | 34 | 272 | 22 | 514 |
| mixed hotword+CER+no-op DPO 20 steps | 0.05728 | 0.9142 | 15 | 19 | 218 | 18 | 572 |
| mixed hotword+CER+no-op DPO 60 steps | 0.08188 | 0.9121 | 1 | 3 | 92 | 2 | 714 |
| hotword+CER DPO 30 steps | 0.06271 | 0.9206 | 5 | 15 | 189 | 10 | 609 |

Decision: mixed DPO confirms the trade-off but does not solve it. Adding CER/no-op preferences makes the model too conservative; it prevents hotword loss but suppresses many useful corrections, so CER worsens sharply. Keep the short hotword-only DPO 30-step adapter as the only useful DPO ablation. The main result remains the no-op preservation SFT adapter.

AISHELL COVO error audit, 2026-06-16:

- Added `src/analysis/covo_error_audit.py`.
- Audited current best full-test predictions:
  - predictions: `src/logs/cbwhisper_covo_predictions_test_full_aishell_train_noop_bs7.jsonl`
  - summary: `src/logs/covo_error_audit_best_noop_summary.json`
  - cases: `src/logs/covo_error_audit_best_noop_cases.csv`

CER / oracle ceiling:

| Source | CER | Exact samples |
| --- | ---: | ---: |
| COVO bridge input baseline | 0.10043 | 368 / 808 |
| n-best oracle only | 0.06178 | 486 / 808 |
| current COVO best | 0.04354 | 523 / 808 |
| oracle over current COVO + n-best | 0.03097 | N/A |
| current COVO best after OpenCC t2s normalization | 0.04261 | 525 / 808 |

Delta categories for current COVO best:

| Category | Count |
| --- | ---: |
| CB-Whisper correct and COVO kept correct | 342 |
| CB-Whisper correct but COVO broke it | 26 |
| CB-Whisper wrong and COVO fixed exactly | 181 |
| CB-Whisper wrong and COVO partially improved | 119 |
| CB-Whisper wrong and COVO left same-distance error | 125 |
| CB-Whisper wrong and COVO worsened | 15 |

Oracle reachability:

| Item | Count |
| --- | ---: |
| n-best better than current COVO | 113 |
| n-best better than CB-Whisper input | 263 |
| current COVO better than n-best oracle | 165 |

Hotword audit:

| Item | Count / Rate |
| --- | ---: |
| true keyword mentions | 942 |
| CB-Whisper input recall | 0.9098 |
| current COVO recall | 0.9055 |
| n-best oracle recall | 0.9108 |
| base-hit hotwords lost by COVO | 39 |
| hotwords gained by COVO | 35 |
| false prompt hotword insertions in CB-Whisper input | 24 |
| false prompt hotword insertions in COVO output | 8 |

Interpretation:

- Current COVO is not just choosing among CB-Whisper n-best; it already beats the n-best oracle (`0.04354` vs `0.06178`), so the ChineseHP-style text-rewrite ability is contributing real corrections.
- The best possible selector over current COVO output plus n-best reaches `0.03097`, which is closer to but still above the ChineseHP `0.0277` reference point. This suggests part of the gap is an evidence/candidate limitation, not just a training objective issue.
- Traditional-to-simplified normalization only improves CER by about `0.00093`, so normalization mismatch is not the main remaining bottleneck.
- The largest immediately actionable bucket is the `113` samples where n-best beats current COVO. A lightweight selector or confidence model between COVO output and n-best could recover some CER without asking the generator to learn everything.
- The risk bucket is small but important: `26` originally correct samples are broken by COVO and `15` wrong samples become worse. Any further method needs to reduce these without becoming as conservative as the mixed DPO probes.

Targeted / longer SFT probes, 2026-06-16:

- Added `src/analysis/build_covo_targeted_sft.py`.
- Added JSON trainer logging to COVO `scripts/train_lora_sft.py` so `loss`, `eval_loss`, `learning_rate`, `grad_norm`, and step are visible in stdout.
- Added `--lr-scheduler-type` to support constant-lr continuation runs.

Targeted SFT variants from current best:

| Variant | Train data | Steps / LR | Eval loss | Full CER | Keyword Recall | Lost hotwords | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted false+nbest+noop | 2800 false-hotword rejection + 2800 n-best repair + 2800 noop | 120 / 5e-6 | N/A | 0.05029 | 0.8432 | 98 | reject |
| targeted nbest+noop | 2800 n-best repair + 5600 noop | 60 / 3e-6 | N/A | 0.04463 | 0.8888 | 56 | reject |
| targeted nbest+noop safe | 1400 n-best repair + 7000 noop | 200 / 1e-6 linear | 0.21948 | 0.04447 | 0.8919 | 52 | reject |
| full-real+noop checkpoint-250 | `train_full_real_plus_noop_preserve2.jsonl` | 250 / 5e-6 constant | 0.21882 | 0.04563 | 0.8496 | 90 | reject |
| full-real+noop final-500 | `train_full_real_plus_noop_preserve2.jsonl` | 500 / 5e-6 constant | 0.21634 | 0.04711 | 0.8379 | 102 | reject |

Interpretation:

- The targeted `false_hotword_rejection` objective is actively harmful: it teaches the model not to trust hotword evidence and collapses recall.
- The `n-best repair` objective is less harmful but still pulls the model away from hotword preservation; longer training did not recover the current best.
- Full-real continuation genuinely trains (`eval_loss` improved from `0.2187` to `0.2163`), but lower loss did not translate to better full-test behavior. It increases rewrite aggressiveness and loses already-present hotwords.
- Current conclusion: do not keep pushing generic SFT volume. The main adapter remains `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`. Further model-capability work needs a target that rewards preserving true hotword spans while correcting surrounding text, not plain reference SFT on noisy KWS prompts.

Legacy CER recalculation, 2026-06-17:

- Reason: the `0.10043` COVO baseline CER was computed with `covo/scripts/evaluate_correction_jsonl.py` as corpus edit distance over normalized text. Earlier CB-Whisper `test_metrics.csv` CER uses a different structure: per-sample CER after `cb_whisper.py` surface normalization, then averaging over samples.
- Recalculated with the current `cb_whisper.py` normalization logic:

| Variant | Legacy mean-sample CER | Normalized corpus CER |
| --- | ---: | ---: |
| COVO bridge input baseline | 0.06600 | 0.06750 |
| current best no-op adapter | 0.04297 | 0.04149 |
| hotword-only DPO 30 steps | 0.04417 | 0.04234 |
| full-real+noop final-500 | 0.04943 | 0.04583 |

- Interpretation: the current best remains best under the old CB-Whisper-style mean CER. The earlier `0.10043` number should be described only as the COVO correction-evaluator baseline, not as the standalone CB-Whisper CER.

Complete AISHELL test-set check, 2026-06-17:

- Clarification: the earlier COVO "full test" evidence/prediction files contain the AISHELL hotword subset (`808` rows), not the full AISHELL test split (`7176` wavs).
- Added `src/analysis/aishell_full_whisper_decode.py` for ordinary full-split Whisper decoding and CER calculation.
- Ran full AISHELL test with `openai/whisper-large-v3`, greedy decoding:
  - output: `src/logs/aishell_full_whisper_large_v3_test_full.jsonl`
  - samples: `7176`
  - mean-sample CER: `0.09060`
  - corpus CER: `0.09023`
- Converted these 7176 outputs through `src/analysis/cbwhisper_covo_bridge.py` and ran the current best no-gate adapter:
  - adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - messages: `src/logs/aishell_full_covo_messages_large_v3_test_full_best_noop.jsonl`
  - predictions: `src/logs/aishell_full_covo_predictions_large_v3_test_full_best_noop.jsonl`
  - COVO evaluator CER: `0.06137`
  - COVO evaluator baseline CER: `0.09288`
  - improved / worsened / unchanged: `1184 / 20 / 5972`
- Recomputed with the older stripped-text口径:
  - baseline mean-sample CER / corpus CER: `0.08070 / 0.08044`
  - COVO mean-sample CER / corpus CER: `0.05752 / 0.05589`
- Interpretation: this is the first true complete-AISHELL result. It meets the "CER below 6%" target under the old mean-sample/full-split口径, but it is not a hotword-recall result; hotword recall should still be reported only on the 808-row hotword subset.

Full AISHELL n-best sanity check, 2026-06-17:

- Added `--num-return-sequences` to `src/analysis/aishell_full_whisper_decode.py`.
- Updated `src/analysis/cbwhisper_covo_bridge.py` so ordinary full-split Whisper decode JSONL can pass top-level `nbest` into COVO prompts.
- Ran `openai/whisper-large-v3` with `num_beams=5`, `num_return_sequences=5`, batch size 2 on all `7176` AISHELL test wavs:
  - output: `src/logs/aishell_full_whisper_large_v3_nbest5_test_full.jsonl`
  - runtime: about `62` minutes
  - top1 mean-sample CER: `0.08735`
  - top1 corpus CER: `0.08730`
  - average unique normalized n-best size: `1.0000`
  - samples with more than one unique hypothesis: `0`
  - n-best oracle mean/corpus CER: same as top1 (`0.08735 / 0.08730`)
- Interpretation: full-split ordinary HF Whisper beam search does not provide useful diverse n-best candidates; all returned beams collapse after surface normalization. The n-best evidence used in earlier CB-Whisper/COVO work comes from `PBAWhisper` with KWS prompt injection and shortform candidate processing, not from plain full-split Whisper generation. Therefore the previous full-AISHELL COVO run had no n-best because the complete-split path bypassed CB-Whisper's PBA/KWS candidate-generation pipeline.

Sampled diverse full-AISHELL n-best check, 2026-06-17:

- Added sampled n-best support to `src/analysis/aishell_full_whisper_decode.py`, keeping greedy top1 and adding de-duplicated sampled candidates.
- Ran `openai/whisper-large-v3` on all `7176` AISHELL test wavs with greedy top1 plus temperature `0.8` sampling:
  - output: `src/logs/aishell_full_whisper_large_v3_sample_t08_nbest5_test_full.jsonl`
  - samples: `7176`
  - average unique n-best size: `2.6506`
  - samples with more than one unique candidate: `4332`
  - samples with five unique candidates: `1943`
  - top1 mean-sample CER / corpus CER: `0.09050 / 0.09012`
  - n-best oracle mean-sample CER / corpus CER: `0.06471 / 0.06558`
- Interpretation: sampling solves the candidate-diversity problem partly, unlike deterministic beam search. But the oracle ceiling is still not strong enough to beat the best complete-AISHELL COVO result (`0.05752` old mean-sample CER, or `0.06137` under the COVO evaluator). The next step, if pursued, should be a conservative COVO rerun with this sampled n-best evidence, not treating sampled n-best selection as a standalone solution.

AISHELL 808 hotword multi-prompt n-best check, 2026-06-18:

- Added `src/analysis/aishell_hotword_multiprompt_nbest.py` for an isolated hotword-subset candidate-pool test.
- Input:
  - evidence: `src/logs/cbwhisper_covo_evidence_test_full.jsonl`
  - uttids/audio: `datasets/aishell/data_aishell/hotword/test/uttid` and `datasets/aishell/data_aishell/wav/test`
  - hotword source: existing CB-Whisper/KWS `prompt_hotwords` followed by `hotwords`
- Decode setup:
  - `openai/whisper-large-v3`
  - prompt variants: no prompt, parenthesized top-3 hotwords, natural-language top-3 hotwords, natural-language top-6 hotwords
  - temperatures: `0` and `0.4`
  - `num_beams=5`, `num_return_sequences=5`, de-duplicated max n-best `12`
  - output: `src/logs/aishell_hotword_multiprompt_nbest_test808_t04.jsonl`
  - summary: `src/logs/aishell_hotword_multiprompt_nbest_test808_t04_summary.json`
- Multi-prompt candidates alone:
  - rows: `808`
  - average unique n-best: `3.9295`
  - samples with more than one unique candidate: `634`
  - samples hitting max n-best 12: `41`
  - top1 mean/corpus CER: `0.11230 / 0.10776`
  - oracle mean/corpus CER: `0.03842 / 0.03917`
  - top1 hotword recall: `0.4672`
  - oracle hotword recall: `0.9364`
- Full-normalization oracle comparison:

| Candidate Pool | Avg Unique | Oracle Mean CER | Oracle Corpus CER | Reference In Candidates |
| --- | ---: | ---: | ---: | ---: |
| CB-Whisper evidence n-best only | 6.0965 | 0.03293 | 0.03344 | 576 / 808 |
| multi-prompt Whisper only | 3.9245 | 0.03842 | 0.03917 | 540 / 808 |
| CB-Whisper + multi-prompt union | 8.0087 | 0.02711 | 0.02725 | 606 / 808 |

- Interpretation: multi-prompt Whisper should not replace CB-Whisper top1 because its top1 quality is bad, but it adds complementary candidates. The union oracle (`0.02711`) is finally in the ChineseHP-like range and slightly below the previously measured ChineseHP test n-best oracle (`0.02818`). The next useful step is a conservative COVO/reranker run over the union candidate pool, with the original CB-Whisper top1 preserved as the default candidate.

CB-Whisper candidate-quality work, 2026-06-26:

- Goal: make CB-Whisper produce ChineseHP-like useful 10-best candidates, not just a single strong top1 or many normalized duplicates.
- Added `src/analysis/build_cbwhisper_candidate_pool.py` as an offline diagnostic bridge. It keeps CB-Whisper top1 first, adds CB-Whisper scored candidates and complementary multi-prompt Whisper candidates, removes normalized duplicates, and applies only light quality checks (empty text, extreme length ratio, obvious repetition).
- Diagnostic input:
  - CB evidence: `src/logs/cbwhisper_covo_evidence_test_full.jsonl`
  - multi-prompt candidates: `src/logs/aishell_hotword_multiprompt_nbest_test808_t04.jsonl`
  - output: `src/logs/cbwhisper_candidate_pool_test808_t04.jsonl`
  - summary: `src/logs/cbwhisper_candidate_pool_test808_t04_summary.json`
- Result on the 808 AISHELL hotword subset:
  - average unique n-best: `7.4022`
  - samples with at least 5 candidates: `679 / 808`
  - samples with 10 candidates: `287 / 808`
  - exact reference in candidate pool: `595 / 808`
  - top1 mean/corpus CER: `0.06600 / 0.06750`
  - oracle mean/corpus CER: `0.03003 / 0.03058`
  - top1 hotword recall: `0.9299`
  - oracle hotword recall: `0.9448`
- Comparison with CB evidence alone:
  - CB-only average unique n-best: `6.0965`
  - CB-only oracle mean/corpus CER: `0.03293 / 0.03344`
  - CB-only exact reference in candidates: `576 / 808`
  - The diagnostic pool improves the oracle and exact-reference count, but still does not consistently reach ChineseHP's near-10 unique candidates.
- Code change: `CBWhisper` now separates target n-best from generation breadth through `rescore_generation_factor`, `rescore_generation_cap`, and optional `covo_nbest`. AISHELL config is set to target `10` candidates and generate up to `32` candidates before de-duplication. This is intended to improve candidate quality at the generation stage while keeping the existing rerank/selection behavior unchanged.

AISHELL v3 10-best generation run, 2026-06-26:

- Config: `src/configs/cb-whisper-aishell-v3-kws.yaml`
- Change: set `rescore_nbest=10`, `rescore_generation_factor=3`, `rescore_generation_cap=32`, `covo_nbest=10`.
- Output files:
  - metrics: `src/logs/test_metrics_aishell_v3_nbest10_gen32.csv`
  - evidence: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32.jsonl`
  - stdout: `src/logs/experiment_aishell_v3_nbest10_gen32_stdout.log`
- Runtime: about `28m33s` for 808 AISHELL hotword test rows; slower than the earlier 8-best baseline because generation now asks for up to 30 returned sequences before de-duplication.
- Final CB-Whisper metrics:
  - Entity Recall: `0.92707`
  - CER: `0.07102`
  - Hotword Only CER: `0.04531`
  - WER: `0.46535`
- Candidate-pool diagnostics from exported evidence:
  - average unique n-best: `8.8205`
  - rows with 10 unique candidates: `524 / 808`
  - exact reference in n-best: `612 / 808`
  - top1 corpus CER under candidate diagnostic normalization: `0.07238`
  - n-best oracle corpus CER: `0.02864`
  - top1 hotword recall: `0.9363`
  - oracle hotword recall: `0.9448`
- Comparison:
  - Prior CB-only evidence pool: avg unique `6.0965`, oracle corpus CER `0.03344`, exact reference `576 / 808`.
  - Offline CB + multi-prompt pool: avg unique `7.4022`, oracle corpus CER `0.03058`, exact reference `595 / 808`.
  - New integrated CB-Whisper 10-best run: avg unique `8.8205`, oracle corpus CER `0.02864`, exact reference `612 / 808`.
- Interpretation: this is the strongest candidate-quality result so far and moves CB-Whisper close to the ChineseHP-style 10-best regime without using offline multi-prompt union. It improves the candidate oracle substantially, but top1 CER is worse than the previous best CB-Whisper endpoint, so the gain should be used mainly for downstream COVO/reranker input rather than reported as a standalone ASR improvement.

10-best candidate quality cleanup, 2026-06-26:

- Motivation: manual/automatic inspection found a small number of meaningless n-best candidates in the new 10-best pool, mostly truncation, long hallucinated tails, repeated tokens, or English video-site tails such as `YoYo Television Series Exclusive`.
- Added a conservative COVO/evidence candidate quality filter in `CBWhisper`:
  - enabled by `enable_covo_candidate_quality_filter`
  - keeps the final CB-Whisper top1
  - de-duplicates by normalized surface text
  - filters only extreme length outliers, repeated-heavy strings, and long Latin tails
  - does not filter by top1-prefix/suffix because top1 itself can be truncated (`美商务部` case), and that rule hurt useful candidates.
- Added `src/analysis/clean_covo_nbest_quality.py` to clean already-exported evidence files with the same conservative policy.
- Cleaned the 10-best evidence:
  - input: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32.jsonl`
  - output: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean.jsonl`
  - summary: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean_summary.json`
- Cleanup result:
  - dropped candidates: `28`
  - drop reasons: `too_long=13`, `too_short=12`, `repeat_heavy=1`, `latin_tail=2`
  - average unique n-best: `8.7859` (from `8.8205`)
  - rows with 10 unique candidates: `510 / 808` (from `524 / 808`)
  - exact reference in n-best: unchanged at `612 / 808`
  - oracle corpus CER: unchanged at `0.02864`
  - oracle hotword recall: unchanged at `0.9448`
- Interpretation: the conservative filter removes obvious garbage without reducing the candidate oracle. This is safer than aggressive prefix/suffix filtering, which mistakenly removed useful completions and worsened oracle CER.

Targeted supplement for near-10-best diversity, 2026-06-26:

- Goal: push AISHELL hotword evidence from average unique n-best `8.8-9.0` toward ChineseHP-like `9.8+` while keeping candidates meaningful.
- Added `src/analysis/targeted_supplement_covo_nbest.py`.
- Procedure:
  1. Start from the cleaned integrated 10-best evidence.
  2. Merge existing multi-prompt candidates and full-AISHELL sampled Whisper candidates.
  3. For rows still below 10 unique candidates, run targeted supplementary Whisper sampling only on those low-diversity rows.
  4. Stop per row once 10 normalized unique candidates are obtained.
- Cheap file-only merge:
  - output: `src/logs/cbwhisper_candidate_pool_v3_nbest10_clean_plus_mp_fullsample.jsonl`
  - average unique n-best: `9.2649`
  - rows with 10 candidates: `613 / 808`
  - oracle corpus CER: `0.02787`
- Targeted supplement round 1:
  - output: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement.jsonl`
  - processed low-diversity rows: `195`
  - average unique n-best: `9.6720`
  - rows with 10 candidates: `722 / 808`
  - oracle corpus CER: `0.02787`
- Targeted supplement round 2:
  - output: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2.jsonl`
  - processed remaining low-diversity rows: `86`
  - average unique n-best: `10.0000`
  - rows with 10 candidates: `808 / 808`
  - oracle corpus CER: `0.02787`
- Strong-cleaned round 2:
  - output: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean.jsonl`
  - average unique n-best: `9.9097`
  - rows with 10 candidates: `789 / 808`
  - dropped candidates: `73`
  - exact reference in n-best: `617 / 808`
  - oracle corpus CER: `0.02864`
  - oracle hotword recall: `0.9459`
- Interpretation:
  - The target `Avg unique n-best >= 9.8` is achievable.
  - The safest candidate-oracle version is the unclean round-2 supplement (`10.0` avg, oracle `0.02787`), but it may contain some high-temperature tail artifacts.
  - The cleaner/reportable candidate-quality version is `round2_clean` (`9.9097` avg, oracle `0.02864`), but its stronger anchor-suffix rule can remove a few useful completions, so it should be compared against the unclean version before downstream COVO use.

Cleanliness-first 9.8+ candidate pool, 2026-06-26:

- User requirement: prioritize candidate cleanliness; do not reach `9.8+` by padding with high-temperature garbage.
- Updated the COVO n-best quality filter:
  - severe pollution is removed before length statistics are computed
  - severe pollution includes video/platform tails (`请不吝点赞`, `订阅`, `转发`, `打赏`, `明镜`, `点点栏目`, `优优独播`, `YoYo`, `Television`, `Exclusive`, `Series`), Unicode replacement character `�`, long Latin tails, and repeated-heavy strings
  - polluted top1 can be removed from `input.nbest` while the original `input.asr_top1` remains available separately
  - aggressive anchor-suffix filtering is optional and stays disabled by default in CB-Whisper because it can remove useful full-sentence completions
- Final cleanliness-first file:
  - output: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`
  - summary: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5_summary.json`
- Final metrics:
  - average unique n-best: `9.9332`
  - rows with 10 candidates: `779 / 808`
  - exact reference in n-best: `616 / 808`
  - oracle corpus CER: `0.03019`
  - oracle hotword recall: `0.9459`
  - dropped candidates: `54` (`replacement_char=16`, `bad_phrase=20`, `too_short=8`, `too_long=9`, `repeat_heavy=1`)
- Final audit:
  - bad phrase tails: `0`
  - long Latin tails: `0`
  - Unicode replacement chars: `0`
  - repeated-heavy candidates: `0`
  - extreme length outliers under the strict audit: `0`
- Interpretation: this version satisfies the `9.8+` diversity target while enforcing a clean candidate pool. It is the preferred candidate file for downstream COVO/reranker testing when cleanliness matters more than the absolute best oracle CER.

COVO on cleanliness-first 9.8+ n-best, 2026-06-26:

- Input evidence: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`
- Messages: `src/logs/cbwhisper_covo_messages_nbest10_clean_strict_slack5.jsonl`
- Predictions: `src/logs/cbwhisper_covo_predictions_nbest10_clean_strict_slack5_best_noop.jsonl`
- Adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
- Bridge settings: `max_nbest=10`, `include_pinyin`, prompt hotwords only, batch size `7`, disable thinking.
- COVO evaluator:
  - baseline CER: `0.10446`
  - prediction CER: `0.04594`
  - improved / worsened / unchanged: `310 / 48 / 450`
- Legacy CB-Whisper-style normalization:
  - baseline mean/corpus CER: `0.07102 / 0.07238`
  - prediction mean/corpus CER: `0.04564 / 0.04327`
- Audit:
  - n-best oracle CER under COVO evaluator: `0.05014`
  - prediction-or-nbest oracle CER: `0.02895`
  - exact matches: base `369`, prediction `513`, n-best oracle `519`
  - hotword recall: base `0.9151`, prediction `0.8907`, n-best oracle `0.9172`
  - base-hit hotwords lost by COVO: `55`
  - hotwords gained by COVO: `32`
- Interpretation: the clean 9.8+ candidate pool improves candidate diversity, but the existing no-op COVO adapter does not exploit it well enough. It is slightly worse than the current best no-op result (`0.04354` COVO evaluator CER, legacy mean around `0.04297`) and loses too many hotwords. Do not promote this COVO output. The next step should be either COVO prompt/training adaptation for 10-best evidence or a learned/rule-light selector before COVO, not just more candidates.

COVO reliability-label/top6 probe, 2026-06-27:

- Diagnosis before the probe: many new COVO errors are not candidate-pool failures. In the `max_nbest=10` run, better candidates often already existed in the n-best list, but COVO selected or generated a worse homophone/variant. Typical failures include `杨锋 -> 杨峰`, `今久 -> 金九`, and `马塞洛特 -> 马赛洛特`.
- Code change: `src/analysis/cbwhisper_covo_bridge.py` now:
  - normalizes n-best surfaces for matching.
  - labels each hypothesis as `trusted_scored` if it matches `input.cbwhisper.candidates`, otherwise `supplemental_unscored`.
  - annotates `keeps_prompt_hotwords` and `keeps_context_hotwords` for each hypothesis.
  - tells COVO that supplemental candidates are evidence only and should not alone override ASR top-1 or a trusted scored candidate, especially when that would replace an already-present prompt hotword with a common homophone.
- Pilot100 on the cleanliness-first pool:
  - `max_nbest=10` reliability labels: COVO evaluator CER `0.04243`; hotword recall `0.7797`.
  - `max_nbest=6` reliability labels: COVO evaluator CER `0.04047`; hotword recall `0.7797`.
  - `max_nbest=4` reliability labels: COVO evaluator CER `0.04112`; hotword recall `0.7712`.
  - `max_nbest=6` plus stronger preservation instruction: COVO evaluator CER `0.03982`; hotword recall `0.7966`.
- Full AISHELL-hotword 808 run with the best pilot setting:
  - Input evidence: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`
  - Messages: `src/logs/cbwhisper_covo_messages_nbest6_reliability_v2_full.jsonl`
  - Predictions: `src/logs/cbwhisper_covo_predictions_nbest6_reliability_v2_full.jsonl`
  - Adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - COVO evaluator CER: base `0.10446`, prediction `0.04494`.
  - Improved / worsened / unchanged: `314 / 43 / 451`.
  - Legacy mean/corpus CER: base `0.10281 / 0.10477`, prediction `0.04671 / 0.04494`.
  - Audit: exact matches prediction `524`, n-best oracle `519`; hotword recall base `0.9151`, prediction `0.9034`, n-best oracle `0.9172`; base-hit hotwords lost by COVO `46`, hotwords gained by COVO `35`.
- Comparison:
  - Previous `max_nbest=10` clean-pool COVO: evaluator CER `0.04594`, hotword recall `0.8907`, lost base-hit hotwords `55`.
  - Reliability/top6 improves both CER and hotword preservation relative to that failed 10-best run.
  - It still does not beat the current best no-gate result on the older evidence (`0.04354` evaluator CER, legacy mean around `0.04297`).
- Interpretation: labels and a smaller trusted candidate set partially repair COVO's use of the rich pool, but prompt-only adaptation is insufficient. The next paper-clean route is to train COVO on reliability-labeled expanded n-best prompts, with hard negatives where supplemental homophones are present but the target preserves the trusted/prompt hotword.

Compact reliability-label COVO training probe, 2026-06-27:

- Motivation: the first reliability-labeled prompt was too long for efficient SFT. A token-length check on 2000 train rows showed p50 around `1512` tokens and p90 around `1758`; `max_length=1024` would truncate most examples.
- Compact prompt construction:
  - `max_nbest=6`
  - `max_pinyin=3`
  - `max_hotwords=6`
  - `max_prompt_hotwords=4`
  - `max_candidates_with_scores=0`
  - `hotword_source=all`
  - keep the reliability labels and hotword-preservation annotations in the n-best lines.
- Compact train/eval files:
  - train: `covo/data/processed/chinesehp_aishell1/train_full_reliability_nbest6_compact.qwen.jsonl`
  - eval: `covo/data/processed/chinesehp_aishell1/dev600_reliability_nbest6_compact.qwen.jsonl`
  - train rows: `17301`
  - eval rows: `600`
- Compact prompt-only full-test result with current best adapter:
  - messages: `src/logs/cbwhisper_covo_messages_nbest6_reliability_compact_full.jsonl`
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_reliability_compact_full.jsonl`
  - COVO evaluator CER: `0.04408`
  - improved / worsened / unchanged: `313 / 40 / 455`
  - legacy mean/corpus CER: `0.04625 / 0.04408`
  - hotword recall: `0.9108`
  - interpretation: removing redundant evidence made the prompt easier for the existing adapter to use; this was better than the longer reliability prompt.
- 80-step compact reliability SFT:
  - start adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - output adapter: `outputs/qwen35_cbwhisper_reliability_nbest6_compact_80steps_bf16`
  - settings: `max_length=1280`, lr `2e-6`, constant scheduler, bf16, batch `2`, grad accumulation `10`, max steps `80`.
  - train loss: `0.7567`; final eval loss: `0.4383`
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_reliability_compact_ft80_full.jsonl`
  - COVO evaluator CER: `0.04362`
  - improved / worsened / unchanged: `319 / 45 / 444`
  - legacy mean/corpus CER: `0.04564 / 0.04362`
  - hotword recall: `0.9076`
- 40-step compact reliability SFT:
  - output adapter: `outputs/qwen35_cbwhisper_reliability_nbest6_compact_40steps_bf16`
  - final eval loss: `0.6857`
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_reliability_compact_ft40_full.jsonl`
  - COVO evaluator CER: `0.04362`
  - improved / worsened / unchanged: `316 / 42 / 450`
  - legacy mean/corpus CER: `0.04574 / 0.04362`
  - hotword recall: `0.9087`
- Comparison:
  - prior clean-pool `max_nbest=10` COVO: evaluator CER `0.04594`, hotword recall `0.8907`
  - reliability/top6 long prompt: evaluator CER `0.04494`, hotword recall `0.9034`
  - compact prompt-only: evaluator CER `0.04408`, hotword recall `0.9108`
  - compact SFT: evaluator CER `0.04362`, hotword recall `0.9076-0.9087`
  - old current best no-gate adapter on older evidence: evaluator CER `0.04354`, legacy mean around `0.04297`, hotword recall around `0.9039`
- Interpretation: retraining is useful but modest. The compact reliability-labeled route almost matches the old CER-best result while preserving more hotwords and using the richer clean n-best pool, but it is not yet a decisive new main result. The next variant should not simply train longer; 80 steps reduced eval loss and mean CER slightly, but also reduced hotword recall versus 40-step/prompt-only. A better next step is a small preference or SFT mix focused on keeping compact-prompt recall while recovering the last `~0.0001` CER gap.

Supported-hotword protection prompt probe, 2026-06-27:

- Motivation: error audit showed many failures where the correct hotword had already been injected into COVO and even appeared in ASR top-1 or trusted n-best, but COVO rewrote it to a common homophone or variant, e.g. `今久 -> 金九`, `杨锋 -> 杨峰`, `宋芳 -> 颂芳`, `瓯文 -> 欧文`, `邬迪 -> 吴迪`.
- Code change: `src/analysis/cbwhisper_covo_bridge.py` gained `--protect-supported-hotwords`.
  - It injects a hard prompt clause saying supported prompt hotwords must be preserved exactly.
  - A hotword is protected only if it is a prompt hotword, appears in ASR top-1 or trusted scored candidates, and is supported by CB-Whisper exact/consensus scoring evidence.
  - This avoids protecting obvious surface-repair false positives such as the first-sample `中心 -> 钟欣` repair, where `钟欣` appears in text but is not in matched/consensus scoring evidence.
- Full 808 run:
  - input evidence: `src/logs/cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`
  - adapter: `outputs/qwen35_cbwhisper_reliability_nbest6_compact_80steps_bf16`
  - messages: `src/logs/cbwhisper_covo_messages_nbest6_hotword_protect_ft80_full.jsonl`
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_hotword_protect_ft80_full.jsonl`
  - COVO evaluator CER: `0.04362`, unchanged from compact ft80.
  - improved / worsened / unchanged: `316 / 45 / 447`, slightly worse than compact ft80's `319 / 45 / 444`.
  - wrong final rows: `278 -> 283`.
  - correct hotword given but final missed: `68 -> 68`.
  - correct hotword in trusted candidate but final missed: `38 -> 38`.
  - false-hotword wrong rows: `19 -> 19`.
  - changed predictions: `32`; it fixed `杨幼萍` in one sample but introduced another trusted-hotword miss (`蚌飞市 -> 蚌淅市`) and several unrelated regressions.
- Interpretation: prompt-only hard preservation is not enough. The model sees the hotwords, but an instruction saying "do not change them" does not reliably override its learned preference for common homophones/variants. The next useful direction is not stronger prompt wording; it should be training-side hard negatives or preference data where the chosen answer preserves supported hotwords and the rejected answer uses common homophones or supplemental-candidate artifacts.

Supported-hotword protection training probes, 2026-06-27:

- Data:
  - Rebuilt compact train/dev prompts with `--protect-supported-hotwords`.
  - train: `covo/data/processed/chinesehp_aishell1/train_full_reliability_nbest6_compact_protect.qwen.jsonl`, `17301` rows.
  - dev: `covo/data/processed/chinesehp_aishell1/dev600_reliability_nbest6_compact_protect.qwen.jsonl`, `600` rows.
- Mixed DPO attempt:
  - pair file: `train_reliability_nbest6_compact_protect_mixed_dpo_pairs.jsonl`, `5431` pairs.
  - pair mix: `2231` hotword-preserve, `1600` CER-candidate, `1600` noop-conservative.
  - start adapter: `qwen35_cbwhisper_reliability_nbest6_compact_80steps_bf16`.
  - output adapter: `outputs/qwen35_cbwhisper_protect_mixed_dpo_60steps_bf16`.
  - DPO settings: `60` steps, lr `2e-6`, beta `0.03`, SFT weight `0.2`, bf16.
  - Full 808 result: CER `0.05378`, improved/worsened/unchanged `242 / 11 / 555`.
  - Interpretation: bad. DPO made the model too conservative, preserving more hotwords but failing to correct many ordinary ASR errors. Do not promote this adapter.
- Protected-prompt SFT attempt:
  - start adapter: `qwen35_cbwhisper_reliability_nbest6_compact_80steps_bf16`.
  - output adapter: `outputs/qwen35_cbwhisper_protect_sft40_from_ft80_bf16`.
  - settings: `40` steps, lr `1e-6`, constant scheduler, bf16, batch `2`, grad accumulation `10`, max length `1280`.
  - train loss decreased from about `0.75` to `0.5901`; dev eval loss decreased from `0.6133` at step 20 to `0.5333` at step 40.
  - full-test messages: `src/logs/cbwhisper_covo_messages_nbest6_protect_sft40_from_ft80_full.jsonl`.
  - full-test predictions: `src/logs/cbwhisper_covo_predictions_nbest6_protect_sft40_from_ft80_full.jsonl`.
  - COVO evaluator: base CER `0.10446`, prediction CER `0.04338`, improved/worsened/unchanged `317 / 45 / 446`.
  - Audit summary: `src/logs/covo_error_audit_nbest6_protect_sft40_from_ft80_full_summary.json`.
  - Audit CER: prediction `0.04338`, n-best oracle `0.05014`, prediction-or-nbest oracle `0.02833`.
  - Exact matches: `524`.
  - Hotword recall: `0.90977`; base-hit hotwords lost by prediction: `41`; false prompt hotwords in prediction: `12`.
- Comparison under the same audit/evaluator family:
  - old best no-gate adapter on older evidence: CER `0.04354`, exact `523`, hotword recall `0.90552`.
  - compact reliability ft80: CER `0.04362`, exact `528`, hotword recall `0.90764`.
  - compact reliability ft40: CER `0.04362`, exact `526`, hotword recall `0.90870`.
  - compact prompt-only: CER `0.04408`, exact `522`, hotword recall `0.91083`.
  - protected-prompt SFT40: CER `0.04338`, exact `524`, hotword recall `0.90977`.
- Interpretation: this is the current best CER result on the 808 AISHELL hotword test with the rich clean n-best pool, and it also improves hotword recall over the old CER-best result. The gain is small but real. DPO with easy candidate-level pairs is not suitable; if more training is attempted, use SFT or construct harder rejected outputs from the model's own actual COVO mistakes, not just n-best alternatives.

Hotword-use-focused SFT on CB-Whisper train evidence, 2026-06-27:

- Motivation: instead of only telling COVO to preserve already-used hotwords, make the model more willing to use hotwords from CB-Whisper/KWS evidence when the reference contains them.
- Added script: `src/analysis/build_covo_hotword_use_sft.py`.
  - Starts from protected compact CB-Whisper train prompts.
  - Keeps the full train set.
  - Oversamples rows where prompt hotwords appear in the reference but are missing from ASR top-1.
  - Also includes rows where false prompt hotwords appear in ASR top-1, so the model does not learn to blindly insert every hotword.
  - Text matching uses OpenCC simplified normalization to avoid treating pure traditional/simplified differences as hotword-missing examples.
- Data built from `train_full_reliability_nbest6_compact_protect.qwen.jsonl`:
  - base rows: `17301`
  - true hotword-use-needed rows after OpenCC normalization: `414`
  - false-hotword-in-ASR rows: `2834`
  - repeats: hotword-use rows `5x`, false-hotword rows `1x`
  - output train file: `covo/data/processed/chinesehp_aishell1/train_full_reliability_nbest6_compact_protect_hotword_use_sft.jsonl`
  - total written rows: `22205`
- Training:
  - start adapter: `outputs/qwen35_cbwhisper_protect_sft40_from_ft80_bf16`
  - output adapter: `outputs/qwen35_cbwhisper_hotword_use_sft60_from_protect_bf16`
  - settings: `60` steps, lr `7e-7`, constant scheduler, bf16, batch `2`, grad accumulation `10`, max length `1280`.
  - train loss decreased from about `0.574` to `0.4309`; dev eval loss decreased from `0.4533` at step 30 to `0.3824` at step 60.
- Full 808 test:
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_hotword_use_sft60_full.jsonl`
  - audit summary: `src/logs/covo_error_audit_nbest6_hotword_use_sft60_full_summary.json`
  - COVO evaluator CER: `0.04323`
  - improved / worsened / unchanged: `317 / 46 / 445`
  - audit exact matches: `522`
  - hotword recall: `0.90658`
  - base-hit hotwords lost by prediction: `43`
  - false prompt hotwords in prediction: `11`
- Comparison:
  - old best: CER `0.04354`, hotword recall `0.90552`, exact `523`
  - protected SFT40: CER `0.04338`, hotword recall `0.90977`, exact `524`
  - hotword-use SFT60: CER `0.04323`, hotword recall `0.90658`, exact `522`
- Interpretation: this is now the lowest CER result, and it still slightly beats the old best on hotword recall, but it trades away some recall compared with protected SFT40. Treat it as the CER-priority main candidate; keep protected SFT40 as the better recall/CER balance candidate.

Synthetic hotword-use SFT expansion, 2026-06-27:

- Motivation: test whether the small gain from hotword-use SFT is caused by the model memorizing or underusing the limited CB-Whisper train supervision. If so, expanding the training set with more hotword-use examples should help.
- Added script: `src/analysis/build_covo_synthetic_hotword_use_sft.py`.
  - It reads full CB-Whisper train evidence.
  - It extracts synthetic positive hotwords from ASR top-1/reference diff spans.
  - It injects those spans as prompt/KWS hotwords and trains the assistant target to output the reference.
  - OpenCC simplified normalization is used before matching.
- Data:
  - synthetic diff-hotword rows: `8897`
  - combined train file: `covo/data/processed/chinesehp_aishell1/train_full_protect_hotword_use_plus_synthetic.qwen.jsonl`
  - combined rows: `31102`
- Training:
  - start adapter: `outputs/qwen35_cbwhisper_protect_sft40_from_ft80_bf16`
  - output adapter: `outputs/qwen35_cbwhisper_hotword_use_synth_sft80_bf16`
  - settings: `80` steps, lr `7e-7`, constant scheduler, bf16, batch `2`, grad accumulation `10`, max length `1280`.
  - train loss: `0.4831`
  - final dev eval loss: `0.3470`
- Full 808 test:
  - predictions: `src/logs/cbwhisper_covo_predictions_nbest6_hotword_use_synth_sft80_full.jsonl`
  - audit summary: `src/logs/covo_error_audit_nbest6_hotword_use_synth_sft80_full_summary.json`
  - COVO evaluator CER: `0.04315`
  - improved / worsened / unchanged: `319 / 46 / 443`
  - audit exact matches: `523`
  - hotword recall: `0.90764`
  - base-hit hotwords lost by prediction: `43`
  - false prompt hotwords in prediction: `11`
- Comparison:
  - old best: CER `0.04354`, hotword recall `0.90552`, exact `523`
  - protected SFT40: CER `0.04338`, hotword recall `0.90977`, exact `524`
  - hotword-use SFT60: CER `0.04323`, hotword recall `0.90658`, exact `522`
  - synthetic hotword-use SFT80: CER `0.04315`, hotword recall `0.90764`, exact `523`
- Interpretation: expanding the hotword-use SFT data gives a real but very small CER improvement and partially recovers recall compared with the non-synthetic hotword-use SFT. This means more data helps, but the bottleneck is not only memorization or train-set size. The likely remaining limits are candidate/evidence quality, ambiguity in synthetic diff spans, and COVO's tendency to prefer fluent common homophones unless the training examples are closer to its actual mistakes.

Actual-error hard SFT, 2026-06-28:

- Motivation: the synthetic expansion helped only slightly, so the next route used the model's own train-set failures rather than generic ASR/reference diffs. This avoids test leakage and gives training examples closer to COVO's real failure modes.
- Added script: `src/analysis/build_covo_actual_error_sft.py`.
  - It reads COVO prediction JSONL.
  - It classifies actual failures: worsened output, base-correct-broken, missed correction, n-best better than prediction, true hotword lost/missing, and false prompt hotword insertion.
  - It rewrites the assistant target to the reference while preserving the same bridge prompt.
  - It also samples correct/successful rows to avoid teaching the model to rewrite too aggressively.
- Train-side failure mining:
  - input messages: `covo/data/processed/chinesehp_aishell1/train_full_reliability_nbest6_compact_protect.qwen.jsonl`
  - inference adapter: `outputs/qwen35_cbwhisper_hotword_use_synth_sft80_bf16`
  - train predictions: `src/logs/cbwhisper_covo_predictions_train_full_protect_hotword_use_synth_sft80.jsonl`
  - samples: `17301`
  - train-side COVO evaluator CER: `0.03604`
  - improved / worsened / unchanged: `7569 / 292 / 9440`
- Strong hard SFT attempt:
  - hard rows: `16859`
  - combined train file: `train_full_protect_hotword_use_plus_synth_actual_hard.qwen.jsonl`, `47961` rows.
  - output adapter: `outputs/qwen35_cbwhisper_actual_hard_sft80_from_synth_bf16`
  - settings: start from synthetic SFT80, `80` steps, lr `5e-7`, bf16.
  - final dev eval loss: `0.3015`
  - full 808 CER: `0.04331`
  - improved / worsened / unchanged: `319 / 49 / 440`
  - hotword recall: `0.90658`
  - interpretation: rejected. The dev loss improved, but test CER and worsened count regressed; hard examples were too strong and made the model over-rewrite.
- Mild hard SFT attempt:
  - hard rows: `10288`
  - combined train file: `train_full_protect_hotword_use_plus_synth_actual_hard_mild.qwen.jsonl`, `41390` rows.
  - output adapter: `outputs/qwen35_cbwhisper_actual_hard_mild_sft40_from_synth_bf16`
  - settings: start from synthetic SFT80, `40` steps, lr `3e-7`, bf16.
  - final dev eval loss: `0.3265`
  - full-test predictions: `src/logs/cbwhisper_covo_predictions_nbest6_actual_hard_mild_sft40_full.jsonl`
  - audit summary: `src/logs/covo_error_audit_nbest6_actual_hard_mild_sft40_full_summary.json`
  - COVO evaluator CER: `0.04284`
  - improved / worsened / unchanged: `320 / 44 / 444`
  - audit exact matches: `523`
  - hotword recall: `0.90764`
  - base-hit hotwords lost by prediction: `43`
  - false prompt hotwords in prediction: `11`
- Comparison:
  - old best: CER `0.04354`, hotword recall `0.90552`, exact `523`
  - protected SFT40: CER `0.04338`, hotword recall `0.90977`, exact `524`
  - synthetic hotword-use SFT80: CER `0.04315`, hotword recall `0.90764`, exact `523`
  - actual-error strong SFT80: CER `0.04331`, hotword recall `0.90658`, exact `523`
  - actual-error mild SFT40: CER `0.04284`, hotword recall `0.90764`, exact `523`
- Interpretation: mild actual-error SFT is the new CER-best route. It does not raise recall beyond the synthetic SFT80 result, but it lowers CER and reduces worsened samples. The key lesson is that actual model mistakes help, but the hard-data weight must be small and mixed with preservation examples; stronger hard training improves dev loss while hurting test behavior.

Expanded AISHELL/COVO training amount probe, 2026-06-28:

- Motivation: test the user's hypothesis that the model may need more training data, including other AISHELL-style rewrite data, rather than only CB-Whisper train evidence.
- Data construction:
  - base current best train mix: `train_full_protect_hotword_use_plus_synth_actual_hard_mild.qwen.jsonl`, `41390` rows.
  - AISHELL hotword noop preservation: `aishell_train_hotword_noop.qwen.jsonl`, `17301` rows.
  - sampled general ASR rewrite data: `train_text_rewrite_hardneg_dropout.qwen.jsonl`, `60000` sampled rows.
  - combined file: `train_full_protect_actual_hard_mild_plus_rewrite60k_noop.qwen.jsonl`, `118691` rows.
  - `train.qwen_messages.jsonl` was intentionally not mixed because it uses an edits-output schema, while the current COVO route uses `{"text": ...}` output.
- Expanded training:
  - start adapter: `outputs/qwen35_cbwhisper_actual_hard_mild_sft40_from_synth_bf16`
  - output adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - settings: `120` steps, lr `2e-7`, bf16.
  - final dev eval loss: `0.3236`
  - full 808 COVO evaluator CER: `0.04292`
  - improved / worsened / unchanged: `320 / 43 / 445`
  - audit exact matches: `524`
  - hotword recall: `0.90870`
  - base-hit hotwords lost by prediction: `42`
- Mild callback after expanded training:
  - start adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - train file: `train_full_protect_hotword_use_plus_synth_actual_hard_mild.qwen.jsonl`
  - output adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_then_mild_sft30_bf16`
  - settings: `30` steps, lr `1e-7`, bf16.
  - final dev eval loss: `0.3196`
  - full 808 COVO evaluator CER: `0.04323`
  - improved / worsened / unchanged: `319 / 47 / 442`
  - hotword recall: `0.90658`
- Comparison:
  - current CER-best actual-error mild SFT40: CER `0.04284`, hotword recall `0.90764`, exact `523`, base-hit hotwords lost `43`.
  - expanded rewrite60k+noop SFT120: CER `0.04292`, hotword recall `0.90870`, exact `524`, base-hit hotwords lost `42`.
  - expanded then mild callback: CER `0.04323`, hotword recall `0.90658`.
- Interpretation: larger AISHELL/COVO training improves hotword recall, exact matches, and preservation slightly, but it does not beat the current CER-best model. A small callback on mild hard data does not recover the CER; it over-specializes and regresses. Keep the expanded model as a recall-priority alternate, but do not replace the current main CER-best adapter.

Recall-priority probes, 2026-06-28:

- Motivation: after the expanded model improved hotword recall to `0.90870`, test whether recall can be pushed further without a large CER regression.
- Recall-focused SFT:
  - data: `train_hotword_recall_focus_plus_noop_use.qwen.jsonl`, `54830` rows.
  - composition: actual train-side hotword-lost/missing failures, AISHELL hotword noop preservation, and hotword-use SFT rows.
  - start adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - output adapter: `outputs/qwen35_cbwhisper_recall_focus_sft60_from_expanded_bf16`
  - settings: `60` steps, lr `1e-7`, bf16.
  - final dev eval loss: `0.3174`
  - full 808 CER: `0.04307`
  - hotword recall: `0.90764`
  - exact matches: `524`
  - interpretation: rejected. It improved some correction counts but did not beat the expanded model on recall and hurt CER.
- More hotword evidence at inference:
  - model: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - setting: `--max-hotwords 8 --max-prompt-hotwords 6`, `--hotword-source all`
  - full 808 CER: `0.04292`
  - hotword recall: `0.90870`
  - exact matches: `526`
  - base-hit hotwords lost: `42`
  - false prompt hotwords in prediction: `12`
  - interpretation: recall did not improve beyond expanded all-hotword decoding, but exact matches improved. This is a useful exact-match variant, not a recall breakthrough.
- Prompt-only hotword evidence:
  - model: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - setting: `--hotword-source prompt`
  - full 808 CER: `0.04284`
  - hotword recall: `0.89915`
  - exact matches: `529`
  - interpretation: rejected for recall. It gives strong exact/CER behavior but drops hotword recall, so prompt-only evidence is too narrow for the current recall goal.
- Current recall-best:
  - `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - all-hotword setting: CER `0.04292`, hotword recall `0.90870`, exact `524`.
  - hw8/prompt6 all-hotword setting: CER `0.04292`, hotword recall `0.90870`, exact `526`.
- Interpretation: recall now appears limited by the available evidence/candidate pool rather than simply by COVO training. The n-best oracle recall in the same audit is `0.91720`, so there is only about `0.0085` absolute recall headroom left from final-text correction under the current evidence. Further recall gain likely requires better CB-Whisper candidate/evidence generation or a carefully justified hotword-preservation mechanism, not simply more COVO SFT.

CB-Whisper-side targeted context supplement, 2026-06-28:

- Motivation: after COVO SFT/recall-focused training saturated, inspect whether true hotwords are present in KWS/context but absent from the Whisper candidate pool.
- Diagnostic on current pool `cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`:
  - 942 true hotword mentions.
  - 921/942 are present in KWS/context top8.
  - 892/942 are present in current n-best10.
  - 36 mentions are in KWS/context top8 but absent from n-best10, so CB-Whisper candidate generation still has recoverable recall.
  - top-k oracle curve: top1 recall `0.93631`, top2 `0.94268`, top6 `0.94480`, top10 `0.94586`; top10 oracle CER `0.03019`.
- Added script support in `src/analysis/targeted_supplement_covo_nbest.py`:
  - `--target-missing-context`: supplement rows whose context hotword is missing from current candidates.
  - `--require-single-prompt-target`: conservative mode; only supplement rows with exactly one prompt hotword and that hotword absent from current n-best.
  - Targeted prompt variants are added before the previous generic prompt variants.
  - Generated target-hit candidates are inserted after ASR top-1, selected by minimal edit distance to top1, and still passed through the existing quality cleaner.
- Conservative full run:
  - input: `cbwhisper_candidate_pool_v3_nbest10_targeted_supplement_round2_clean_strict_slack5.jsonl`
  - output: `cbwhisper_candidate_pool_v3_context_target_singleprompt.jsonl`
  - processed rows: `27`
  - changed rows: `10`
  - candidate-pool oracle recall: `0.94586 -> 0.95541`
  - candidate-pool oracle CER: `0.03019 -> 0.02957`
  - exact reference in n-best: `616 -> 620`
- COVO full 808 validation with recall-priority expanded adapter:
  - adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - input pool: `cbwhisper_candidate_pool_v3_context_target_singleprompt.jsonl`
  - setting: nbest6, all hotwords, protected supported hotwords.
  - CER: `0.04284`
  - improved / worsened / unchanged: `320 / 42 / 446`
  - audit exact matches: `527`
  - hotword recall: `0.91083`
  - n-best oracle recall in audit: `0.92463`
- Comparison:
  - previous CER-best actual-error mild SFT40: CER `0.04284`, hotword recall `0.90764`, exact `523`.
  - previous recall-priority expanded adapter: CER `0.04292`, hotword recall `0.90870`, exact `524`.
  - CB-side targeted supplement + expanded adapter: CER `0.04284`, hotword recall `0.91083`, exact `527`.
- Interpretation: this is the current best combined AISHELL result. The useful gain comes from candidate generation/evidence, not COVO training. The route is promising but must stay conservative: a broader target-missing-context smoke test produced good candidates such as `姊弟恋`, but also false hotword candidates like `德郭队`; therefore keep `--require-single-prompt-target` as the clean setting unless a better confidence/competition filter is added.

Lost-hotword audit after targeted supplement, 2026-06-28:

- Prediction file: `src/logs/cbwhisper_covo_predictions_nbest6_context_target_singleprompt.jsonl`
- Audit files:
  - `src/logs/covo_lost_hotword_audit_context_target_singleprompt_summary.json`
  - `src/logs/covo_lost_hotword_audit_context_target_singleprompt.csv`
- Final recall state:
  - total mentions: `942`
  - prediction hits: `858`
  - lost hotwords: `84`
  - recall: `0.91083`
- Lost-hotword categories:
  - `covo_lost_base_hotword`: `42`
    - The hotword is already present in ASR top1/base, but COVO rewrites it to a more common homophone or fluent form.
    - Examples: `今久 -> 金九`, `杨锋 -> 杨峰`, `宋芳 -> 颂芳`.
    - Important: the prompt already contains protected-hotword instructions and the protected list for cases like `今久整合营销集团,今久`, but the model still disobeys. This is now the largest recall loss source.
  - `candidate_generation_missing`: `26`
    - KWS/context contains the true hotword, but CB-Whisper/Whisper n-best still lacks a clean candidate containing it.
    - Examples: `许玮甯/玮甯`, `马特里亚斯`, `欧帕拉迪尼`, `丰台区域潘家村`.
    - This remains the main CB-Whisper-side candidate generation target.
  - `covo_failed_with_nbest6`: `6`
    - The needed hotword appears in the n-best6 evidence, but COVO does not choose or use it.
    - Examples: `姊弟恋`, `佟健`, `弗菜戈`, `海珠湖公园`.
  - `kws_context_missing`: `10`
    - The true hotword is not present in the KWS/context evidence at all, so CB/COVO cannot recover it without improving KWS or adding an external lexicon/oracle-like source.
    - Examples: `朱圣祎`, `苹果`, `银联`, `龟趺`.
- Interpretation:
  - The next biggest recall gain is not only CB candidate generation. About half of the remaining lost hotwords are caused by COVO rewriting protected/base-present hotwords into common homophones.
  - A pure prompt instruction is not enough; the model has already ignored explicit protected-hotword text.
  - Two promising next routes:
    1. Train COVO on protected-hotword preservation failures mined from train predictions, with examples where the output must keep base/prompt hotwords exactly while still fixing other characters.
    2. Continue CB-side targeted supplement for the `candidate_generation_missing` bucket, but add stronger cleanliness filters to avoid false KWS words such as `德郭队`.

Protected-hotword preservation training probe, 2026-06-28:

- Motivation: the user correctly pointed out that the model should learn not to rewrite protected hotwords, rather than relying only on prompt instructions or post-hoc constraints.
- Added scripts:
  - `src/analysis/build_covo_protected_preserve_pairs.py`
    - Mines train-only COVO predictions where ASR top-1 contains a true prompt hotword but the model prediction loses it.
    - Builds DPO pairs: chosen is the reference, rejected is the model prediction that rewrote the protected hotword.
  - `src/analysis/build_covo_protected_preserve_sft.py`
    - Converts those pairs to repeated Qwen-message SFT rows.
- Train-only mined data:
  - source predictions: `src/logs/cbwhisper_covo_predictions_train_full_protect_hotword_use_synth_sft80.jsonl`
  - strict pairs: `64`
  - loose pairs: `67`
  - repeated SFT rows with x20: `1280`
- DPO probes from expanded adapter:
  - start adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - low-lr DPO: lr `5e-7`, beta `0.05`, 20/40 steps observed, preference accuracy stayed `0.0`; stopped early.
  - stronger DPO: lr `2e-6`, beta `0.1`, 15/40 steps observed, preference accuracy stayed `0.0`; stopped early.
  - interpretation: actual protected-preserve pairs are too few / too hard for this lightweight DPO setup; DPO did not learn.
- Targeted SFT x20 from expanded adapter:
  - start adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - data: `train_actual_protected_preserve_sft_x20.qwen.jsonl`, `1280` rows
  - 40-step output adapter: `outputs/qwen35_cbwhisper_actual_protected_preserve_sft40_x20_from_expanded_bf16`
  - training loss stayed around `0.38` and ended at `0.3723`.
  - full AISHELL 808 with current CB candidate pool:
    - CER: `0.04292`
    - improved / worsened / unchanged: `318 / 40 / 450`
    - exact matches: `527`
    - hotword recall: `0.91295`
    - base-hit hotwords lost by prediction: `40`
    - base-correct-broken: `28`
  - 20-step variant:
    - CER: `0.04307`
    - hotword recall: `0.91189`
    - base-hit hotwords lost by prediction: `42`
    - base-correct-broken: `30`
    - interpretation: worse than 40-step, rejected.
- Comparison with previous best combined result:
  - previous CB-targeted + expanded adapter: CER `0.04284`, recall `0.91083`, base-hit lost `42`, base-correct-broken `30`.
  - protected-preserve SFT40: CER `0.04292`, recall `0.91295`, base-hit lost `40`, base-correct-broken `28`.
- Interpretation:
  - The model can be taught not to rewrite protected/base-present hotwords: recall improved by `+0.00212`, and both base-hit lost hotwords and base-correct-broken counts dropped by `2`.
  - The current targeted SFT also costs one extra edit overall (`552 -> 553`), so it is recall-best but not CER-best.
  - This route is no longer "no signal"; it works, but the data is too small and slightly over-specialized. The next version should enlarge train-only protected-preserve cases, preferably by running current best model on train or generating controlled homophone-preservation negatives, then mix with enough general correction/no-op rows to keep CER from drifting.

Protected-hotword preservation follow-up, 2026-06-28:

- Goal: test whether the model can learn protected-hotword preservation without a hard decoding gate.
- Mixed preserve + anchor SFT:
  - script: `src/analysis/build_covo_preserve_anchor_mix.py`
  - data: `train_protected_preserve_x40_plus_anchor6k.qwen.jsonl`
  - composition: `1280` protected-preserve rows repeated twice plus `6000` general anchor rows.
  - training: `80` steps from `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`, lr `3e-7`.
  - output adapter: `outputs/qwen35_cbwhisper_protected_preserve_x40_anchor6k_sft80_from_expanded_bf16`
  - full AISHELL 808 result:
    - CER: `0.04354`
    - exact matches: `524`
    - hotword recall: `0.90870`
    - base-hit hotwords lost by prediction: `44`
    - base-correct-broken: `29`
  - interpretation: rejected. The large generic anchor mix diluted the preserve signal and worsened both CER and recall.
- Protected-positive SFT:
  - script: `src/analysis/build_covo_protected_positive_sft.py`
  - data: `train_protected_positive6k_lostx40_anchor2k.qwen.jsonl`
  - composition: `6000` train rows where ASR top-1 and reference both contain a true prompt hotword, `76` actual lost-protected rows repeated `40` times, plus `2000` anchor rows.
  - target text is normalized/simplified reference text, not raw ASR no-op, to avoid teaching繁体 ASR output.
  - training: `100` steps from `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`, lr `3e-7`.
  - output adapter: `outputs/qwen35_cbwhisper_protected_positive6k_lostx40_anchor2k_sft100_from_expanded_bf16`
  - full AISHELL 808 result:
    - CER: `0.04331`
    - exact matches: `526`
    - hotword recall: `0.91083`
    - base-hit hotwords lost by prediction: `42`
    - base-correct-broken: `28`
  - interpretation: rejected as main result. It recovers the previous best recall but costs `+6` edits versus the current combined best (`0.04284`). The model is learning protection, but the added positive-reference training weakens general correction quality.
- Current decision:
  - Keep the current main result as CB-side targeted supplement + expanded adapter: CER `0.04284`, recall `0.91083`, exact `527`.
  - Keep protected-preserve SFT40 as recall-best only: CER `0.04292`, recall `0.91295`.
  - Do not use the two follow-up adapters as main results.
  - Next useful direction is not simply "more preserve SFT"; it needs a training objective that preserves hotwords while maintaining the existing correction distribution, e.g. harder contrastive examples or a small span-level preference/calibration module.

Shuili transfer test, 2026-06-29:

- Dataset and endpoint:
  - dataset: `datasets/shuili/data_shuil_largev3`
  - CB-Whisper runner: `src/run_cbwhisper_shuili_v3_kws_test.py`
  - KWS checkpoint: `src/outputs/aishell_large_v3_kws_true/checkpoints/f1G/f1G-epoch=11-step=72060.ckpt`
  - evidence output: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl` (`1152` rows; not committed because it is large)
  - metrics CSV: `src/logs/test_metrics_shuili_v3_current_rerun_20260629.csv`
- CB-Whisper-only result:
  - samples: `1152`
  - Entity Recall: `0.87155`
  - CER: `0.14022`
  - Hotword Sentence CER: `0.13963`
  - Hotword Only CER: `0.39557`
  - WER: `0.76128`
  - n-best diagnostic aggregate: average n-best size `10.41`; top1 CER `0.14059`; oracle CER `0.04668`; oracle beats top1 on `0.18016` of rows.
- COVO with current main expanded adapter:
  - adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - raw COVO eval, without opencc simplification: base CER `0.15415` -> prediction CER `0.13313`
  - opencc-aligned CER: base `0.13917` -> prediction `0.13142`
  - opencc edit counts: `2192 -> 2070`
  - opencc exact matches: `275 -> 287`
  - opencc improved / worsened / unchanged: `123 / 66 / 963`
  - hotword recall in audit: `0.90154 -> 0.92231`
  - base-hit hotwords lost by prediction: `4`
- COVO with protected-preserve SFT40 adapter:
  - adapter: `outputs/qwen35_cbwhisper_actual_protected_preserve_sft40_x20_from_expanded_bf16`
  - raw COVO eval: base CER `0.15415` -> prediction CER `0.13332`
  - opencc-aligned CER: base `0.13917` -> prediction `0.13155`
  - hotword recall in audit: `0.90154 -> 0.92231`
  - base-hit hotwords lost by prediction: `4`
- Interpretation:
  - Shuili transfer is much harder than AISHELL in absolute CER, but COVO is clearly useful on this dataset: the current main expanded adapter reduces opencc-aligned CER by about `0.00775` absolute and improves hotword recall by about `0.02078`.
  - The protected-preserve SFT40 adapter does not help on Shuili; it has identical hotword recall and slightly worse CER than the expanded adapter.
  - The n-best oracle CER (`0.04668`) is far below both CB-only and COVO output, so the main remaining opportunity on Shuili is better candidate selection/correction from the existing candidate pool rather than only more KWS recall.
  - Use expanded SFT120 as the current Shuili COVO adapter among tested options.

Shuili COVO error analysis, 2026-06-29:

- Question: why does COVO reduce Shuili CER by only about one percentage point despite a strong n-best oracle?
- OpenCC-aligned accounting for `expanded_sft120`:
  - base CER: `0.13917` (`2192` edits)
  - COVO CER: `0.13142` (`2070` edits)
  - n-best oracle CER from the same prediction file: `0.05200` (`819` edits)
  - COVO gain: `122` edits
  - oracle available gain: `1373` edits
  - realized oracle gain ratio: `0.0889`
- Error categories under OpenCC-aligned audit:
  - `unchanged_error`: `713` rows, `1695` prediction edits.
    - `599/713` have a better n-best candidate than COVO output.
    - `416/713` have an exact-reference candidate in n-best, but COVO keeps/normalizes top1 instead.
    - This is the dominant failure mode.
  - `improved_partial`: `86` rows.
  - `fixed_to_exact`: `37` rows.
  - `worsened_error`: `41` rows.
  - `base_correct_broken`: `25` rows.
- Candidate-pool facts:
  - n-best exact rows: `760/1152`.
  - n-best better than COVO rows: `711/1152`.
  - best-candidate rank distribution is not always rank1: rank1 `420`, rank2 `291`, rank3 `171`, rank4 `94`, rank5 `66`, rank6 `47`, rank7 `34`, rank8 `29`.
  - Therefore the candidate pool is useful, but the COVO adapter is not reliably selecting non-top1 candidates.
- Linguistic error pattern:
  - The most frequent missing reference character in COVO output is `呢` (`607` missing edits), followed by `啊` (`84`), `的` (`83`), `个` (`78`), `这` (`71`), `一` (`50`).
  - Shuili references preserve lecture-style fillers and discourse particles such as `呢/啊/那么/这一个/咱们`.
  - The current COVO training is biased toward clean ASR correction and minimal edits, so it often refuses to insert these fillers even when an n-best candidate contains them.
  - It can also delete them from already-correct base text, e.g. `咱们下面呢进入...` -> `咱们下面进入...`.
- Hotword evidence pattern:
  - COVO is not mainly failing by losing hotwords on Shuili. Hotword recall improves from `0.90154` to `0.92231`, and only `4` base-hit hotwords are lost.
  - However, prompts are noisy: false prompt hotwords are extremely common, especially `闸门` (`630` rows where it appears in prompt but not reference), `石方` (`221`), `地基` (`191`), `基坑` (`103`).
  - This explains why the model cannot simply trust prompt hotwords and tends to stay conservative.
- Interpretation:
  - The one-point CER reduction is not caused by lack of n-best diversity; the candidate pool contains many exact answers.
  - It is mainly a COVO selection/style mismatch: the model was trained to be conservative and clean, while Shuili scoring rewards recovering lecture fillers and longer spoken-form candidates.
  - Next promising route for Shuili is a candidate-selection/reranking or SFT objective that explicitly teaches choosing the best n-best candidate, including filler-preserving lecture transcripts, instead of only generic ASR correction.

Shuili filler-normalized CER diagnostic, 2026-06-29:

- Added script: `src/analysis/evaluate_filler_normalized_cer.py`
  - Applies OpenCC `t2s`, removes punctuation/spaces, then optionally removes a configurable filler list before CER.
  - This is diagnostic only; raw CER remains reported.
- Minimal filler list:
  - `呢, 啊, 呃, 嗯`
  - expanded adapter:
    - base CER: `0.09900`
    - COVO CER: `0.08965`
    - base edits: `1472`
    - COVO edits: `1333`
    - exact matches: `555 -> 599`
  - protected-preserve SFT40:
    - COVO CER: `0.08978`
    - exact matches: `597`
- Extended lecture filler list:
  - `这一个, 那么, 的话, 这个, 那个, 咱们, 我们呢, 就是, 呢, 啊, 呃, 嗯`
  - expanded adapter:
    - base CER: `0.09814`
    - COVO CER: `0.08903`
    - base edits: `1250`
    - COVO edits: `1134`
    - exact matches: `615 -> 650`
  - protected-preserve SFT40:
    - COVO CER: `0.08919`
    - exact matches: `648`
- Interpretation:
  - The large raw/opencc CER (`~0.139`) is heavily inflated by lecture fillers, especially `呢`.
  - Even the conservative four-token filler list lowers the base CER to about `0.099`, confirming that the issue is largely evaluation style rather than hotword modeling.
  - COVO still improves over base after filler normalization (`0.09900 -> 0.08965` minimal; `0.09814 -> 0.08903` extended).
  - Expanded SFT120 remains the better Shuili adapter under both filler-normalized metrics.

Shuili pre-strip fillers then COVO, 2026-06-29:

- User hypothesis: remove filler tokens from the evidence first, then let COVO correct the cleaned text.
- Added script: `src/analysis/strip_covo_fillers.py`
  - Applies OpenCC `t2s`, removes configured filler tokens from `reference`, `input.asr_top1`, and `input.nbest`.
  - Default strip list: `这一个, 那么, 的话, 这个, 那个, 我们呢, 就是, 呢, 啊, 呃, 嗯`.
  - Did not remove `咱们` by default because it is more like a pronoun than a pure filler and may change the transcript subject.
- Experiment:
  - stripped evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_filler_stripped_20260629.jsonl`
  - COVO adapter: `outputs/qwen35_cbwhisper_expanded_rewrite60k_noop_sft120_from_mild_bf16`
  - prediction file: `src/logs/cbwhisper_covo_predictions_shuili_v3_filler_stripped_expanded_sft120_20260629.jsonl`
- Result after stripping before COVO:
  - samples: `1149`
  - base CER: `0.09603`
  - COVO CER: `0.08976`
  - base edits: `1271`
  - COVO edits: `1188`
  - improved / worsened / unchanged: `114 / 82 / 953`
  - hotword recall: `0.91147 -> 0.92412`
  - base-hit hotwords lost by prediction: `2`
  - n-best oracle CER: `0.04065`
- Comparison:
  - original evidence + COVO + filler-normalized scoring had expanded-adapter CER `0.08903`.
  - pre-strip evidence + COVO has CER `0.08976`.
- Interpretation:
  - Pre-stripping fillers is not better than leaving original evidence intact and applying filler-normalized scoring afterward.
  - It cleans the target but also changes n-best competition and removes context that COVO can use.
  - Keep this as a rejected route; for reporting, prefer raw CER plus filler-normalized CER rather than modifying COVO input.

Shuili COVO n-best selector SFT, 2026-06-29:

- Constraint:
  - Shuili has no reliable train split, so all COVO capability training in this round uses AISHELL train evidence only.
  - Shuili is used only as the held-out transfer/evaluation set.
- Motivation:
  - Prior Shuili analysis showed the n-best pool has strong oracle quality, but COVO often keeps CB-Whisper/top1.
  - Therefore this round trains COVO to select the best n-best candidate rather than directly rewrite to the reference.
- Added script:
  - `src/analysis/build_covo_nbest_selector_sft.py`
  - It reads AISHELL train CB-Whisper/COVO evidence, finds the n-best candidate with minimum CER to the reference, and writes Qwen-style SFT messages.
  - Hard rows where oracle n-best improves top1 are repeated more often; exact-oracle rows get extra weight.
- Added bridge option:
  - `src/analysis/cbwhisper_covo_bridge.py --prompt-mode selector`
  - Selector prompt asks the model to choose or minimally merge n-best candidates, instead of default conservative correction.
- Training data:
  - Source evidence: `src/logs/cbwhisper_covo_evidence_train_full.jsonl`
  - Rows scanned: `17301`
  - Hard rows: `6609`
  - Exact hard rows: `4188`
  - Base/keep rows: `6000`
  - Written SFT rows after repetition: `40812`
- Experiment A: selector target, conservative/default prompt.
  - Adapter: `outputs/qwen35_cbwhisper_nbest_selector_hardx4_sft160_from_expanded_bf16`
  - Raw Shuili CER: `0.13275` vs expanded SFT120 `0.13313`
  - Extended filler-normalized CER: `0.08903`
  - Minimal filler-normalized CER: `0.08938`
  - Hotword recall: `0.92231`
  - Interpretation: only tiny gain because train target and inference prompt conflicted; prompt still emphasized preserving CB-Whisper output.
- Experiment B: selector target plus selector prompt.
  - Adapter: `outputs/qwen35_cbwhisper_nbest_selector_prompt_hardx4_sft160_from_expanded_bf16`
  - Raw Shuili CER: `0.13199` vs base/top1 `0.15415`
  - Extended filler-normalized CER: `0.08778` vs expanded SFT120 `0.08903`
  - Minimal filler-normalized CER: `0.08844` vs expanded SFT120 `0.08965`
  - Hotword recall: `0.92322` vs base/top1 `0.90154`
  - Exact matches under minimal filler normalization: `555 -> 605`
- Selector-prompt audit:
  - n-best oracle CER is still much lower: `0.06488`.
  - n-best exact rows: `688/1152` in the audit script, while prediction exact rows are only `287/1152`.
  - n-best better than prediction: `664/1152`.
  - COVO still copies top1 very often: `918/1152`.
  - It copies the best n-best candidate only `407/1152`; most of those are still rank1.
- Conclusion:
  - AISHELL-trained selector SFT transfers positively to Shuili and is the best Shuili COVO result so far under filler-normalized CER.
  - The improvement is real but small because the model is still too conservative and has not learned to trust non-top1 n-best candidates enough.
  - Next routes should increase explicit non-top1 selection pressure, for example more rank>1 oracle rows, contrastive/DPO pairs between top1 and oracle n-best, or inference-time n-best selection with a learned lightweight scorer.

Shuili COVO strong selector full-epoch SFT, 2026-06-30:

- User correction:
  - For established routes, do full training by default instead of judging from very short 100-step runs.
- Prompt/training changes:
  - Strengthened selector system message and instruction so the model is an `ASR N-best candidate selector`, not a conservative post-corrector.
  - The prompt explicitly says ASR top-1 is only one candidate and should not be copied by default.
  - Rebuilt AISHELL train selector SFT data with the stronger selector system prompt.
  - Continued from `outputs/qwen35_cbwhisper_nbest_selector_prompt_hardx4_sft160_from_expanded_bf16`.
  - Learning rate: `1e-6`, constant.
  - Full training: `1 epoch`, `10203/10203` optimizer steps.
  - Final training loss: `0.2769`.
  - Output adapter: `outputs/qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`.
- Shuili validation:
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`
  - Prompt mode: `selector`
  - Samples: `1152`
  - Raw CER:
    - base/top1: `0.15415`
    - strong selector: `0.11936`
    - previous selector-prompt SFT160: `0.13199`
  - Extended filler-normalized CER:
    - base/top1: `0.09814`
    - strong selector: `0.08566`
    - previous selector-prompt SFT160: `0.08778`
  - Minimal filler-normalized CER (`呢,啊,呃,嗯`):
    - base/top1: `0.09900`
    - strong selector: `0.08003`
    - previous selector-prompt SFT160: `0.08844`
  - Exact matches under minimal filler normalization:
    - base/top1: `555`
    - strong selector: `607`
- N-best utilization:
  - copy top1: `918 -> 594` compared with selector-prompt SFT160.
  - copy any n-best: `1080 -> 1127`.
  - copy oracle-best n-best: `407 -> 423`.
  - prediction exact rows: `289 -> 322`.
  - prediction better than base: `134 -> 298`.
  - prediction worse than base: `68 -> 187`.
- Hotword tradeoff:
  - Hotword recall drops:
    - base/top1: `0.90154`
    - previous selector-prompt SFT160: `0.92322`
    - strong selector full epoch: `0.88708`
  - base-hit hotwords lost by prediction increases to `35`.
  - This means full-epoch strong selector finally learned to use n-best and lowers CER substantially, but it over-corrects and weakens hotword preservation.
- Interpretation:
  - The earlier bottleneck really was COVO's over-conservative top1 copying.
  - Full training plus stronger selector prompt fixes much of that behavior and gives the best Shuili CER so far.
  - The next problem is balancing n-best selection with protected hotword retention. A likely next route is a mixed objective: keep strong selector rows, but add protected-hotword preservation pairs/rows so non-top1 selection does not delete supported domain terms.

AISHELL validation of strong selector full-epoch SFT, 2026-06-30:

- Purpose:
  - Check whether the Shuili strong selector adapter also improves the AISHELL 808 hotword test.
  - This uses the rich clean n-best evidence, not the older plain full-test evidence.
- Input/evidence:
  - `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean.jsonl`
  - Candidate-pool summary before COVO:
    - rows: `808`
    - avg unique n-best: `8.7859`
    - exact reference in n-best: `612`
    - summary top1 corpus CER: `0.07238`
    - summary oracle corpus CER: `0.02864`
    - summary top1 hotword recall: `0.93631`
    - summary oracle hotword recall: `0.94480`
- Adapter/eval setting:
  - Adapter: `outputs/qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`
  - Prompt mode: `selector`
  - Same bridge settings as Shuili validation: n-best max `6`, all hotwords, protected supported hotwords.
- COVO evaluator result:
  - base/top1 CER: `0.10446`
  - prediction CER: `0.07055`
  - improved / worsened / unchanged: `264 / 104 / 440`
- Audit result:
  - base CER: `0.10446`
  - prediction CER: `0.07055`
  - n-best oracle CER: `0.05355`
  - prediction-or-nbest oracle CER: `0.04323`
  - base exact: `369`
  - prediction exact: `436`
  - n-best exact: `513`
  - n-best better than prediction: `199`
- N-best utilization:
  - copy top1: `410/808`
  - copy any n-best: `759/808`
  - copy oracle-best n-best: `525/808`
  - prediction better than base: `231`
  - prediction worse than base: `118`
- Hotword tradeoff:
  - base recall: `0.91507`
  - prediction recall: `0.87261`
  - n-best oracle recall: `0.91614`
  - base-hit hotwords lost by prediction: `62`
- Comparison to AISHELL main result:
  - Current AISHELL main combined result remains CB-side targeted supplement + expanded adapter:
    - CER: `0.04284`
    - hotword recall: `0.91083`
    - exact: `527`
  - Strong selector full epoch is much worse on AISHELL (`0.07055` CER and `0.87261` recall).
- Interpretation:
  - The strong selector transfer helps Shuili because Shuili's conservative COVO behavior was the main bottleneck.
  - On AISHELL, the existing main pipeline is already well calibrated; aggressive n-best selection over-corrects and loses too many hotwords.
  - Do not promote the strong selector adapter for AISHELL. Keep it as a Shuili-oriented CER-reduction branch and use AISHELL main result for paper-level AISHELL comparison unless a mixed hotword-preserving selector recovers recall.

Detailed Shuili error analysis for strong selector full-epoch SFT, 2026-06-30:

- Overall behavior:
  - Raw edits: base `2192` -> prediction `1880`, net gain `312` edits.
  - Minimal filler-normalized edits: base `1472` -> prediction `1190`, net gain `282` edits.
  - Extended filler-normalized edits: base `1252` -> prediction `1093`, net gain `159` edits.
  - Raw sample movement: improved `298`, worsened `187`, unchanged `667`.
  - Minimal filler-normalized movement: improved `251`, worsened `171`, unchanged `730`.
  - Extended filler-normalized movement: improved `205`, worsened `160`, unchanged `787`.
- Why raw CER improvement is smaller than hoped:
  - The model now uses n-best much more, but it also breaks many already-good sentences.
  - Raw worsened rows: `187`.
  - Among worsened rows, `77` had base distance `0`, and `151` had base distance `<=2`; many regressions are on already nearly-correct top1 outputs.
  - `27` raw regressions disappear under minimal filler normalization, and `43` disappear under extended filler normalization, so some regressions are just deleting lecture fillers such as `呢`.
  - But `160` regressions remain even after minimal filler normalization, so over-correction is a real issue, not only evaluation style.
- Hotword/content regressions:
  - Worsened rows with lost prompt/reference hotword: `21`.
  - Frequent lost hotwords: `水利`, `施工`, `运输`, `施工过程`, `水利工程`, `浇筑`.
  - Examples:
    - `地基处理方法来解决` -> `立即处理方法来解决`: loses `地基/地基处理`.
    - `水闸的施工...` -> `水灾的时空...`: loses `水闸`.
    - `水利工程的` -> `水滴工程的`: loses `水利工程`.
  - This explains the recall drop from `0.92322` in the short selector to `0.88708` in full strong selector.
- Filler/style pattern:
  - Most frequent missing reference characters in prediction remain `呢` (`641`), `的` (`93`), `啊` (`86`), `这` (`50`), `个` (`49`).
  - Many base-correct examples are worsened by deleting lecture particles:
    - `咱们下面呢进入...` -> `咱们下面进入...`
    - `我们目前呢...` -> `我们目前...`
    - `组织呢` -> `组织`
  - This means Shuili scoring rewards preserving spoken lecture style, while the model has a strong normalization/cleanup bias.
- Remaining oracle space:
  - n-best oracle CER is `0.06488`, while strong selector CER is `0.11936`.
  - n-best better than prediction: `630/1152`.
  - n-best exact rows: `688/1152`, while prediction exact rows are only `322/1152`.
  - Remaining high-gap examples often have an oracle candidate that preserves omitted lecture context:
    - ref `那么以及呢咱们的这一个呢相关的质量检查`, base/pred omit the beginning and fillers, oracle is exact.
    - ref `水利工程咱们的水利工程啊`, base/pred output only `水利工程`, oracle contains the longer phrase.
- Conclusion:
  - Full strong selector proves that the model can be moved away from top1 copying, but the current objective is unbalanced.
  - Next route should not simply train longer or make the prompt even stronger.
  - The right next route is mixed training: keep the selector objective, add explicit no-break rows for near-correct top1, and add protected-hotword preservation examples/pairs so the model does not delete domain terms while selecting non-top1 candidates.

Shuili content-priority selector prompt-only test, 2026-06-30:

- Motivation:
  - User asked whether the first problem, over-weighting fillers/discourse particles, can be improved by prompt engineering.
  - Test is prompt-only: same full-epoch strong selector adapter, no retraining.
- Code change:
  - Added `--prompt-mode selector_content` in `src/analysis/cbwhisper_covo_bridge.py`.
  - The new prompt treats fillers such as `呢/啊/嗯/呃/那么/这个/这一个/的话/就是` as low-priority evidence.
  - It prioritizes domain terms, named entities, numbers, and main semantics over filler completeness.
  - It explicitly warns not to change correct domain terms just to add/delete fillers.
- Adapter/evidence:
  - Adapter: `outputs/qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`
- Raw CER:
  - base/top1: `0.15415`
  - strong selector prompt: `0.11936`
  - content-priority selector prompt: `0.11428`
- Filler-normalized CER:
  - minimal fillers (`呢,啊,呃,嗯`): `0.08003 -> 0.07472` compared with strong selector.
  - extended fillers: `0.08566 -> 0.07945` compared with strong selector.
- Error movement:
  - strong selector improved/worsened/unchanged: `349 / 172 / 631` in the COVO stdout summary.
  - content-priority prompt improved/worsened/unchanged: `320 / 104 / 728`.
  - Audit base-correct-broken drops: `73 -> 44`.
  - Audit worsened-error drops: `99 -> 60`.
- N-best behavior:
  - copy top1: `594 -> 717`.
  - copy any n-best: `1127 -> 1130`.
  - copy oracle-best n-best: `423 -> 459`.
  - prediction exact rows: `322 -> 336`.
  - n-best better than prediction: `630 -> 602`.
  - prediction worse than base: `187 -> 116` in the relation diagnostic.
  - Therefore the prompt does not merely revert to top1; it makes selection more selective and reduces harmful edits.
- Hotword behavior:
  - strong selector recall: `0.88708`.
  - content-priority selector recall: `0.90425`.
  - base recall: `0.90154`.
  - base-hit hotwords lost by prediction drops: `35 -> 15`.
- Interpretation:
  - Yes, prompt engineering can reduce over-attention to fillers and recover much of the hotword loss without retraining.
  - This is the best Shuili prompt setting so far: it improves raw CER and filler-normalized CER over the strong selector, while keeping hotword recall slightly above base.
  - Remaining gap to n-best oracle (`0.06488`) still requires training/objective changes, but the next training should use the content-priority prompt as the inference/training style rather than the overly aggressive strong selector prompt.

Content-priority protected/near-correct selector SFT, 2026-07-01:

- Motivation:
  - Train with the better `selector_content` prompt style instead of only using it at inference.
  - Add two stabilizers to the selector data: protected-hotword rows and near-correct no-break rows, so the model is less likely to damage correct domain terms or nearly-correct top1 outputs.
- Training data:
  - `covo/data/processed/chinesehp_aishell1/train_nbest_selector_content_protect_near2_hardx4_exactx2_base6k_noopx2.qwen.jsonl`
  - Source rows: `17301`; written rows: `69514`.
  - hard rows `6561`, exact-hard rows `4188`, base rows `6000`, near-correct rows `14447`, protected rows `130`.
  - Prompt mode: `selector_content`; protected target margin `2`; near-correct distance `2`.
- Training:
  - Continued from `outputs/qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`.
  - Output adapter: `outputs/qwen35_cbwhisper_selector_content_protect_near2_lr5e7_bs2_1epoch_from_strong_bf16`.
  - Stable setting was batch size `2`, grad accumulation `2`; batch size `4` OOMed.
  - Full epoch completed: `17379/17379` steps, train loss `0.2478`.
- Shuili validation:
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`.
  - Raw CER: base/top1 `0.15415` -> prediction `0.12088`.
  - COVO movement: improved / worsened / unchanged `234 / 60 / 858`.
  - Audit CER: base `0.15415`, prediction `0.12088`, n-best oracle `0.06488`, prediction-or-nbest oracle `0.05689`.
  - Exact rows: base `257`, prediction `320`, n-best oracle `688`.
  - Hotword recall: base `0.90154`, prediction `0.90786`, n-best oracle `0.88708`.
  - Lost base-hit hotwords: `10`; gained over base: `17`; false prompt insertions drop `44 -> 30`.
  - Extended filler-normalized CER: base `0.09814`, prediction `0.08534`.
  - This is better than the strong selector on recall, but worse than the prompt-only content selector on Shuili raw CER (`0.11428`) and extended filler-normalized CER (`0.07945`).
- AISHELL 808 validation:
  - Evidence: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean.jsonl`.
  - Raw CER: base/top1 `0.10446` -> prediction `0.06698`.
  - COVO movement: improved / worsened / unchanged `213 / 28 / 567`.
  - Audit CER: base `0.10446`, prediction `0.06698`, n-best oracle `0.05355`, prediction-or-nbest oracle `0.04354`.
  - Exact rows: base `369`, prediction `468`, n-best oracle `513`.
  - Hotword recall: base `0.91507`, prediction `0.91189`, n-best oracle `0.91614`.
  - Lost base-hit hotwords: `24`; gained over base: `21`; false prompt insertions drop `30 -> 13`.
  - Filler-normalized CER: base `0.08995`, prediction `0.06676`.
- Interpretation:
  - The new training clearly improves AISHELL versus the previous strong selector validation (`0.07055` CER and `0.87261` recall), especially hotword preservation.
  - It does not improve Shuili over the prompt-only content selector, likely because all training rows come from AISHELL and teach AISHELL-style selection rather than Shuili lecture-style/domain errors.
  - Keep this adapter as an AISHELL selector-improvement checkpoint, but do not promote it as the best Shuili model. For Shuili, the best current COVO setting remains the prompt-only content selector on the previous strong selector adapter.

ChineseHP-style clean/consensus/hard-negative evidence preparation, 2026-07-01:

- Motivation:
  - ChineseHP succeeds partly because the model sees structured evidence: stable spans, uncertain spans, variants, and hard-negative/confusable candidates.
  - Our CB-Whisper -> COVO bridge previously exposed mostly whole-sentence n-best candidates plus scores/hotword labels, leaving the model to infer all local candidate structure by itself.
  - User asked to first clean two gaps: candidate pollution/outliers and missing ChineseHP-style hard-negative/confusable evidence.
- Code change:
  - Extended `src/analysis/cbwhisper_covo_bridge.py` with optional prompt-time n-best cleaning:
    - `--clean-nbest`
    - removes duplicates, obvious pollution tails, replacement characters, long Latin tails, repeat-heavy strings, and length outliers.
  - Added optional structured evidence:
    - `--include-consensus-spans`
    - `Stable spans`: majority-supported top1 spans.
    - `Uncertain spans`: low-support top1 spans with candidate variants.
    - `--max-confusables`: ChineseHP-style local diff summaries for similar but divergent candidates.
  - Added a short instruction explaining how to use stable/uncertain/confusable evidence while still respecting hotword evidence.
- Generated data:
  - Train:
    - `covo/data/processed/chinesehp_aishell1/train_full_cbwhisper_consensus_hotword_clean.qwen.jsonl`
    - rows: `17301`
    - size: `216.45 MB`
    - rows changed by cleaning: `273`
    - dropped candidates: `587`
    - avg stable spans: `1.2691`
    - avg uncertain spans: `1.3523`
    - avg confusable candidates: `3.6559`
    - avg hotwords: `8.0`
    - avg prompt chars: `3351`
  - AISHELL 808 validation messages:
    - `covo/data/processed/chinesehp_aishell1/test808_cbwhisper_consensus_hotword_clean.qwen.jsonl`
    - rows: `808`
    - size: `13.50 MB`
    - rows changed by cleaning: `19`
    - dropped candidates: `55`
    - avg stable spans: `1.3428`
    - avg uncertain spans: `1.3923`
    - avg confusable candidates: `4.7079`
    - avg hotwords: `5.3032`
    - avg prompt chars: `3860`
- Example behavior:
  - For `国务院发展研究钟欣市场经济研修所...`, the bridge now exposes:
    - stable spans: `国务院发展研究`, `市场经济`.
    - uncertain span `钟欣` with variants `钟欣/中心`.
    - uncertain span `研修` with variants such as `研修/研教/研求/学院/研作`.
    - confusable candidates with local diffs and hotword preservation tags.
- Decision:
  - Keep this as the next clean COVO training input format.
  - Do not train yet until comparing whether starting from the current hotword-preserving adapter or the original ChineseHP hard-negative adapter is preferred.

ChineseHP-style + hotword-aware COVO SFT data, 2026-07-01:

- Motivation:
  - The first consensus/hard-negative bridge still expressed hotword information mostly as inline text.
  - User clarified the desired route: keep ChineseHP text-rewrite hard-negative style, but explicitly expose hotword fields so the model learns:
    - use n-best to fix true ASR errors;
    - reject false hotwords;
    - preserve already-correct hotwords;
    - keep top1 when evidence is insufficient.
- Code change:
  - Extended `src/analysis/cbwhisper_covo_bridge.py` with `--include-hotword-evidence`.
  - Added structured prompt fields:
    - `protected_hotwords`
    - `prompt_hotwords`
    - `kws_hotwords`
    - `hotword_conflict`
    - per-candidate `variant_keeps_hotword`
    - per-candidate `variant_drops_hotword`
    - `false_hotword_warning`
  - Added OpenCC t2s normalization inside `normalize_text`, so traditional Whisper outputs like `經濟` match simplified hotwords like `经济`.
  - Confusable candidates now include a `hotword_delta` JSON block.
  - Uncertain span variants include local keep/drop/false-warning hotword annotations.
- Generated data:
  - Train:
    - `covo/data/processed/chinesehp_aishell1/train_full_cbwhisper_chinesehp_hotword_aware.qwen.jsonl`
    - rows: `17301`
    - size: `280.47 MB`
    - prompt token estimate on first 1000 rows: p50 `2122.5`, p90 `2652`, p95 `2758`, p99 `3089`, max `3308`
    - prompt char stats: p50 `5739`, p95 `7495`, p99 `8114`, max `9018`
    - protected rows: `13952`
    - hotword conflict rows: `5025`
    - rows changed by n-best cleaning: `273`
    - dropped candidates: `587`
    - avg stable spans: `1.060`
    - avg uncertain spans: `1.145`
  - AISHELL 808 validation messages:
    - `covo/data/processed/chinesehp_aishell1/test808_cbwhisper_chinesehp_hotword_aware.qwen.jsonl`
    - rows: `808`
    - size: `17.71 MB`
    - prompt char stats: p50 `7015`, p95 `8591`, p99 `9132`, max `9561`
    - protected rows: `754`
    - hotword conflict rows: `303`
    - rows changed by n-best cleaning: `19`
    - dropped candidates: `55`
    - avg stable spans: `1.212`
    - avg uncertain spans: `1.271`
- Training plan:
  - Start from the current best hotword-preserving adapter:
    - `outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - Use `max_length=4096` because token estimates show 3072 may truncate the long evidence prompts.
  - Use conservative LR `5e-7` to add evidence usage without destroying hotword-preservation behavior.
- Training run:
  - Started in tmux session `covo_hotword_aware_train`.
  - Log: `src/logs/train_covo_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_stdout.log`
  - Output adapter: `covo/outputs/qwen35_cbwhisper_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_bf16`
  - Settings: `max_length=4096`, batch size `2`, grad accumulation `2`, LR `5e-7`, constant scheduler, bf16, gradient checkpointing.
  - Initial runtime check: about `7.5s/step`, `4326` total steps, GPU memory about `30.2/32.6GB`, utilization about `83%`.
  - Completed full epoch on 2026-07-02: `4326/4326` steps, train loss `0.2478`, runtime about `8h43m50s`.
- AISHELL 808 validation:
  - Messages: `src/logs/cbwhisper_covo_messages_aishell_v3_chinesehp_hotword_aware_20260702.jsonl`
  - Final checkpoint predictions: `src/logs/cbwhisper_covo_predictions_aishell_v3_chinesehp_hotword_aware_20260702.jsonl`
  - Final checkpoint COVO evaluator: base CER `0.10446` -> prediction CER `0.04967`, improved / worsened / unchanged `330 / 83 / 395`.
  - Final checkpoint audit:
    - CER: base `0.10446`, prediction `0.04967`, n-best oracle `0.05363`, prediction-or-nbest oracle `0.02864`.
    - Exact rows: base `369`, prediction `500`, n-best oracle `513`.
    - Hotword recall: base `0.91507`, prediction `0.84395`, n-best oracle `0.91614`.
    - Base-hit hotwords lost by prediction: `98`; gained over base: `31`.
    - Base-correct-broken rows: `53`; worsened-error rows: `30`.
  - Checkpoint-1000 early-stop probe:
    - Predictions: `src/logs/cbwhisper_covo_predictions_aishell_v3_chinesehp_hotword_aware_ckpt1000_20260702.jsonl`
    - COVO evaluator: base CER `0.10446` -> prediction CER `0.04463`, improved / worsened / unchanged `322 / 52 / 434`.
    - Audit CER: base `0.10446`, prediction `0.04463`, n-best oracle `0.05363`, prediction-or-nbest oracle `0.02895`.
    - Exact rows: base `369`, prediction `524`, n-best oracle `513`.
    - Hotword recall: base `0.91507`, prediction `0.88323`, n-best oracle `0.91614`.
    - Base-hit hotwords lost by prediction: `61`; gained over base: `31`.
- Interpretation:
  - Reject this route as a main result.
  - The full-epoch model overfits/over-rewrites badly: it improves generic CER over base but destroys hotword preservation, often changing already-correct protected names into common homophones, e.g. `王晔君 -> 王艳君`, `宋芳 -> 颂芳`, `今久 -> 金九`, `刘澄 -> 刘成`.
  - Checkpoint-1000 is much safer than the final checkpoint but still worse than the current main AISHELL result (`CER 0.04284`, recall `0.91083`) and worse than protected-preserve recall-best (`CER 0.04292`, recall `0.91295`).
  - Likely cause: long ChineseHP-style evidence + reference SFT teaches the model to trust fluent correction/reference normalization more than exact hotword preservation. Structured fields alone are not enough; future attempts need loss masking/targeting only uncertain spans, or explicit contrastive/protected-hotword objectives, rather than full reference SFT on the whole long prompt.

Shuili model sweep after stopping AISHELL work, 2026-07-04:

- User direction:
  - Stop working on AISHELL for now.
  - Switch to the Shuili dataset and evaluate the currently useful COVO models.
- Dataset/evidence:
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`
  - Rows: `1152`
  - Baseline CB-Whisper top1:
    - Raw CER `0.15415`
    - Extended filler-normalized CER `0.09814`
    - Minimal filler-normalized CER `0.09900`
    - Hotword recall `0.90154`
    - Exact rows `257`
- Newly evaluated models on 2026-07-04:
  - `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - `qwen35_cbwhisper_actual_hard_mild_sft40_from_synth_bf16`
  - `qwen35_cbwhisper_protected_positive6k_lostx40_anchor2k_sft100_from_expanded_bf16`
  - `qwen35_cbwhisper_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_bf16/checkpoint-1000`
- Results table:

| Model / setting | Raw CER | Extended filler CER | Minimal filler CER | Hotword recall | Exact rows | Base-hit lost | Improved / worsened / unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CB-Whisper top1/base | 0.15415 | 0.09814 | 0.09900 | 0.90154 | 257 | - | - |
| preserve2 no-op bs7 | 0.13402 | 0.09013 | 0.09059 | 0.91960 | 280 | 4 | 151 / 45 / 956 |
| actual-error mild SFT40 | 0.13244 | 0.08911 | 0.08918 | 0.92141 | 284 | 4 | 173 / 56 / 923 |
| expanded rewrite60k+noop SFT120 | 0.13313 | 0.08903 | 0.08965 | 0.92231 | 284 | 4 | 171 / 58 / 923 |
| protected-preserve SFT40 | 0.13332 | 0.08919 | 0.08978 | 0.92231 | 281 | 4 | 165 / 55 / 932 |
| protected-positive SFT100 | 0.13225 | 0.08872 | 0.08898 | 0.92141 | 285 | 4 | 170 / 54 / 928 |
| selector hardx4 SFT160 | 0.13275 | 0.08903 | 0.08938 | 0.92231 | 284 | 4 | 172 / 58 / 922 |
| selector-prompt hardx4 SFT160 | 0.13199 | 0.08778 | 0.08844 | 0.92322 | 287 | 4 | 182 / 60 / 910 |
| strong selector 1epoch | 0.11936 | 0.08566 | 0.08003 | 0.88708 | 322 | 35 | 349 / 172 / 631 |
| strong selector + content prompt | 0.11428 | 0.07945 | 0.07472 | 0.90425 | 336 | 15 | 320 / 104 / 728 |
| content-protect-near selector | 0.12088 | 0.08534 | 0.08084 | 0.90786 | 320 | 10 | 234 / 60 / 858 |
| hotword-aware checkpoint-1000 | 0.13009 | 0.08652 | 0.08635 | 0.92322 | 276 | 2 | 186 / 70 / 896 |

- Interpretation:
  - Best raw CER and best filler-normalized CER remain `strong selector + content prompt`:
    - Raw CER `0.11428`
    - Extended filler CER `0.07945`
    - Minimal filler CER `0.07472`
  - Best hotword recall is tied by `selector-prompt hardx4 SFT160` and `hotword-aware checkpoint-1000`, both `0.92322`.
  - `hotword-aware checkpoint-1000` transfers better to Shuili than it did to AISHELL recall-wise, losing only `2` base-hit hotwords, but its raw CER/exact rows are not competitive with the selector route.
  - `protected-positive SFT100` slightly improves over expanded/protected-preserve in raw CER and filler-normalized CER, but it is still far behind the Shuili selector-content route.
  - For Shuili, the bottleneck is still candidate selection / lecture-style output, not only hotword preservation. Selector-style behavior is much more important than the AISHELL-style conservative correction adapters.
- Current Shuili recommendations:
  - Main Shuili CER result: `strong selector + content prompt`.
  - Recall-oriented Shuili ablation: `selector-prompt hardx4 SFT160` or `hotword-aware checkpoint-1000`.
  - Balanced fallback with low hotword loss and moderate CER: `protected-positive SFT100` or `actual-error mild SFT40`.

Shuili spoken-style selector prompt-only ablation, 2026-07-04:

- Motivation:
  - Raw CER analysis showed the dominant remaining errors are deleted classroom spoken fillers/discourse words (`呢/啊/这个/这一个/那么/咱们/的话`) plus some water-domain homophone errors.
  - User rejected adding domain lexicons or new training for this step because the work should remain paper-clean.
  - Therefore this is a prompt-only ablation on the existing strong selector adapter.
- Code change:
  - Added `--prompt-mode selector_spoken` to `src/analysis/cbwhisper_covo_bridge.py`.
  - The prompt tells COVO to preserve classroom spoken style and avoid turning lecture transcription into a written summary.
  - It treats ordinary spoken fillers as valid transcript content rather than pollution when they appear in top1 or trusted n-best candidates.
- Setup:
  - Adapter: `outputs/qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`
  - Predictions: `src/logs/cbwhisper_covo_predictions_shuili_v3_selector_spoken_lr1e6_1epoch_20260704.jsonl`
- Result:

| Prompt | Raw CER | Extended filler CER | Minimal filler CER | Recall | Exact rows | Base-hit lost | Improved / worsened / unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong selector | 0.11936 | 0.08566 | 0.08003 | 0.88708 | 322 | 35 | 349 / 172 / 631 |
| selector_content | 0.11428 | 0.07945 | 0.07472 | 0.90425 | 336 | 15 | 320 / 104 / 728 |
| selector_spoken | 0.11206 | 0.07985 | 0.07445 | 0.90425 | 356 | 16 | 335 / 94 / 723 |

- Spoken-word deletion diagnostic:
  - `selector_content` missing rows:
    - `呢`: `477`
    - `啊`: `88`
    - `这个`: `48`
    - `这一个`: `28`
    - `那么`: `15`
    - `咱们`: `5`
    - `的话`: `5`
  - `selector_spoken` missing rows:
    - `呢`: `456`
    - `啊`: `83`
    - `这个`: `46`
    - `这一个`: `25`
    - `那么`: `14`
    - `咱们`: `3`
    - `的话`: `3`
- Interpretation:
  - This is a clean prompt-only gain and becomes the current best Shuili raw CER result.
  - It improves raw CER from `0.11428` to `0.11206` and exact rows from `336` to `356` while keeping recall unchanged at `0.90425`.
  - The improvement is consistent with the diagnosis: preserving spoken lecture style recovers some raw-CER errors that the content-priority prompt still deleted.
  - Extended filler-normalized CER is slightly worse than selector_content (`0.07985` vs `0.07945`) because this metric removes many of the recovered words; raw CER should be the main metric for this ablation.

Shuili high-quality n-best candidate-pool audit, 2026-07-04:

- Motivation:
  - The previous Shuili COVO runs used the existing CB-Whisper evidence directly.
  - A normalized audit showed this was not equivalent to the AISHELL `nbest10` work: Shuili had average unique n-best only about `7.49`, and no rows with 10 unique candidates.
  - Therefore the Shuili pipeline needed a candidate-pool construction step before judging COVO.
- Generation:
  - Ran multi-prompt Whisper large-v3 candidate generation on all `1152` Shuili rows.
  - Prompts: no prompt, parenthesized top-3 hotwords, natural top-3, natural top-6.
  - Temperatures: `0`, `0.4`, `0.6`; beam/return: `5/5`.
  - Output: `src/logs/shuili_hotword_multiprompt_nbest_t046_full_20260704.jsonl`.
  - The auxiliary multiprompt pool alone had average unique n-best `6.8819`, oracle corpus CER `0.05911`, and oracle hotword recall `0.93658`; its top1 hotword recall was poor (`0.66464`), so it is useful only as a complementary candidate source.
- CB-Whisper + multiprompt union:
  - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_20260704.jsonl`.
  - Policy: keep CB-Whisper top1 first, keep the first 5 CB n-best candidates, insert multiprompt candidates, then fill remaining slots with later CB candidates.
  - Metrics:
    - Average unique n-best: `9.0634`
    - Rows with 10 unique candidates: `798/1152`
    - Exact reference in pool: `789/1152`
    - Oracle corpus CER: `0.04330`
    - Oracle hotword recall: `0.93496`
  - This is much better than the previous Shuili evidence pool, but not yet as clean as the AISHELL strict `9.8+` candidate pool.
- Strict cleanliness audit:
  - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_audit_20260704.jsonl`.
  - Strict cleaning removed `245` candidates:
    - `repeat_noise=122`
    - `too_short=73`
    - `too_long=26`
    - `unusual_unicode=13`
    - `latin_tail=8`
    - `bad_phrase=3`
  - Cleaned metrics:
    - Average unique n-best: `8.8507`
    - Rows with 10 unique candidates: `702/1152`
    - Exact reference in pool: `776/1152`
    - Oracle corpus CER: `0.04647`
    - Oracle hotword recall: `0.93315`
  - Representative clean candidates are meaningful spoken-ASR variants such as missing/keeping `呢`, `这个/这一个`, homophones, and phrase-boundary variants.
  - Remaining conclusion: the cleaned Shuili pool is COVO-usable and clearly better than the original pool, but it does not yet reach ChineseHP/AISHELL-level `9.8+` clean unique n-best. Prefer the strict-clean file for downstream COVO unless the goal is explicitly to test sensitivity to noisier high-diversity candidates.
- Short-utterance length-consistency follow-up:
  - Problem: strict clean still allowed fragment-like candidates such as `那么这个 / 那么这个呢 / 这个呢`; these are not ideal whole-utterance n-best alternatives.
  - Added optional length-consistency filtering for short utterances in `clean_covo_nbest_quality.py`.
  - Exact same-length filtering is too aggressive:
    - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shortlen_v2_20260704.jsonl`
    - Average unique n-best `8.2179`, exact ref in pool `726`, oracle corpus CER `0.05270`.
  - A softer one-character margin is a better compromise:
    - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shortmargin1_v2_20260704.jsonl`
    - Average unique n-best `8.5113`, exact ref in pool `758`, oracle corpus CER `0.04850`.
    - Example `那么这个 / 那么这个呢 / 这个呢` becomes `那么这个 / 那么这个呢`, removing the obvious fragment while retaining a plausible filler-drop variant.
  - Recommendation: use `--short-length-margin 1` for Shuili short-utterance cleanup if the goal is cleaner whole-utterance candidates; avoid `--short-exact-length` unless the evaluation explicitly requires equal-length hypotheses.
- Downstream COVO check on cleaned Shuili pools:
  - Adapter: `qwen35_cbwhisper_nbest_selector_strongprompt_lr1e6_1epoch_from_selector_bf16`
  - Prompt mode: `selector_spoken`
  - `strict-clean` input:
    - Predictions: `src/logs/cbwhisper_covo_predictions_shuili_v3_strict_selector_spoken_20260704.jsonl`
    - Raw CER `0.11745`, recall `0.91057`, exact rows `329`, base-hit lost `16`
    - Extended filler CER `0.08102`, minimal filler CER `0.07613`
  - `short-margin=1` input:
    - Predictions: `src/logs/cbwhisper_covo_predictions_shuili_v3_shortmargin1_selector_spoken_20260704.jsonl`
    - Raw CER `0.11669`, recall `0.90967`, exact rows `331`, base-hit lost `16`
    - Extended filler CER `0.08126`, minimal filler CER `0.07600`
  - Comparison to previous best Shuili selector_spoken on original evidence:
    - Previous best raw CER `0.11206`, recall `0.90425`, exact rows `356`, base-hit lost `16`
  - Interpretation:
    - `short-margin=1` is slightly better than `strict-clean` on raw CER and exact rows, so the whole-utterance length-consistency direction is reasonable.
    - However, both cleaned 10-best pools regress relative to original selector_spoken COVO output. The current COVO selector adapter does not exploit the richer/cleaner Shuili candidate pool yet.
    - Do not regenerate the whole Shuili multiprompt pool immediately. The next useful step is either targeted regeneration for rows with low unique/oracle miss, or COVO training/adaptation on this cleaner Shuili-style evidence.

Shuili regenerated conservative candidate pool + 4.35 COVO, 2026-07-04:

- Motivation:
  - The first Shuili cleaned-pool COVO check reused the previously generated `t046` multiprompt pool.
  - Re-ran the candidate generation itself with a more conservative temperature set before judging the 4.35 COVO adapter.
- Regenerated multiprompt generation:
  - File: `src/logs/shuili_hotword_multiprompt_nbest_t035_full_20260704.jsonl`
  - Whisper: `openai/whisper-large-v3`
  - Temperatures: `0,0.3,0.5`; beams/return: `5/5`; max auxiliary n-best: `20`.
  - Summary: average unique n-best `5.7934`, top1 corpus CER `0.12133`, oracle corpus CER `0.06457`, top1 hotword recall `0.66464`, oracle hotword recall `0.93223`.
  - Interpretation: the auxiliary pool alone is not a good final recognizer, but it adds useful alternatives for the union pool.
- CB-Whisper + regenerated multiprompt union:
  - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t035_keep5_20260704.jsonl`
  - Policy: keep CB-Whisper top1, keep first 5 CB n-best, insert multiprompt alternatives, then fill remaining slots with later CB candidates.
  - Metrics:
    - Average unique n-best: `9.0087`
    - Rows with 10 unique candidates: `777/1152`
    - Exact reference in pool: `787/1152`
    - Top1 corpus CER: `0.13910`
    - Oracle corpus CER: `0.04400`
    - Top1 hotword recall: `0.91147`
    - Oracle hotword recall: `0.93857`
- Short-margin cleanup:
  - File: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t035_keep5_clean_shortmargin1_20260704.jsonl`
  - Cleaning removed `570` candidates:
    - `short_len_mismatch=362`
    - `repeat_noise=111`
    - `too_short=67`
    - `too_long=16`
    - `latin_tail=7`
    - `unusual_unicode=4`
    - `bad_phrase=3`
  - Cleaned metrics:
    - Average unique n-best: `8.5139`
    - Rows with 10 unique candidates: `652/1152`
    - Exact reference in pool: `760/1152`
    - Top1 corpus CER: `0.13040`
    - Oracle corpus CER: `0.04825`
    - Top1 hotword recall: `0.91057`
    - Oracle hotword recall: `0.93496`
- 4.35 COVO result:
  - Adapter: `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - Predictions: `src/logs/cbwhisper_covo_predictions_shuili_v3_regen_t035_clean_shortmargin1_preserve2_noop_bs7_20260704.jsonl`
  - Raw evaluator result:
    - COVO CER: `0.13377`
    - Baseline CER from `input.asr_top1`: `0.15415`
    - Improved/worsened/unchanged samples: `165/62/925`
  - Error audit:
    - Base CER: `0.15415`
    - Prediction CER: `0.13377`
    - N-best oracle CER: `0.05796`
    - Best of prediction and n-best oracle CER: `0.05244`
    - Delta counts: `fixed_to_exact=48`, `improved_partial=117`, `worsened_error=41`, `base_correct_broken=21`, `unchanged_error=689`, `base_correct_kept=236`.
    - Hotword recall: base `0.90154`, prediction `0.92051`, n-best oracle `0.90515`; COVO gained `27` hotword hits and lost `6` base hits.
  - Filler-normalized CER:
    - Extended filler list: base `0.09814`, prediction `0.08676`.
    - Minimal filler list (`呃,呢,啊,嗯`): base `0.09900`, prediction `0.08797`.
- Interpretation:
  - Regenerating and cleaning the candidate pool improved the candidate-side upper bound: cleaned oracle CER is `0.04825`, much lower than the final COVO CER.
  - The 4.35 adapter increases hotword recall, but it does not select or rewrite toward the oracle candidates well enough on Shuili.
  - The largest remaining error bucket is `unchanged_error=689`, so the model often keeps the input even when the n-best pool contains better evidence.
  - This route proves the bottleneck is no longer only candidate availability; the downstream COVO model needs Shuili-style evidence adaptation or a stronger ChineseHP-style training format before the richer n-best pool can translate into sub-10 CER.

MagicData-RAMC oral data inspection, 2026-07-04:

- Source:
  - Downloaded OpenSLR 123 `MagicData-RAMC.tar.gz` to `datasets/magicdata_ramc/MagicData-RAMC.tar.gz`.
  - Size: `15G`; downloaded from the CN mirror with `wget --no-check-certificate` because the mirror certificate was expired.
  - Full extraction was not performed because the workspace had only about `33G` free after download.
- Extracted inspection subsets:
  - `datasets/magicdata_ramc/sample_extract/MDT2021S003/{SPKINFO.txt,UTTERANCEINFO.txt,TXT/CTS-CN-F2F-2019-11-15-506.txt,WAV/CTS-CN-F2F-2019-11-15-506.wav}`
  - `datasets/magicdata_ramc/text_only/MDT2021S003/{SPKINFO.txt,UTTERANCEINFO.txt,TXT/*.txt}`
- Structure:
  - Root directory: `MDT2021S003`
  - Metadata: `SPKINFO.txt`, `UTTERANCEINFO.txt`
  - Audio: `WAV/*.wav`, each file is a long conversation recording.
  - Transcript: `TXT/*.txt`, with segment-level timestamps and speaker ids.
  - Example transcript row format: `[start,end]\tspeaker\tgender,accent\ttext`
- Audio sample:
  - Example file `CTS-CN-F2F-2019-11-15-506.wav`
  - `16000 Hz`, mono, signed 16-bit PCM.
  - Duration `1867.475s` (`31.12 min`), size about `60M`.
- Corpus text statistics from extracted `TXT/*.txt`:
  - Sessions: `351`
  - Valid hours from `UTTERANCEINFO.txt`: `150.73h`
  - Total hours from `UTTERANCEINFO.txt`: `180.18h`
  - Text files: `351`
  - All transcript rows: `219325`
  - Valid text rows: `204399`
  - Special/noise rows (`[*]`, `[+]`): `14926`
  - Segment duration: mean `2.54s`, median `1.91s`, p90 `5.75s`
  - Normalized chars per segment: mean `12.91`, median `10`, p90 `28`
- Oral markers:
  - `就是`: `32626`
  - `嗯`: `25996`
  - `然后`: `25222`
  - `那个`: `20757`
  - `啊`: `16948`
  - `吧`: `11348`
  - `这个`: `10534`
  - `嘛`: `8758`
  - `呀`: `7189`
  - `儿`: `6434`
  - `呢`: `6205`
  - `的话`: `6161`
  - `呃`: `5808`
  - `咱`: `2868`
  - `那么`: `1782`
- Suitability judgment:
  - This dataset is highly suitable for Shuili-style oral COVO training.
  - It contains real spontaneous Mandarin conversation, short timestamped utterances, and abundant filler/oral discourse markers.
  - The most useful path is not to train a new ASR model immediately, but to use the text segments to synthesize oral-style COVO SFT pairs:
    - complete oral reference as target;
    - simulated top1 with deleted fillers, shortened spans, and false hotword insertions;
    - n-best candidates containing both short/wrong and complete/oral variants.
  - Because audio is long conversation-level WAV, full audio preprocessing would require segment cutting from timestamps; this is feasible but should be done only after deciding whether we need Whisper/KWS hidden states from RAMC.

ChineseHP coverage check and RAMC direct-training setup, 2026-07-04:

- User asked whether ChineseHP has n-best candidates for MagicData-RAMC and suggested training without CB-Whisper if needed.
- Public and local checks:
  - ChineseHP official repo: `https://github.com/tzyll/ChineseHP`
  - Repo file tree contains n-best/text/pinyin for:
    - `aishell-1`
    - `aishell-4`
    - `kespeech`
    - `wenetspeech`
  - No `MagicData-RAMC`, `RAMC`, `MDT2021S003`, or OpenSLR 123 split is present.
  - Conclusion: ChineseHP does not provide RAMC n-best candidates. RAMC n-best must be generated or synthesized.
- Direct RAMC synthetic oral-rewrite data:
  - Added script: `src/analysis/build_ramc_oral_rewrite_data.py`
  - Input: `datasets/magicdata_ramc/text_only/MDT2021S003/TXT`
  - Output raw files:
    - `covo/data/processed/ramc_oral/ramc_oral_synthetic_raw.jsonl`
    - `covo/data/processed/ramc_oral/train_raw.jsonl`
    - `covo/data/processed/ramc_oral/dev_raw.jsonl`
  - Available filtered RAMC text records: `162059`
  - First split used for pilot:
    - train: `50000`
    - dev: `3000`
  - Synthetic corruption types:
    - delete one or multiple oral/filler words;
    - shorten prefix/suffix/middle spans;
    - duplicate local spans;
    - insert Shuili-like false domain words such as `水利工程`, `闸门`, `地基`, `石方`, `施工技术`;
    - include no-op rows where ASR top1 is already correct.
  - Qwen message exports:
    - `covo/data/processed/ramc_oral/train_oral_rewrite_50k.qwen.jsonl`
    - `covo/data/processed/ramc_oral/dev_oral_rewrite_3k.qwen.jsonl`
  - Export mode: final-text rewrite with n-best, pinyin, and hard-negative/confusable candidates.
  - Dry-run passed with `train_rows=50000`, `eval_rows=3000`.
- Planned pilot training:
  - Start adapter: original ChineseHP text-rewrite hard-negative model `covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch`, not a CB-Whisper adapter.
  - Base model: `covo_migration/.../models/Qwen3.5-4B`.
  - Training target: teach the model to recover full oral-style text from synthetic ASR top1/n-best evidence without relying on CB-Whisper.
  - First run should be capped around `200` steps to verify loss/format before scaling.

RAMC direct oral-rewrite full SFT result, 2026-07-05:

- User asked to run full training after the pilot.
- Pilot issue:
  - The first capped `200` step run used `max_length=1536`, batch `7`, grad accumulation `4`.
  - It reached step `23/200` and then failed with CUDA OOM while allocating about `6.98 GiB`.
  - The first logged loss before OOM was about `1.037`.
  - Fix for full run: reduce `max_length` to `1024`, reduce batch to `4`, use grad accumulation `7`, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Full synthetic RAMC data:
  - Raw full file: `covo/data/processed/ramc_oral/ramc_oral_synthetic_full_raw.jsonl`
  - Train raw: `covo/data/processed/ramc_oral/train_full_raw.jsonl`
  - Dev raw: `covo/data/processed/ramc_oral/dev_full_raw.jsonl`
  - Train messages: `covo/data/processed/ramc_oral/train_oral_rewrite_full157k_hn2.qwen.jsonl`
  - Dev messages: `covo/data/processed/ramc_oral/dev_oral_rewrite_full5k_hn2.qwen.jsonl`
  - Train/dev sizes: `157000/5000`.
  - Hard-negative/confusable candidates capped at `2` for length stability.
- Full SFT setup:
  - Start adapter: `covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch`
  - Output adapter: `covo/outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Base model: `covo_migration/.../models/Qwen3.5-4B`
  - Epochs: `1`
  - Steps: `5608`
  - Batch/accumulation: `4 x 7`
  - Max length: `1024`
  - Learning rate: `5e-6`, linear schedule
  - Precision: bf16
  - Train runtime: about `16:51:19`
  - Final train loss: `0.2206`
  - Final dev eval loss: `0.1733`
  - Output size: about `1.7G` including checkpoints.
- RAMC synthetic dev inference:
  - Predictions: `src/logs/ramc_oral_dev_predictions_full1epoch_from_chinesehp_20260705.jsonl`
  - Log: `src/logs/eval_ramc_oral_dev_full1epoch_from_chinesehp_20260705_stdout.log`
  - Metrics: `src/logs/eval_ramc_oral_dev_full1epoch_from_chinesehp_20260705_metrics.log`
  - Samples: `5000`
  - Baseline CER from synthetic ASR top1: `0.16486`
  - Model CER: `0.00084`
  - Improved/worsened/unchanged: `4083/5/912`
  - Changed predictions: `4089/5000`
  - Parse warnings: none.
- Interpretation:
  - The full RAMC direct SFT cleanly learned the synthetic oral-rewrite task.
  - It can restore deleted oral words such as `就是`, remove duplicate spans, and reject false Shuili-like insertions when the correct variant appears in n-best.
  - This result is not yet a real Shuili test improvement, because the dev set is synthetic RAMC and the n-best is constructed from text.
  - However, it confirms that the model capacity/training objective can learn the exact behavior missing in Shuili: using n-best evidence to recover complete oral-style utterances instead of keeping an over-short top1.
  - Next useful validation is to run this RAMC oral adapter on Shuili cleaned/re-generated n-best and compare against the previous 4.35 COVO result (`raw CER 0.13377`, minimal filler CER `0.08797`).

Shuili validation with RAMC oral adapter, 2026-07-05:

- User asked to test the RAMC oral-rewrite adapter on Shuili.
- Input candidate pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t035_keep5_clean_shortmargin1_20260704.jsonl`
- Prepared COVO messages:
  - `src/logs/shuili_v3_regen_t035_clean_shortmargin1_textrewrite_messages_20260705.jsonl`
- Adapter:
  - `covo/outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
- Predictions and metrics:
  - Predictions: `src/logs/shuili_v3_regen_t035_clean_shortmargin1_ramc_oral_predictions_20260705.jsonl`
  - Eval metrics: `src/logs/eval_shuili_ramc_oral_20260705_metrics.log`
  - Audit summary: `src/logs/covo_error_audit_shuili_v3_regen_t035_clean_shortmargin1_ramc_oral_20260705_summary.json`
  - Minimal filler-normalized metrics: `src/logs/filler_normalized_cer_shuili_v3_regen_t035_clean_shortmargin1_ramc_oral_minfillers_20260705.json`
- Raw Shuili result:
  - Samples: `1152`
  - Baseline CER: `0.15415`
  - COVO CER: `0.11180`
  - Improved/worsened/unchanged samples: `540/218/394`
  - N-best oracle CER: `0.05796`
- Minimal filler-normalized result, removing only `呃/呢/啊/嗯`:
  - Baseline CER: `0.09900`
  - COVO CER: `0.08326`
  - Improved/worsened/unchanged samples: `265/231/656`
- Compared with previous Shuili 4.35 preserve/no-op adapter on the same regenerated clean pool:
  - Previous raw CER: about `0.13377`
  - Previous minimal filler-normalized CER: about `0.08797`
  - Previous unchanged_error was about `689`.
  - RAMC oral adapter raw CER is much better: `0.13377 -> 0.11180`.
  - RAMC oral adapter minimal filler-normalized CER is also better: `0.08797 -> 0.08326`.
  - RAMC oral adapter reduces unchanged_error sharply: `689 -> 241`.
- Main problem:
  - Hotword recall drops badly.
  - Baseline hotword recall in this run: `0.90154`
  - RAMC adapter hotword recall: `0.85366`
  - N-best oracle hotword recall: `0.90515`
  - The model loses `61` hotword mentions that baseline had, and gains only `8`.
  - It also breaks correct baseline outputs often: `base_correct_broken=104`, `worsened_error=114`.
- Interpretation:
  - The RAMC oral training direction is effective for Shuili oral-style incompleteness and unchanged errors.
  - The pure RAMC adapter is too aggressive and not hotword-aware because it starts from the ChineseHP rewrite adapter rather than the hotword-preserving CB-Whisper adapter.
  - This is a useful intermediate result, not a final model: next likely route is RAMC oral data mixed with hotword-preserve/no-op data, or continuing from the strongest hotword-preserving adapter so that oral completion does not sacrifice hotword recall.

Shuili candidate-pool distribution check, 2026-07-05:

- Motivation:
  - The RAMC oral adapter still had `unchanged_error=241` on Shuili, so the next question was whether the n-best pool was mismatched to Shuili's oral lecture distribution.
- Full-label audit of the previous RAMC Shuili run:
  - `unchanged_error=241`
  - Among those unchanged errors, `154` had a better n-best candidate than the model output.
  - `84` had an exact-reference candidate in n-best.
  - `138` had candidates longer than baseline, and `100` had candidates that added oral markers/fillers over baseline.
  - Interpretation: the candidate pool still contains useful evidence, but the model often refuses to use it.
- Candidate-pool comparison:
  - Previous pool used for RAMC validation:
    - `cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t035_keep5_clean_shortmargin1_20260704.jsonl`
    - Avg n-best: `8.576`
    - Exact refs in n-best: `701`
    - Oracle CER: `0.05796`
    - Oral marker coverage: `0.9151`
  - Raw `t046` pool:
    - Avg n-best: `9.063`
    - Exact refs in n-best: `720`
    - Oracle CER: `0.05263`
    - Oral marker coverage: `0.9231`
    - But it retains some severe pollution/noise.
  - `t046 clean_audit` pool:
    - Avg n-best: `8.865`
    - Exact refs in n-best: `709`
    - Oracle CER: `0.05574`
    - Severe pollution is nearly removed, but useful oral repetition is over-cleaned.
- Cleaning issue found:
  - The existing repeat cleaner incorrectly removed Shuili-natural repetitions such as:
    - `土石土石`
    - `一层一层`
    - `浇筑浇筑`
    - `施工组织施工组织`
    - `怎么选怎么选`
  - These repetitions are natural in the Shuili classroom speech style and sometimes are the exact reference.
- Code change:
  - Added optional `--allow-natural-repeats` to `src/analysis/clean_covo_nbest_quality.py`.
  - Default behavior is unchanged.
  - With the option enabled, adjacent multi-character repetitions are kept unless they are still caught by other severe quality filters.
- Selected Shuili-distribution pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
  - Generated from raw `t046` using:
    - `--allow-natural-repeats`
    - `--max-ratio 1.8`
  - Summary:
    - Avg n-best: about `8.963`
    - Exact refs in n-best: `720`
    - Oracle CER: `0.05333`
    - Oral marker coverage: `0.9225`
    - Severe pollution rate: effectively `0`
    - For the old `241` unchanged-error samples: `85` exact refs, `157` better candidates, `103` oral-marker-adding candidates.
- RAMC oral adapter validation on the selected pool:
  - Predictions:
    - `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_ramc_oral_predictions_20260705.jsonl`
  - Raw result:
    - Baseline CER: `0.15415`
    - COVO CER: `0.11593`
    - Improved/worsened/unchanged: `479/169/504`
    - N-best oracle CER: `0.05333`
  - Audit:
    - `unchanged_error=321`
    - `base_correct_broken=74`
    - `worsened_error=95`
    - Hotword recall: baseline `0.90154`, prediction `0.86631`, oracle `0.90063`
  - Minimal filler-normalized CER, removing only `呃/呢/啊/嗯`:
    - Baseline CER: `0.09900`
    - Prediction CER: `0.08245`
- Interpretation:
  - The selected pool is more Shuili-like and has a better oracle ceiling than the previous pool.
  - However, the current RAMC oral adapter does not convert that better ceiling into better raw CER: raw CER worsens from `0.11180` to `0.11593`.
  - The model becomes more conservative on the richer/more oral pool: fewer broken-correct and worsened-error samples, but many more unchanged errors.
  - For filler-normalized CER the selected pool is slightly better (`0.08326 -> 0.08245`), which supports the idea that this pool is distributionally closer to oral Shuili, but the adapter still needs training/objective changes to use these candidates for raw CER.

Shuili COVO hotword-recall training probes, 2026-07-05:

- User requirement:
  - Hotword recall should also be solved through training, not only by prompt/gating/rerank patches.
  - Small-step training first; only extend training if the direction is effective.
- New data builder:
  - Added `src/analysis/build_shuili_style_oral_sft.py`.
  - It builds Shuili-style oral COVO SFT rows from RAMC raw text, synthesizing 8-10 n-best candidates with short fragments, filler deletion, natural repetition, false domain insertions, and no-op rows.
  - It can mix external AISHELL/CB-Whisper hotword/no-op Qwen-message rows.
  - Added `--hotword-aware-ramc` so synthetic RAMC rows also expose `protected_hotwords`, `prompt_hotwords`, false hotword warnings, and per-candidate keep/drop hotword support.
- Probe A: RAMC oral + external hotword mix.
  - Training data:
    - `train_shuili_style_oral_hotword_mix30k_20260705.qwen.jsonl`
    - `30000` train / `1500` dev rows.
    - RAMC rewrite `15357`, RAMC no-op `5118`, external hotword/no-op `11025`.
  - Start adapter:
    - `outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Output adapter:
    - `outputs/qwen35_ramc_shuili_style_hotword_mix_sft160_from_ramc_oral_bf16`
  - Training:
    - `160` steps, final eval loss `0.83793`.
  - Shuili selected-pool validation:
    - Raw CER: `0.11587`
    - Minimal filler-normalized CER: `0.08198`
    - Hotword recall: `0.86721`
    - Base-hit hotwords lost: `44`
    - `unchanged_error=329`, `base_correct_broken=76`, `worsened_error=89`
  - Interpretation:
    - Generic/oral CER is essentially tied with the RAMC oral selected-pool result (`0.11593 -> 0.11587`), and minimal filler CER is slightly better (`0.08245 -> 0.08198`).
    - It does not solve hotword recall: recall remains around `0.867`.
- Probe B: hotword-first.
  - Training data:
    - `train_shuili_style_hotwordfirst_mix30k_20260705.qwen.jsonl`
    - `30000` train / `1500` dev rows.
    - RAMC rewrite `10080`, RAMC no-op `2520`, external hotword/no-op `18900`.
  - Start adapter:
    - `outputs/qwen35_cbwhisper_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_bf16/checkpoint-1000`
  - Output adapter:
    - `outputs/qwen35_ramc_shuili_style_hotwordfirst_sft160_from_hotwordaware_ckpt1000_bf16`
  - Training:
    - `160` steps, final eval loss `0.39357`.
  - Shuili selected-pool validation:
    - Raw CER: `0.13523`
    - Minimal filler-normalized CER: `0.09012`
    - Hotword recall: `0.91238`
    - Base-hit hotwords lost: `7`
    - Hotwords gained over base: `19`
    - `unchanged_error=683`, `base_correct_broken=23`, `worsened_error=47`
  - Interpretation:
    - This proves training can improve hotword recall: recall rises above the CB-Whisper base (`0.90154 -> 0.91238`) and base-hit loss drops sharply.
    - But it is far too conservative and weak at candidate use, so raw CER and filler-normalized CER regress badly.
    - Do not promote this adapter as a main Shuili model.
- Probe C: RAMC-hotword-aware balanced mix.
  - Code path:
    - Enable `--hotword-aware-ramc` so the RAMC oral synthetic rows themselves carry hotword evidence, rather than relying mainly on external hotword rows.
  - Training data:
    - `train_shuili_style_ramc_hotwordaware_mix30k_20260705.qwen.jsonl`
    - `30000` train / `1500` dev rows.
    - RAMC rewrite `17640`, RAMC no-op `4410`, external hotword/no-op `9450`.
  - Start adapter:
    - `outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Output adapter:
    - `outputs/qwen35_ramc_oral_hotwordaware_mix_sft160_from_ramc_oral_bf16`
  - Training:
    - `160` steps, final eval loss `1.21155`.
  - Shuili selected-pool validation:
    - Raw CER: `0.11644`
    - Minimal filler-normalized CER: `0.08286`
    - Hotword recall: `0.86721`
    - Base-hit hotwords lost: `45`
    - `unchanged_error=317`, `base_correct_broken=79`, `worsened_error=95`
  - Interpretation:
    - The balanced mix keeps the oral behavior close to RAMC oral, but still does not improve hotword recall.
    - Starting from the RAMC oral adapter seems to preserve its hotword-loss tendency; merely adding structured hotword evidence for 160 steps is insufficient.
- Current conclusion:
  - The recall/CER tradeoff is now well isolated:
    - Hotword-aware start solves recall but becomes conservative and leaves too many errors unchanged.
    - RAMC oral start solves oral candidate use better but loses hotwords.
  - A longer run of Probe A or C is not justified yet because the recall signal did not move.
  - The next paper-clean route should either:
    - start from the hotword-aware checkpoint and explicitly train candidate selection on harder RAMC oral rows to reduce `unchanged_error`; or
    - use a contrastive/preference objective where chosen outputs both lower CER and preserve protected hotwords, instead of full reference SFT that optimizes only one side of the tradeoff.

### 2026-07-05 4.35 起点继续训练：ChineseHP-style n-best selector/content-protect 全量

- User correction:
  - Do not run only small-step probing for this route.
  - Use the 4.35 hotword-preserve model as the starting point and run a full training pass.
  - Keep the data format close to the previous ChineseHP-style evidence format.
- Start adapter:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
- Training data:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_nbest_selector_content_protect_near2_hardx4_exactx2_base6k_noopx2.qwen.jsonl`
  - `69514` Qwen-message rows.
  - ChineseHP-style selector/content-protect format: exposes n-best candidates, protected hotwords, near-correct no-break rows, hard/exact repeats, and no-op anchors.
- Dev file for periodic eval:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/splits/train_nbest_selector_content_protect_near2_from435_dev1000.qwen.jsonl`
- Full training command state:
  - Initial `batch=4, grad_accum=7` run was stopped at about step `43/2483` because GPU memory reached about `30.5G/32.6G`, leaving too little OOM margin for long samples/eval.
  - Restarted safer full run:
    - `tmux` session: `covo_435_selector_full_bs3`
    - Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_selector_content_from_435_full1epoch_bs3ga9_bf16`
    - Log: `src/logs/train_covo_selector_content_from_435_full1epoch_bs3ga9_20260705_stdout.log`
    - Key hyperparameters: `epochs=1.0`, `lr=5e-7`, `max_length=1536`, `bf16`, `batch=3`, `grad_accum=9`, `save/eval every 500 steps`.
    - Effective batch remains close to the original (`27` vs `28` sequences per optimizer step), with lower memory pressure.
- Status at launch:
  - Model and adapter loaded successfully.
  - Dataset mapping started normally.
  - At the earlier 160-step probe settings, one optimizer step was about `20-21s`; full 1 epoch is expected to be a long run.
- Planned validation after training:
  - Evaluate on Shuili clean n-best candidate pool first, especially the `t046_keep5_clean_shuili_repeat_len18` pool.
  - Compare against the 4.35 base adapter and RAMC oral/hotword-aware probes on raw CER, minimal filler-normalized CER, hotword recall, base-hit hotword loss, and `unchanged_error`.

#### Completed validation: Shuili `t046_keep5_clean_shuili_repeat_len18`

- Final training status:
  - Completed full `1.0` epoch: `2575/2575` steps.
  - Final eval loss: `0.287788`.
  - Train loss: `0.4716`.
  - Runtime: about `14h53m`.
  - Final adapter saved at:
    - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_selector_content_from_435_full1epoch_bs3ga9_bf16`
- Validation files:
  - Messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_from435_full1epoch_bs3ga9_messages_20260706.jsonl`
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_from435_full1epoch_bs3ga9_predictions_20260706.jsonl`
  - Eval log: `src/logs/eval_shuili_t046_clean_shuili_repeat_len18_from435_full1epoch_bs3ga9_20260706_stdout.log`
  - Filler CER: `src/logs/filler_normalized_cer_shuili_v3_t046_clean_shuili_repeat_len18_from435_full1epoch_bs3ga9_minfillers_20260706.json`
  - Error audit: `src/logs/covo_error_audit_shuili_v3_t046_clean_shuili_repeat_len18_from435_full1epoch_bs3ga9_20260706_summary.json`
- Main metrics:
  - Raw CER: `0.13479`
  - Baseline raw CER on this pool: `0.15415`
  - Minimal filler-normalized CER: `0.08723`
  - Minimal filler-normalized baseline CER: `0.09814`
  - Hotword recall: `0.92141`
  - Base-hit hotwords lost: `4`
  - Hotwords gained over base: `26`
  - `unchanged_error=726`, `fixed_to_exact=38`, `improved_partial=107`, `base_correct_broken=21`, `worsened_error=24`.
- Comparison:
  - RAMC oral adapter on the same pool:
    - Raw CER `0.11593`, hotword recall `0.86631`, `unchanged_error=321`, base-hit hotwords lost `46`.
  - Hotword-first 160-step adapter:
    - Raw CER `0.13523`, hotword recall `0.91238`, `unchanged_error=683`, base-hit hotwords lost `7`.
  - This full 4.35-start selector model:
    - Improves hotword preservation/recall further than hotword-first (`0.92141` recall, only `4` base-hit losses).
    - But it remains too conservative for Shuili oral correction: raw CER is essentially tied with hotword-first and far worse than RAMC oral.
- Interpretation:
  - Full ChineseHP-style selector/content-protect training from the 4.35 hotword-preserve model teaches hotword protection well.
  - It does not teach enough Shuili-style oral correction behavior, so many reachable candidate improvements are ignored.
  - Do not promote this adapter as the main Shuili CER model.
  - It is useful evidence for the recall/CER tradeoff: AISHELL hotword-preserve supervision transfers hotword use, but not domain-specific oral correction.

### 2026-07-06 Lower `unchanged_error`: continue from 4.35 full selector on Shuili-style oral mix

- Motivation:
  - The full 4.35-start selector model reached high hotword recall (`0.92141`) but was too conservative on Shuili:
    - Raw CER `0.13479`
    - `unchanged_error=726`
  - Goal for this run is to reduce no-op/unchanged behavior by continuing training on rewrite-heavy oral data.
- Start adapter:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_selector_content_from_435_full1epoch_bs3ga9_bf16`
- Training data:
  - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/train_shuili_style_oral_hotword_mix30k_20260705.qwen.jsonl`
  - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/dev_shuili_style_oral_hotword_mix1500_20260705.qwen.jsonl`
  - This mix previously showed much lower unchanged behavior when starting from the RAMC oral adapter, while retaining AISHELL hotword/no-op anchors.
- Training command state:
  - `tmux` session: `covo_435_unfreeze_oral_full`
  - Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_from435_fullselector_oral_hotword_mix_full1epoch_bf16`
  - Log: `src/logs/train_covo_from435_fullselector_oral_hotword_mix_full1epoch_20260706_stdout.log`
  - Key hyperparameters: `epochs=1.0`, `lr=1e-6`, `max_length=1536`, `bf16`, `batch=3`, `grad_accum=9`, `save/eval every 250 steps`.
- Expected effect:
  - Lower `unchanged_error` substantially compared with `726`.
  - Watch for hotword recall regression versus `0.92141`; acceptable only if CER/unchanged improve enough.
- Planned validation:
  - Same Shuili pool as previous run:
    - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
  - Same metrics: raw CER, minimal filler-normalized CER, hotword recall, base-hit hotword loss, `unchanged_error`.

#### Completed validation

- Final training status:
  - Completed full `1.0` epoch: `1112/1112` steps.
  - Final eval loss: `0.268509`.
  - Train loss: `0.4006`.
  - Runtime: about `6h03m`.
- Validation files:
  - Messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_from435_oral_hotword_mix_full1epoch_messages_20260706.jsonl`
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_from435_oral_hotword_mix_full1epoch_predictions_20260706.jsonl`
  - Eval log: `src/logs/eval_shuili_t046_clean_shuili_repeat_len18_from435_oral_hotword_mix_full1epoch_20260706_stdout.log`
  - Filler CER: `src/logs/filler_normalized_cer_shuili_v3_t046_clean_shuili_repeat_len18_from435_oral_hotword_mix_full1epoch_minfillers_20260706.json`
  - Error audit: `src/logs/covo_error_audit_shuili_v3_t046_clean_shuili_repeat_len18_from435_oral_hotword_mix_full1epoch_20260706_summary.json`
- Main metrics:
  - Raw CER: `0.13345`
  - Minimal filler-normalized CER: `0.08652`
  - Hotword recall: `0.91960`
  - Base-hit hotwords lost: `4`
  - Hotwords gained over base: `24`
  - `unchanged_error=702`, `fixed_to_exact=43`, `improved_partial=121`, `base_correct_broken=23`, `worsened_error=29`.
- Comparison vs previous full 4.35 selector:
  - Raw CER: `0.13479 -> 0.13345`
  - Minimal filler-normalized CER: `0.08723 -> 0.08652`
  - Hotword recall: `0.92141 -> 0.91960`
  - `unchanged_error: 726 -> 702`
  - Base-hit hotword loss remains `4`.
- Interpretation:
  - Continuing on rewrite-heavy Shuili-style oral data does reduce no-op behavior, but only modestly.
  - The hotword-preserve prior is still dominant; the model remains much more conservative than the RAMC oral adapter (`unchanged_error=321`, raw CER `0.11593`).
  - This route alone is insufficient for the desired Shuili CER drop. A stronger intervention is needed: either more Shuili-like rewrite data with less no-op anchoring, or a staged/merged approach that starts from RAMC oral and adds targeted hotword preservation without reintroducing no-op dominance.

### 2026-07-06 Aborted RAMC-aggressive continuation

- User question:
  - Whether we can start from RAMC oral and make the model more willing to edit.
- Attempted run:
  - Start adapter: `outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Train file: `data/processed/ramc_oral/train_shuili_style_oral_hotword_mix30k_20260705.qwen.jsonl`
  - Output dir: `outputs/qwen35_ramc_oral_aggressive_shuili_hotword_mix_full1epoch_bf16`
  - Initial hyperparameters: `lr=2e-6`, `batch=3`, `grad_accum=9`, `max_length=1536`, full `1 epoch`.
- Status:
  - Stopped early at user request before any usable checkpoint/validation.
  - Reason for stopping: analyze RAMC oral errors first. Lowering `unchanged_error` blindly is risky because RAMC oral already edits aggressively and its main errors include hotword loss, base-correct breakage, hallucinated prefixes/fillers, and occasional repeated-span generation.

### 2026-07-06 Phonetic hotword-span evidence probe on Shuili

- Motivation:
  - RAMC oral errors show many professional-term and near-homophone mistakes, e.g. `岩基/研机`, `坝肩/霸建`, `开挖/开发`, `工种/公主`.
  - Instead of adding a gate or hard patch, construct a paper-style structured evidence prompt:
    - candidate local spans with similar pinyin;
    - `copy_term` targets for supported domain terms;
    - candidate keep/drop evidence for each term.
- New script:
  - `src/analysis/build_phonetic_hotword_span_sft.py`
  - It converts CB-Whisper/COVO candidate-pool JSONL into Qwen messages with a `Phonetic hotword span evidence` section.
  - Important safeguard: for the current training-data probe, `copy_term` is restricted to terms that appear in the reference text. This avoids teaching false hotword insertion from KWS/prompt false positives.
  - Added `--compact-prompt` after the first full-evidence prompt caused OOM and was too noisy.
- Data:
  - Source pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
  - Compact train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/train_shuili_phonetic_hotword_span_refonly_compact_probe900_20260706.qwen.jsonl`
  - Compact dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/dev_shuili_phonetic_hotword_span_refonly_compact_probe252_20260706.qwen.jsonl`
  - Generated `575` rows with phonetic evidence: `323` train / `252` dev, `577` skipped.
  - Quality check: all `copy_term` targets appear in the assistant/reference answer; no false-hotword target remained after the ref-only safeguard.
- Training:
  - Start adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_phonetic_hotword_span_refonly_compact_probe_3epoch_bf16`
  - Log: `src/logs/train_covo_ramc_oral_phonetic_hotword_span_refonly_compact_probe_3epoch_20260706_stdout.log`
  - First attempt with full CB-Whisper evidence + phonetic evidence at `max_length=2048` OOMed.
  - Second attempt with compact prompt still OOMed without gradient checkpointing.
  - Final successful config: `max_length=1024`, `epochs=3`, `lr=1e-6`, `batch=4`, `grad_accum=7`, `bf16`, `gradient_checkpointing=True`.
  - Final train loss `2.101`; eval loss improved to `1.951`.
- Dev mechanism validation:
  - Prediction file: `src/logs/shuili_phonetic_hotword_span_refonly_compact_probe252_predictions_20260706.jsonl`
  - Same-input RAMC oral control: `src/logs/shuili_phonetic_hotword_span_refonly_compact_probe252_ramc_oral_predictions_20260706.jsonl`
  - ASR top1 baseline on this dev subset: CER `0.16303`.
  - RAMC oral with compact phonetic prompt: CER `0.11543`, `110` improved / `37` worsened / `105` unchanged.
  - Continued phonetic-span adapter: CER `0.11622`, `111` improved / `37` worsened / `104` unchanged.
  - Direct comparison with RAMC oral on the same dev rows:
    - New adapter better on `4` rows.
    - New adapter worse on `3` rows.
    - Same edit distance on `245` rows.
    - Exact same normalized output on `244` rows.
- Interpretation:
  - The structured phonetic evidence prompt itself is useful and gives the model local term-disambiguation information.
  - The 323-row continuation is too small and too close to the existing RAMC behavior to materially change the adapter.
  - This Shuili ref-only run is a mechanism probe only, not a publishable held-out result, because reference text was used to select true `copy_term` targets.
  - A fair/paper-usable next step would construct similar phonetic evidence from a real train split, then test with terms coming only from KWS/prompt/candidate evidence; it also needs false-hotword negative examples so the model learns when not to copy a near-homophone term.

### 2026-07-06 AISHELL diff-based phonetic span data

- Motivation:
  - User requested constructing the same kind of phonetic/homophone correction cases from AISHELL, instead of relying only on Shuili reference-only probe data.
  - The first automatic reference n-gram attempt was too noisy: it created unnatural targets such as arbitrary middle substrings and many weak spans.
- Method update:
  - Extended `src/analysis/build_phonetic_hotword_span_sft.py` with `--diff-reference-terms`.
  - This mode compares each N-best candidate with the reference using character-level diff and keeps only `replace` spans where:
    - reference target length is `2-4`;
    - candidate span length is at least `2`;
    - length gap is at most `1`;
    - pinyin edit distance is at most `1`.
  - This directly extracts real candidate errors such as `葫芦 -> 胡润`, `羊起 -> 央企`, `编辑 -> 边际`, `未免 -> 卫冕`, `索性 -> 所幸`, `父请/付钱 -> 父亲`.
- Source:
  - `src/logs/cbwhisper_covo_evidence_train_full.jsonl`
  - Total source rows: `17301`.
- Output data:
  - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_aishell_phonetic_hotword_span_diff_ref_20260706.qwen.jsonl`
  - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_aishell_phonetic_hotword_span_diff_ref_20260706.qwen.jsonl`
  - Generated: `8459` train / `1000` dev / `9459` total.
  - Skipped: `7842` rows without a qualifying near-phonetic replace span.
- Data stats:
  - Unique replacement targets: `11764`.
  - Unique candidate->target pairs: `20799`.
  - Average compact user prompt length: about `1008` chars.
  - P95 compact user prompt length: about `1711` chars.
- Frequent targets/pairs:
  - Targets: `城市`, `对于`, `会徽`, `世锦`, `实现`, `技术`, `时间`, `经济`, `卫冕`, `市场`, `计划`, `楼市`, `视频`, `纪录`, `将于`.
  - Pairs include `程式 -> 城市`, `未免 -> 卫冕`, `城郊 -> 成交`, `基础/计数 -> 技术`, `市井 -> 世锦`, `卉卉 -> 会徽`, `秉天 -> 炳添`, `索性 -> 所幸`, `调导 -> 钓岛`, `付钱/负请 -> 父亲`.
- Caveat:
  - Because Whisper/Candidate text often uses traditional Chinese, this data also contains many simplification-normalization examples such as `對於 -> 对于`, `經濟 -> 经济`, `時間 -> 时间`.
  - This is acceptable if we want the COVO module to normalize output style, but if we want a pure homophone-professional-term experiment, a later filter should remove pairs whose only difference is traditional/simplified form.

#### Expanded/focused AISHELL phonetic data

- User request:
  - Make more data from AISHELL once diff-based construction proved convenient.
- Additional variants generated:
  - Expanded utterance-level:
    - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_aishell_phonetic_hotword_span_diff_ref_expanded_20260706.qwen.jsonl`
    - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_aishell_phonetic_hotword_span_diff_ref_expanded_20260706.qwen.jsonl`
    - Rows: `8765` train / `1500` dev / `10265` total.
    - Settings: target len `2-5`, pinyin distance `<=2`, max length gap `2`, max evidence terms `16`.
  - Focused evidence rows:
    - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_aishell_phonetic_hotword_span_diff_ref_focused_20260706.qwen.jsonl`
    - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_aishell_phonetic_hotword_span_diff_ref_focused_20260706.qwen.jsonl`
    - Rows: `15636` train / `1500` dev / `17136` total.
    - Construction: same expanded settings, but `--explode-evidence --max-exploded-per-row 4`, so each row focuses on one local phonetic target.
  - Focused-large evidence rows:
    - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_aishell_phonetic_hotword_span_diff_ref_focused_large_20260706.qwen.jsonl`
    - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_aishell_phonetic_hotword_span_diff_ref_focused_large_20260706.qwen.jsonl`
    - Rows: `15333` train / `2000` dev / `17333` total.
    - Construction: `--explode-evidence --max-exploded-per-row 8`.
- Focused-large stats:
  - Unique candidate->target pairs: `23886`.
  - Unique replacement targets: `13099`.
  - Average user prompt length: about `833` chars.
  - P95 user prompt length: about `1118` chars.
  - Max user prompt length: `1331` chars.
- Common pair types:
  - Homophone/entity/content: `空间 -> 攻坚`, `搜狗 -> 收购`, `空废/空肺 -> 控费`, `旧信/旧姓 -> 救性`, `索性 -> 所幸`, `付钱/负请 -> 父亲`.
  - Number normalization: `20 -> 二十`, `12 -> 十二`, `200 -> 二百`, `2000 -> 两千`.
  - Simplified normalization: `對於 -> 对于`, `這個 -> 这个`, `時間 -> 时间`, `經濟 -> 经济`.
- Recommended training use:
  - Use `focused_large` as the high-volume phonetic local-correction set.
  - Mix with previous no-op/hotword-preserve data rather than training on it alone; otherwise the model may become too eager to rewrite numbers/traditional forms.

### 2026-07-06 Non-overlap data expansion after duplication check

- User correction:
  - Do not simply reuse the earlier partially/full-trained data; generate from other data where possible.
- Duplication audit:
  - `train_oral_rewrite_full157k_hn2.qwen.jsonl` has `157000` base ids.
  - `train_full_raw.jsonl` also has `157000` base ids; they are the same RAMC oral source rows.
  - `dev_oral_rewrite_full5k_hn2.qwen.jsonl` and `dev_full_raw.jsonl` likewise both have `5000` base ids.
  - Therefore the previously generated `train_aishell_phonetic_oral_local_mix100k_20260706.qwen.jsonl` would have repeated the old full RAMC oral training source; it was deleted and should not be used.
- Script changes:
  - Added `src/analysis/build_aishell_synthetic_phonetic_sft.py`.
  - Extended `src/analysis/build_ramc_oral_rewrite_data.py` with `--exclude-jsonl`; `--train-size <= 0` now means use all remaining records after dev split.
  - Added `src/analysis/build_oral_diff_span_sft.py` for local oral insert/delete/replace evidence.
- New AISHELL synthetic phonetic/hotword data:
  - Source: full AISHELL transcript `datasets/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt`.
  - Exclusions: previous `aishell_train_hotword_noop.qwen.jsonl` plus hotword dev/test text ids.
  - Construction: mine same-pinyin word pairs from unused AISHELL train text, replace one reference word with a confusable word, then expose `protected_hotwords`, `prompt_hotwords`, `kws_hotwords`, and local pinyin evidence.
  - Clean version uses `--min-word-freq 3` to avoid rare/noisy words.
  - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/train_aishell_unused_synthetic_phonetic_hotword_freq3_20260706.qwen.jsonl`
  - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/chinesehp_aishell1/dev_aishell_unused_synthetic_phonetic_hotword_freq3_20260706.qwen.jsonl`
  - Rows: `58000` train / `2000` dev.
  - Overlap with previous AISHELL no-op ids: `0`.
  - Example pairs: `注意 -> 主意`, `就业 -> 酒业`, `回乡 -> 回想`, `作为 -> 座位`, `工具 -> 共聚`.
- Truly unused RAMC oral supplement:
  - Excluding old RAMC train/dev leaves very little usable text: only `328` records when allowing up to `160` normalized chars.
  - Local-edit SFT rows from that non-overlap subset:
    - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/train_oral_unused_len160_local_edit_20260706.qwen.jsonl` (`636` rows).
    - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/dev_oral_unused_len160_local_edit_20260706.qwen.jsonl` (`270` rows).
- Mixed non-overlap training set:
  - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/mixed/train_aishell_unused_phonetic_plus_oral_unused_20260706.qwen.jsonl`
  - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/mixed/dev_aishell_unused_phonetic_plus_oral_unused_20260706.qwen.jsonl`
  - Rows: `58636` train / `2270` dev.
  - Intended use: continue from a strong Shuili/COVO checkpoint to teach local phonetic hotword recovery without repeating old RAMC full training data.
- Training run started:
  - tmux session: `covo_unused_phonetic_0706`.
  - Base adapter: `outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`.
  - Output adapter: `outputs/qwen35_ramc_oral_unused_phonetic_mix_full1epoch_bf16`.
  - Log: `src/logs/train_covo_ramc_oral_unused_phonetic_mix_full1epoch_20260706_stdout.log`.
  - Settings: 1 epoch, `2095` steps, LR `1e-6`, max length `1024`, batch `4`, grad accumulation `7`, bf16, gradient checkpointing.
  - Follow-up queued on user request:
    - tmux session: `covo_unused_phonetic_0706_round2`, waiting for the first session to finish.
    - Base adapter: `outputs/qwen35_ramc_oral_unused_phonetic_mix_full1epoch_bf16`.
    - Output adapter: `outputs/qwen35_ramc_oral_unused_phonetic_mix_2epoch_bf16`.
    - Log: `src/logs/train_covo_ramc_oral_unused_phonetic_mix_second_epoch_20260707_stdout.log`.
    - Settings: another 1 epoch on the same non-overlap mix, LR `5e-7`.

#### First-epoch inspection on Shuili

- User stopped the queued second epoch and asked to inspect the first epoch first.
- Action:
  - Stopped the second epoch before it produced a completed model.
  - Evaluated `outputs/qwen35_ramc_oral_unused_phonetic_mix_full1epoch_bf16` on the existing Shuili textrewrite messages:
    - Input messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_textrewrite_messages_20260705.jsonl`
    - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_unused_phonetic_epoch1_predictions_20260707.jsonl`
    - Eval log: `src/logs/eval_shuili_t046_clean_shuili_repeat_len18_unused_phonetic_epoch1_20260707_stdout.log`
- Training health:
  - First epoch completed normally at `2095` steps.
  - Final train loss: about `0.3549`.
  - Dev eval loss on the synthetic/non-overlap mix: `0.269998`.
  - Adapter was saved successfully.
- Shuili result, raw CER:
  - RAMC oral baseline adapter: `0.115929`.
  - New unused-phonetic first epoch: `0.118977`.
  - Conclusion: worse by about `+0.00305` absolute CER.
- Shuili result, minimal-filler CER (`呃,呢,啊,嗯` removed):
  - RAMC oral baseline adapter: `0.082453`.
  - New unused-phonetic first epoch: `0.085076`.
  - Conclusion: worse by about `+0.00262` absolute CER.
- Error/audit comparison:
  - Exact predictions decreased: `409 -> 393`.
  - Base-correct-broken increased: `74 -> 90`.
  - Worsened-error increased: `95 -> 107`.
  - Unchanged-error decreased: `321 -> 292`, so the model became more willing to edit, but the extra edits are not reliable enough.
  - Hotword recall decreased: `0.86631 -> 0.86269`; base-lost-by-pred increased `46 -> 50`.
- Interpretation:
  - The non-overlap AISHELL synthetic phonetic data did teach the model to make more edits, but it did not transfer cleanly to Shuili oral/professional audio.
  - The generated same-pinyin replacements are too AISHELL/news-style and too “single local word” oriented, while Shuili errors are dominated by oral filler, domain-term preservation, and candidate evidence reliability.
  - Do not continue this exact second epoch as a main route unless the data is rebalanced with stronger no-op/protect/domain-term examples or evaluated only as an ablation.

### 2026-07-07 Shuili domain-term knowledge-base route

- Motivation:
  - Recent Shuili experiments show that more blind COVO training is not enough: the model becomes more willing to edit, but often edits the wrong term or loses domain terms.
  - The next paper-friendly route is to expose structured domain knowledge rather than add a hard gate or a case-specific post-processing patch.
  - The deployable version must not use held-out references. Reference-derived term/confusion information is allowed only for diagnostic/oracle analysis.
- New script:
  - `src/analysis/build_domain_term_kb.py`
  - It builds a domain-term KB from CB-Whisper evidence JSONL.
  - Default mode is non-leakage: terms come from KWS/prompt/candidate evidence fields, including `hotwords`, `prompt_hotwords`, `keyword_mentions`, candidate `consensus_keywords`, and candidate exact keyword matches.
  - Optional `--include-reference` is diagnostic only and mines reference-side hit/confusable spans; do not use that file for formal held-out validation.
- Source evidence:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
  - Rows: `1152`.
- Generated non-leak KB:
  - `src/logs/shuili_domain_term_kb_nonleak_20260707.jsonl`
  - `src/logs/shuili_domain_term_kb_nonleak_20260707_summary.json`
  - Terms: `124`.
  - Top supported terms include `施工`, `地基`, `闸门`, `水利`, `土石`, `石方`, `填筑`, `基坑`, `水量`, `开挖`, `钻孔爆破`, `水利工程`, `导流`, `水轮机`, `混凝土`.
- Generated diagnostic KB:
  - `src/logs/shuili_domain_term_kb_diagnostic_ref_20260707.jsonl`
  - `src/logs/shuili_domain_term_kb_diagnostic_ref_20260707_summary.json`
  - Terms: `124`.
  - Adds `reference_hit` and candidate/reference confusables. Examples include `地基 -> 低级/第几/第一`, `闸门 -> 扎/栅木`, `填筑 -> 天主/潜住/浅重`, `开挖 -> 开发`, `水闸 -> 杂/閘/沙/灶`.
  - Some mined confusables are noisy because they come from character diff spans over imperfect candidates, so this is mainly for error analysis and later cleaning.
- Recommended next experiment:
  - First do inference-only KB injection with the current best RAMC oral/COVO model; no retraining yet.
  - For each sample, retrieve compact KB evidence from non-leak sources using KWS support, candidate consensus/exact evidence, and pinyin similarity to unstable n-best spans.
  - Prompt COVO with: protected domain terms, likely confusable forms, and a warning that unsupported KB terms must not be forced into the output.
  - If this improves Shuili CER/recall without hurting base-correct cases, then convert the same KB-evidence format into SFT data. If it fails, the issue is likely candidate/evidence reliability rather than model capacity.

#### Inference-only KB injection probe

- New script:
  - `src/analysis/inject_domain_kb_to_covo_messages.py`
  - It injects a compact `Domain KB evidence` section into existing COVO/Qwen messages.
  - Default mode is conservative and non-leak: only terms already supported by ASR top-1, N-best, candidate exact matches, or candidate consensus are injected.
  - `--include-absent-hotwords` reproduces the broader ablation where prompt/KWS terms absent from all candidates are also shown as weak evidence; this is useful diagnostically but can make the model too conservative.
- Base model for both probes:
  - Adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`
  - Input messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_textrewrite_messages_20260705.jsonl`
  - Non-leak KB: `src/logs/shuili_domain_term_kb_nonleak_20260707.jsonl`
- Baseline same-input RAMC oral result:
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_ramc_oral_predictions_20260705.jsonl`
  - Raw CER: `0.115929`
  - Minimal filler CER (`呃,呢,啊,嗯` removed): `0.082453`
  - Hotword recall: `0.86631`
  - Delta counts: `fixed_to_exact=226`, `unchanged_error=321`, `base_correct_broken=74`, `worsened_error=95`.
- Probe A, broad KB with absent prompt/KWS terms included as weak evidence:
  - Messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_messages_20260707.jsonl`
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_ramc_oral_predictions_20260707.jsonl`
  - Raw CER: `0.117326`
  - Minimal filler CER: `0.077678`
  - Hotword recall: `0.88708`
  - Delta counts: `fixed_to_exact=180`, `unchanged_error=413`, `base_correct_broken=44`, `worsened_error=68`.
  - Interpretation: KB evidence substantially reduces destructive edits and improves hotword recall, but the model becomes too conservative; raw CER regresses because many originally fixable errors are left unchanged.
- Probe B, supported-only KB terms:
  - Messages: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_messages_20260707.jsonl`
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_ramc_oral_predictions_20260707.jsonl`
  - Raw CER: `0.116183`
  - Minimal filler CER: `0.076199`
  - Hotword recall: `0.87986`
  - Delta counts: `fixed_to_exact=180`, `unchanged_error=413`, `base_correct_broken=48`, `worsened_error=62`.
  - Interpretation: supported-only KB is better than broad KB and gives the best minimal-filler CER so far on this Shuili setting, but raw CER is still slightly worse than RAMC oral. The method is useful as domain-term/preservation evidence, but prompt-only KB injection is not enough to lower raw CER; the next version should train on this exact KB-evidence format so the model learns to use it without losing edit aggressiveness.

### 2026-07-07 Shuili CER decomposition and sub-10 engineering result

- User concern:
  - Hotword recall is still around the mid/high 80s, but raw CER remains high; therefore many errors may be outside hotwords.
- New diagnostic script:
  - `src/analysis/decompose_cer_errors.py`
  - It aligns prediction/reference at character level and decomposes CER edits into:
    - `true_hotword_region`: edits overlapping true `keyword_mentions`;
    - `filler_region`: edits overlapping `呃/呢/啊/嗯`;
    - `non_hotword_region`: all other errors.
- Decomposition, ASR top-1 baseline:
  - Prediction field: `input.asr_top1`.
  - CER: `0.154149`, edits `2428`.
  - Hotword-region edits: `241` (`9.93%`).
  - Filler edits: `767` (`31.59%`).
  - Non-hotword edits: `1420` (`58.48%`).
- Decomposition, RAMC oral COVO:
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_ramc_oral_predictions_20260705.jsonl`
  - CER: `0.115929`, edits `1826`.
  - Hotword-region edits: `239` (`13.09%`).
  - Filler edits: `437` (`23.93%`).
  - Non-hotword edits: `1150` (`62.98%`).
  - Conclusion: most remaining CER is outside true hotword spans.
- Decomposition, KB supported-only COVO:
  - Predictions: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_ramc_oral_predictions_20260707.jsonl`
  - CER: `0.116183`, edits `1830`.
  - Hotword-region edits: `227` (`12.40%`).
  - Filler edits: `509` (`27.81%`).
  - Non-hotword edits: `1094` (`59.78%`).
- Main non-hotword error sources:
  - Traditional/simplified mismatch: `们/們`, `个/個`, `这/這`, `进/進`, `对/對`, `较/較`.
  - Function-word insertion/deletion: `的`, `个`, `这`, `一`, `了`, `那么`.
  - Repetition/hallucinated phrase: e.g. repeated `那么/这里`.
  - Non-hotword domain/common confusions: `分/封`, `它/他`, `参建/参见`, `重力式/重力是`, `围海/为海`, `堤防/敌方`, `挖运方案/Volume 5`.
- Evaluation normalization probe:
  - `src/analysis/evaluate_filler_normalized_cer.py --fillers ''` uses OpenCC `t2s`, so this is “繁简统一 only”, not filler removal.
  - RAMC oral after OpenCC-only CER: `0.100882`.
  - KB supported-only after OpenCC-only CER: `0.100121`.
  - Interpretation: a large part of the raw CER above 10% is writing-system mismatch, not acoustic/semantic ASR failure.
- New diagnostic/engineering normalizer:
  - `src/analysis/normalize_covo_predictions.py`
  - Repro command used:
    - `--opencc-t2s --collapse-repeats --shuili-domain-normalize`
  - Input: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_ramc_oral_predictions_20260707.jsonl`
  - Output: `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_ramc_oral_norm_predictions_20260707.jsonl`
  - Changed rows: `75`.
  - Raw CER after normalization: `0.099295`, below 10%.
  - Decomposition after normalization:
    - Edits: `1564`.
    - Hotword-region edits: `214` (`13.68%`).
    - Filler edits: `509` (`32.54%`).
    - Non-hotword edits: `841` (`53.77%`).
- Important interpretation:
  - The sub-10 result is real under the current evaluator, but it uses an engineering output-normalization layer. It is useful as a diagnostic and possible deployment baseline.
  - For the paper route, the cleaner claim should be: Shuili CER is dominated by non-hotword writing-style, filler, and ordinary lexical errors; future COVO/KB training should teach simplified output, repetition avoidance, and common domain-term normalizations directly, rather than relying on a hand-written normalizer.

#### Non-leak SFT data for learning the normalization behavior

- Motivation:
  - The engineering normalizer gets Shuili below 10% CER, but it is not a clean paper method.
  - Convert the discovered behavior into model training: simplified output, repetition suppression, function-word preservation, and common non-hotword domain lexical corrections.
- New script:
  - `src/analysis/build_shuili_norm_domain_sft.py`
  - It does not use Shuili test references.
  - Sources:
    - RAMC oral train raw: `covo/data/processed/ramc_oral/train_full_raw.jsonl`
    - Non-leak Shuili term KB: `src/logs/shuili_domain_term_kb_nonleak_20260707.jsonl`
  - Synthetic corruptions:
    - simplified target vs traditional ASR;
    - repeated local spans;
    - deleted/inserted function words and oral particles;
    - domain lexical confusions such as `参建单位/参见单位`, `重力式码头/重力是码头`, `挖运方案/Volume 5`, `开挖/开发`, `土石堤防/土石敌方`, `围海造田/为海造田`.
  - The prompt explicitly includes the output norm: default simplified Chinese, collapse clear repetition, preserve context-supported oral/function words, and do not insert unsupported domain terms.
- Generated data:
  - Train: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/train_shuili_norm_domain_sft60k_20260707.qwen.jsonl`
  - Dev: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/ramc_oral/dev_shuili_norm_domain_sft3k_20260707.qwen.jsonl`
  - Rows: `60000` train / `3000` dev.
  - Source mix in train:
    - RAMC oral synthetic normalization rows: `44135`.
    - Shuili-domain synthetic rows from non-leak KB templates: `15865`.
  - Tag counts:
    - `traditional=28905`
    - `delete_function=14903`
    - `repeat=11275`
    - `noop_simplified_target=11102`
    - `insert_function=9341`
    - `domain_noop=3406`
    - `domain_confusion=1795`
- Planned training:
  - Continue from `outputs/qwen35_ramc_oral_rewrite_full1epoch_from_chinesehp_bf16`.
  - Output: `outputs/qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`.
  - Settings: 1 epoch, LR `1e-6`, max length `1024`, batch `4`, gradient accumulation `7`, bf16, gradient checkpointing.

#### Shuili normalization/domain SFT result

- Training completed:
  - Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`
  - Train runtime: about `6:04:02`.
  - Train loss: `0.2971`.
  - Eval loss: `0.2117`.
- Evaluation input:
  - `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_domainkb_supported_messages_20260707.jsonl`
- Predictions:
  - `src/logs/shuili_v3_t046_clean_shuili_repeat_len18_normdomain_sft60k_predictions_20260707.jsonl`
- Raw evaluation:
  - CER: `0.115294`.
  - Baseline ASR CER: `0.154149`.
  - Improved samples: `415`.
  - Worsened samples: `90`.
  - Unchanged samples: `647`.
- Filler-normalized evaluation, removing `呃/呢/啊/嗯`:
  - Base CER: `0.098998`.
  - Prediction CER: `0.073778`.
  - Prediction exact samples: `643`.
- Error audit:
  - N-best oracle CER: `0.053330`.
  - Prediction-or-oracle CER: `0.052568`.
  - `fixed_to_exact=178`.
  - `unchanged_error=430`.
  - `base_correct_broken=40`.
  - `worsened_error=50`.
  - Hotword base recall: `0.901536`.
  - Hotword prediction recall: `0.884372`.
- CER decomposition:
  - Edits: `1816`.
  - Hotword edits: `228` (`12.56%`).
  - Filler edits: `528` (`29.07%`).
  - Non-hotword edits: `1060` (`58.37%`).
- Interpretation:
  - The model learned some normalization behavior: raw CER is slightly better than RAMC oral (`0.115929`) and KB-supported prompt (`0.116183`), and filler-normalized CER improves clearly to `0.073778`.
  - It did not solve the raw sub-10 target. The main problem is still non-hotword/filler-heavy oral style plus conservative unchanged errors.
  - Hotword recall drops from the input-side level, so further training should explicitly protect supported hotwords while increasing willingness to fix non-hotword errors.

#### Shuili CER error analysis after ignoring hotword priority

- Current conclusion:
  - Hotword errors are not the main raw CER bottleneck.
  - In `normdomain_sft60k`, CER edits decompose as:
    - Hotword-region edits: `228` (`12.56%`).
    - Filler edits: `528` (`29.07%`).
    - Non-hotword edits: `1060` (`58.37%`).
  - The remaining target should be ordinary ASR correction, oral-style preservation, and better N-best use.
- Normalization probes, using an internal edit-distance diagnostic:
  - Raw diagnostic CER: about `0.11904`.
  - Traditional-to-simplified only: about `0.10317`.
  - Filler removal only (`呃/呢/啊/嗯`): about `0.09463`.
  - Traditional-to-simplified + filler removal: about `0.07775`.
  - Traditional-to-simplified + filler removal + simple repeat collapse: about `0.07503`.
  - Interpretation: raw CER above 10 is heavily driven by output style mismatch and oral filler handling, not by domain-term errors alone.
- Top raw error types:
  - Filler deletion/substitution: many `呢` deletions and `呢->的/了`.
  - Traditional/simplified mismatch: `们/們`, `么/麼`, `个/個`, `这/這`, `话/話`, `对/對`, `进/進`.
  - Function-word instability: deletion/insertion of `的`, `个`, `这`, `一`, `了`.
  - Non-hotword lexical confusions: `分/封`, `它/他`, `运/用`, `浇/交`, `挖/发`, `施/时`, `场/厂`, `筑/流`.
- N-best utilization problem:
  - After simplified + filler + space normalization:
    - `nbest_exact_pred_wrong=268`.
    - `nbest_better_than_pred=395`.
    - `pred_same_error=325`.
    - `pred_worse_base=95`.
  - Exact-answer rank among the 268 missed cases:
    - rank 1: `59`
    - rank 2: `68`
    - rank 3: `53`
    - rank 4-5: `36`
    - rank >=6: `52`
  - This means many errors are not candidate-generation failures. The right answer is already in N-best, often in top 3, but COVO keeps or lightly edits the top-1 instead of selecting the better candidate.
- Length-error pattern after simplified + filler + no-space normalization:
  - same length wrong rows: `216`.
  - prediction short by 1-2 chars: `129`.
  - prediction long by 1-2 chars: `115`.
  - prediction short by 3-5 chars: `35`.
  - prediction short by 6+ chars: `5`.
  - Key long-deletion examples are missing clauses already present in a lower-ranked N-best candidate, e.g. `水利工程的地基尤其水利工程的地基问题更为复杂`.
- Next clean route:
  - Stop optimizing hotword behavior for now.
  - Train a COVO candidate-selection/rewrite objective that explicitly learns:
    - prefer simplified Chinese output;
    - preserve oral fillers when supported by N-best;
    - use lower-ranked N-best when it repairs a missing clause or function-word pattern;
    - avoid free-form additions when base is already exact.
  - The most promising supervision is oracle-candidate SFT / preference data from datasets where references exist, because Shuili already has many oracle candidates in N-best.

#### Shuili converted to clean COVO format

- New converter:
  - `src/analysis/convert_shuili_to_covo_format.py`
- Purpose:
  - Convert the current Shuili CB-Whisper candidate pool into COVO/Qwen message format without hotword prompts.
  - The prompt focuses on ordinary CER reduction: N-best, pinyin, stable/uncertain spans, and confusable candidates.
  - It explicitly tells the model not to delete oral particles for written style and not to freely add prefixes/suffixes unsupported by N-best.
- Source candidate pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
- Generated COVO data:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_clean.qwen.jsonl`
  - Summary: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_clean.summary.json`
- Statistics:
  - Rows: `1152`.
  - Average N-best count: `8.9462`.
  - Exact reference in N-best under raw whitespace normalization: `716`.
  - Rows with consensus spans: `991`.
  - Rows with confusable candidates: `1144`.
  - Hotword mentions in system/user prompts: `0`.
- Validation:
  - `scripts/train_lora_sft.py --dry-run --input-format qwen-messages` succeeds with the generated file.

#### Retest: clean no-hotword COVO format with norm-domain adapter

- Model:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`
- Input:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_clean.qwen.jsonl`
- Predictions:
  - `src/logs/shuili_covo_clean_nohotword_normdomain_sft60k_predictions_20260707.jsonl`
- Raw result:
  - CER: `0.111739`.
  - Baseline CER under this converted input: `0.149959`.
  - Improved samples: `544`.
  - Worsened samples: `192`.
  - Unchanged samples: `416`.
- Filler-normalized result, removing `呃/呢/啊/嗯`:
  - Base CER: `0.094895`.
  - Prediction CER: `0.082453`.
- Error audit:
  - N-best oracle CER: `0.054473`.
  - Prediction-or-oracle CER: `0.053520`.
  - `fixed_to_exact=261`.
  - `unchanged_error=249`.
  - `base_correct_broken=92`.
  - `worsened_error=100`.
  - `nbest_exact=719`.
  - `nbest_better_than_pred=517`.
- Decomposition:
  - Edits: `1760`.
  - Filler edits: `368` (`20.91%`).
  - Non-hotword edits: `1392` (`79.09%`).
- Interpretation:
  - Compared with the previous norm-domain evaluation (`CER=0.115294`, `fixed_to_exact=178`, `unchanged_error=430`), the clean no-hotword COVO format improves raw CER and makes the model use N-best more actively.
  - The tradeoff is over-editing: `base_correct_broken` and `worsened_error` both increase. This confirms the next step should not be blind candidate selection, but training a local span/uncertain-region editor that can use N-best evidence while protecting already-correct top1 spans.

#### Retest: clean no-hotword format with older COVO adapters

- Same input:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_clean.qwen.jsonl`
- Adapter A, AISHELL 4.35% CER no-op best:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`
  - Predictions: `src/logs/shuili_covo_clean_nohotword_preserve2_435_predictions_20260707.jsonl`
  - Raw CER: `0.134468`.
  - Filler-normalized CER: `0.089448`.
  - `fixed_to_exact=60`.
  - `unchanged_error=592`.
  - `base_correct_broken=44`.
  - `worsened_error=85`.
  - `nbest_better_than_pred=704`.
- Adapter B, ChineseHP original hard-negative text rewrite:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch`
  - Predictions: `src/logs/shuili_covo_clean_nohotword_chinesehp_hardneg_predictions_20260707.jsonl`
  - Raw CER: `0.137007`.
  - Filler-normalized CER: `0.094828`.
  - `fixed_to_exact=87`.
  - `unchanged_error=489`.
  - `base_correct_broken=61`.
  - `worsened_error=112`.
  - `nbest_better_than_pred=693`.
- Comparison:
  - Both older adapters improve over base (`0.149959`) but are much weaker than the Shuili norm-domain adapter on the same clean format (`0.111739`).
  - The AISHELL 4.35 no-op model is the most conservative: fewer broken-correct cases, but huge `unchanged_error`.
  - ChineseHP hard-negative is more willing to edit, but still leaves many oracle-reachable errors.
  - This supports the view that COVO has useful general correction ability, but Shuili needs domain/oral-style adaptation to unlock it. Clean format alone is not enough.

#### Raw Shuili candidate-pool retest, preserving more oral candidates

- Motivation:
  - The `clean_shuili_repeat_len18` candidate pool may remove oral-style candidates such as short filler-heavy outputs or repeated spoken fragments.
  - Test the earlier unclean pool directly, without the downstream COVO-format cleaner.
- Raw source pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_20260704.jsonl`
- Converted COVO input:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_rawpool.qwen.jsonl`
  - Summary: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_rawpool.summary.json`
- Candidate statistics:
  - Raw pool avg N-best: `9.0634`, exact reference in N-best: `717`.
  - Clean pool avg N-best: `8.9462`, exact reference in N-best: `716`.
  - Raw pool has slightly better oracle reachability, but also includes more noisy/repeated candidates.
- Current best model:
  - `qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`
- Rawpool prediction:
  - `src/logs/shuili_covo_rawpool_nohotword_normdomain_sft60k_predictions_20260707.jsonl`
- Result:
  - Raw CER: `0.113390`.
  - Filler-normalized CER: `0.084337`.
  - N-best oracle CER: `0.052632`.
  - `fixed_to_exact=260`.
  - `unchanged_error=247`.
  - `base_correct_broken=92`.
  - `worsened_error=100`.
  - `nbest_better_than_pred=522`.
- Comparison with clean no-hotword pool:
  - Clean pool result was raw CER `0.111739`, filler-normalized CER `0.082453`.
  - Raw pool is not better for the current model, despite slightly better oracle.
- Interpretation:
  - The problem is not simply that the previous pool deleted oral candidates. Fully raw candidates introduce noisy repeats and short fragments that COVO does not reliably ignore.
  - Next useful pool policy should be "oral-preserving but pollution-filtered": keep `呢/啊/呃/嗯`, short oral fragments, and natural repeated phrases when they are supported by N-best, but still remove obvious bad pollution, unusual Unicode, Latin tails, and extreme repetition.

#### FunASR/SenseVoice preprocessing probe for Shuili

- Motivation:
  - Current CB-Whisper candidate pools contain strange prompt-biased hypotheses such as `水利工程 闸门闸`.
  - Test a strong external Chinese ASR as preprocessing/candidate source.
- Added scripts:
  - `src/analysis/run_funasr_shuili_preprocess.py`
  - `src/analysis/merge_funasr_candidate_pool.py`
- Models tested on first 100 Shuili samples:
  - `paraformer-zh`: CER `0.030719`, exact `73/100`.
  - `iic/SenseVoiceSmall`: CER `0.027486`, exact `73/100`.
  - CB-Whisper top1 on same first 100: CER `0.118836`, exact `24/100`.
- Full Shuili preprocessing with SenseVoiceSmall:
  - Output: `src/logs/shuili_funasr_sensevoice_small_full_20260707.jsonl`
  - Summary: `src/logs/shuili_funasr_sensevoice_small_full_20260707_summary.json`
  - CER: `0.046537`.
  - Exact: `720/1152`.
  - This is already far below the 10% Shuili CER target.
- SenseVoice-first candidate pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_with_sensevoice_first_20260707.jsonl`
  - Average N-best: `9.3707`.
  - SenseVoice transcript was added to every row and used as `asr_top1`.
- COVO on SenseVoice-first input with current norm-domain adapter:
  - Predictions: `src/logs/shuili_covo_sensevoice_first_normdomain_sft60k_predictions_20260707.jsonl`
  - Raw CER: `0.084566`.
  - Baseline/SenseVoice input CER: `0.046537`.
  - Filler-normalized prediction CER: `0.065102`; baseline filler-normalized CER: `0.048356`.
  - `base_correct_broken=206`, `worsened_error=150`, `fixed_to_exact=69`.
  - N-best oracle CER: `0.026792`.
- Interpretation:
  - SenseVoiceSmall is a very strong Shuili preprocessor and fixes the candidate-quality problem directly.
  - Passing SenseVoice-first outputs through the current COVO adapter worsens them; COVO is trained for noisier CB-Whisper evidence and over-edits strong ASR hypotheses.
  - If using SenseVoice in the final workflow, either use it directly as preprocessing output or train a new COVO variant with strong no-op preservation for high-confidence SenseVoice hypotheses.

#### Original COVO input-format check on Shuili

- Question:
  - Whether our Shuili COVO inputs match the original COVO model's training format.
- Finding:
  - Earlier Shuili inputs were Qwen-message compatible, but not exactly the original COVO hard-negative prompt distribution.
  - The original COVO hardneg format uses `ASR top-1 + N-best + Pinyin + Confusable candidates`.
  - Our Shuili clean/no-hotword format added Shuili-specific oral constraints and explicit stable/uncertain-span wording, so it is a related but shifted input distribution.
- Regenerated original-style inputs with COVO's own script:
  - Clean CB pool:
    - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_originalprompt_clean.qwen.jsonl`
  - SenseVoice-first pool:
    - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_originalprompt_sensevoice_first.qwen.jsonl`
  - Command style:
    - `prepare_text_rewrite_data.py --evidence-mode nbest --include-pinyin --max-nbest 10 --max-pinyin 5 --max-hard-negatives 6`
- Original COVO model:
  - `qwen35_text_rewrite_hardneg_dropout_lora_2epoch`
- Clean CB pool result:
  - Prediction file: `src/logs/shuili_originalcovo_hardneg_originalprompt_cleanpool_predictions_20260707.jsonl`
  - Raw CER: `0.138404`.
  - Baseline CER: `0.154149`.
  - Filler-normalized CER: `0.095784`.
  - Filler-normalized baseline CER: `0.098139`.
  - N-best oracle CER: `0.053330`.
  - `fixed_to_exact=91`, `unchanged_error=461`, `base_correct_broken=64`, `worsened_error=131`.
- SenseVoice-first result:
  - Prediction file: `src/logs/shuili_originalcovo_hardneg_originalprompt_sensevoice_first_predictions_20260707.jsonl`
  - Raw CER: `0.100946`.
  - Baseline/SenseVoice CER: `0.046537`.
  - Filler-normalized CER: `0.076078`.
  - Filler-normalized baseline CER: `0.055508`.
  - N-best oracle CER: `0.026792`.
  - `fixed_to_exact=52`, `unchanged_error=161`, `base_correct_broken=323`, `worsened_error=169`.
- Interpretation:
  - With exact original-style input, original COVO can slightly improve the weak CB clean pool, but remains far behind Shuili-adapted COVO.
  - On strong SenseVoice top-1, original COVO over-edits heavily and degrades CER from `4.65%` to `10.09%`.
  - Therefore the current issue is not a file-format incompatibility. The original COVO correction distribution does not match Shuili oral/domain data or strong-ASR no-op preservation.

#### Shuili CB-Whisper candidate-quality diagnosis: neutral Whisper view

- Motivation:
  - We need to stay inside the CB-Whisper workflow, but current CB-Whisper candidate pools contain prompt-biased artifacts such as hotword repetition or short polluted hypotheses.
  - Test whether the problem is Whisper's base acoustic ability or the strong KWS/context-bias prompt.
- Diagnostic run:
  - Script: `src/analysis/aishell_full_whisper_decode.py`
  - Dataset: first `100` Shuili test utterances.
  - Model: `openai/whisper-large-v3`.
  - Setting: neutral/no-hotword beam search, `num_beams=5`, `num_return_sequences=5`, `batch_size=1`.
  - Output: `src/logs/shuili_whisper_large_v3_neutral_beam5_smoke100_20260708.jsonl`
- Result:
  - Neutral Whisper top1 CER: `0.097817`.
  - Unique n-best: `1.0` average. HF beam search mostly returns one normalized unique candidate for these short utterances.
  - Current CB-Whisper pool on the same first 100 rows:
    - top1 CER: `0.118836`.
    - oracle CER: `0.025061`.
    - exact reference in n-best: `80/100`.
    - average unique n-best: `8.68`.
- Interpretation:
  - Large-v3 neutral decoding is cleaner than current CB-Whisper top1 on Shuili, so the KWS/context prompt is hurting some top1 hypotheses.
  - However, CB-Whisper candidate pools have much better oracle reachability than neutral beam search. The right answer is often in lower-ranked CB candidates.
  - The next CB-Whisper-side route should be a stratified candidate generation policy:
    - neutral/acoustic Whisper view supplies a clean anchor;
    - weak context-bias view supplies small lexical variants;
    - strong hotword-biased CB view supplies recall candidates;
    - downstream rerank/COVO sees source tags and consensus, rather than treating all candidates as equal.
  - Plain beam n-best is not enough for diversity; useful diversity needs sampling or multi-view prompting, but strong hotword prompts should not dominate the first candidate.

#### Shuili neutral-first CB-Whisper candidate-pool experiment

- Goal:
  - Stay inside the CB-Whisper workflow while reducing prompt-biased top1 pollution.
  - Use neutral Whisper-large-v3 as a clean acoustic view, then keep CB-Whisper candidates behind it for hotword/oral variants.
- Added script:
  - `src/analysis/merge_asr_candidate_pool.py`
  - It merges an auxiliary ASR view into a CB-Whisper candidate pool, can place it first, refreshes source tags and pinyin, and reports top1/oracle CER.
- Full neutral decode:
  - Command used `openai/whisper-large-v3`, no hotword prompt, beam5, `num_return_sequences=5`, `batch_size=1`.
  - Output: `src/logs/shuili_whisper_large_v3_neutral_beam5_full_20260708.jsonl`
  - Full Shuili neutral top1 CER: `0.121326`.
  - Beam n-best diversity remained low; the output is essentially one unique neutral candidate per utterance.
- Neutral-first over clean CB pool:
  - Source pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_clean_shuili_repeat_len18_20260705.jsonl`
  - Merged pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_clean_len18_20260708.jsonl`
  - Pool top1 CER: `0.121389`.
  - Pool oracle CER: `0.042981`.
  - Oracle exact: `790/1152`.
  - COVO input: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_neutralfirst_beam5.qwen.jsonl`
  - COVO prediction: `src/logs/shuili_covo_neutralfirst_beam5_normdomain_sft60k_predictions_20260708.jsonl`
  - Raw CER: `0.104184`.
  - Filler-normalized CER: `0.088639`.
  - Audit: `fixed_to_exact=214`, `unchanged_error=254`, `base_correct_broken=111`, `worsened_error=110`.
- Neutral-first over raw CB pool:
  - Source pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_nbest10_multiprompt_t046_keep5_20260704.jsonl`
  - Merged pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_20260708.jsonl`
  - Pool top1 CER: `0.121389`.
  - Pool oracle CER: `0.042981`.
  - Oracle exact: `790/1152`.
  - COVO input: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_hardneg_consensus_nohotword_neutralfirst_beam5_rawpool.qwen.jsonl`
  - COVO prediction: `src/logs/shuili_covo_neutralfirst_beam5_rawpool_normdomain_sft60k_predictions_20260708.jsonl`
  - Raw CER: `0.103866`.
  - Filler-normalized CER: `0.089896`.
  - Audit: `fixed_to_exact=211`, `unchanged_error=258`, `base_correct_broken=112`, `worsened_error=109`.
- Comparison:
  - Previous clean no-hotword norm-domain COVO result: raw CER `0.111739`, filler-normalized CER `0.082453`.
  - Neutral-first improves raw CER by about `0.78` absolute points, but does not reach `<10%`.
  - It worsens filler-normalized CER because neutral Whisper tends to omit lecture fillers that appear in the reference.
- Interpretation:
  - The route is real: replacing the strong hotword-biased top1 with a neutral acoustic view reduces raw CER.
  - The remaining gap is not only candidate generation. The oracle is around `4.30%`, but COVO still outputs around `10.39%`; it leaves many oracle-reachable errors unresolved.
  - Next useful step is not more beam search. It should train or prompt a source-aware selector/editor that knows `neutral_whisper` is the clean anchor while CB-Whisper lower candidates may preserve oral fillers and hotword variants.

#### Shuili source-aware COVO prompt experiment

- Goal:
  - Test whether COVO can use candidate source information without retraining.
  - The prompt tells the model that `neutral_whisper` is a clean acoustic anchor, while `cbwhisper` candidates may add oral fillers/domain terms but may also contain context-bias hallucinations.
- Code change:
  - `src/analysis/convert_shuili_to_covo_format.py` now has optional flags:
    - `--include-source-tags`
    - `--source-aware-guidance`
    - `--source-aware-strict`
  - Default behavior is unchanged.
- Input pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_20260708.jsonl`
- Source-aware prompt, normal guidance:
  - COVO input: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_sourceaware_neutralfirst_beam5_rawpool.qwen.jsonl`
  - Prediction: `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_normdomain_sft60k_predictions_20260708.jsonl`
  - Raw CER: `0.102597`.
  - Filler-normalized CER: `0.087697`.
  - Audit: `fixed_to_exact=215`, `unchanged_error=262`, `base_correct_broken=108`, `worsened_error=102`.
- Source-aware prompt, strict guidance:
  - COVO input: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_sourceaware_strict_neutralfirst_beam5_rawpool.qwen.jsonl`
  - Prediction: `src/logs/shuili_covo_sourceaware_strict_neutralfirst_beam5_rawpool_normdomain_sft60k_predictions_20260708.jsonl`
  - Raw CER: `0.104120`.
  - Filler-normalized CER: `0.087540`.
  - Audit: `fixed_to_exact=215`, `unchanged_error=257`, `base_correct_broken=121`, `worsened_error=106`.
- Comparison:
  - Neutral-first rawpool without source-aware prompt: raw CER `0.103866`, filler-normalized CER `0.089896`.
  - Source-aware normal guidance is currently best on raw CER: `0.102597`.
  - Strict guidance improves filler-normalized CER slightly but hurts raw CER, mainly by breaking more originally correct cases.
- Interpretation:
  - Candidate source information is useful, but prompt-only use is not enough to pass the `<10%` raw CER target.
  - The remaining issue is learnable selection behavior: the model sees oracle-reachable candidates but still chooses/rewrites imperfectly.
  - A proper next experiment should train a source-aware selector/editor on this exact prompt family, not only append source tags at inference.

#### Shuili complete-candidate filtering

- Motivation:
  - Inspection found incomplete fragment candidates such as `混凝土工种` inside N-best.
  - These fragments are useful as local evidence, but harmful when presented as full-sentence candidates to COVO.
- Code change:
  - `src/analysis/merge_asr_candidate_pool.py` now supports:
    - `--filter-incomplete`
    - `--min-length-ratio`
    - `--length-slack`
  - It keeps the neutral anchor, then filters auxiliary CB/multiprompt candidates that are much shorter/longer than the anchor or contain known polluted video-tail phrases.
- First filter:
  - `--min-length-ratio 0.68 --length-slack 5`
  - Dropped `150` candidates.
  - It removed exact short fragments such as `混凝土工种`, but still allowed some too-short variants like `还有混凝土工种`.
- Stricter filter:
  - `--min-length-ratio 0.75 --length-slack 3`
  - Output pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_complete_v2_20260708.jsonl`
  - Dropped `396` incomplete/polluted candidates.
  - Average N-best: `8.7283`.
  - Top1 CER: `0.121389`.
  - Oracle CER: `0.042981` unchanged.
  - Oracle exact: `790/1152` unchanged.
  - Example `BAC009S0001W0467` after filtering:
    - Removed `混凝土工种` and `还有混凝土工种`.
    - Kept longer evidence-like candidates such as `还有红柠桃公种混凝土工种`.
- Source-aware COVO on complete-v2:
  - Input: `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_sourceaware_neutralfirst_beam5_rawpool_complete_v2.qwen.jsonl`
  - Prediction: `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_complete_v2_normdomain_sft60k_predictions_20260708.jsonl`
  - Raw CER: `0.100375`.
  - Filler-normalized CER: `0.086598`.
  - Audit: `fixed_to_exact=215`, `unchanged_error=262`, `base_correct_broken=105`, `worsened_error=104`.
- Comparison:
  - Source-aware rawpool before completeness filtering: raw CER `0.102597`.
  - Complete-v2 improves raw CER by about `0.22` absolute points and is now just above the `<10%` target.
  - Completeness filtering improves COVO behavior without hurting oracle reachability, so this is a cleaner candidate-pool policy than feeding fragment candidates.

#### Shuili candidate-pool reinspection after complete-v2

- User concern:
  - Some N-best entries were still not full utterance candidates, e.g. `水利工程 闸门闸` for reference `那么咱们这一个水利工程`.
  - The desired candidate pool should be sentence-level and roughly length-consistent, not local fragments.
- Complete-v2 audit:
  - Pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_complete_v2_20260708.jsonl`.
  - Average N-best: `8.7283`.
  - Samples with fewer than 5 candidates: `43/1152`.
  - Samples with candidates still short relative to reference: `113/1152`.
  - Samples with far/off-topic candidates relative to neutral anchor: `196/1152`, including `95` with multiprompt-derived far candidates.
  - Samples whose oracle candidate CER is still above `10%`: `199/1152`.
- More strict length-only check:
  - Command settings: `--min-length-ratio 0.88 --length-slack 1`.
  - Output pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_complete_strict_20260708.jsonl`.
  - Dropped `1404` incomplete/polluted candidates.
  - Average N-best decreased to `7.8481`.
  - Oracle CER stayed `0.042981`; oracle exact stayed `790/1152`.
  - Short-vs-reference samples decreased from `113` to `31`.
  - Example `BAC009S0001W0006` became:
    - `那么咱们这个水利工程`
    - `那么咱们这一个水利工程`
    - `咱们这一个水利工程`
    - The bad candidates `水利工程 闸门闸` and `水利工程 闸门门` were removed.
- Remaining issue:
  - Strict length filtering does not remove full-length but semantically off-topic candidates.
  - Example `BAC009S0001W0660` still keeps multiprompt candidates such as `这种事情在我们的生活上比较常见`, although they are not plausible sentence-level alternatives for the current audio.
  - Next candidate-pool cleanup should add an anchor-similarity/coverage filter, especially for multiprompt candidates, while preserving the current oracle CER.

#### Shuili anchor-consistency candidate cleanup

- Code change:
  - `src/analysis/merge_asr_candidate_pool.py` now supports:
    - `--filter-anchor-inconsistent`
    - `--max-anchor-edit-ratio`
    - `--anchor-filter-source-pattern`
  - The filter compares each selected-source candidate with the neutral acoustic anchor using normalized edit distance.
  - Current use only targets `multiprompt` candidates, because CB-Whisper candidates often contain useful domain/hotword variants even when they are phonetically noisy.
- Generated pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_clean_anchor_20260708.jsonl`
  - Settings:
    - `--min-length-ratio 0.88 --length-slack 1`
    - `--filter-anchor-inconsistent --max-anchor-edit-ratio 0.48 --anchor-filter-source-pattern multiprompt`
- Summary:
  - Dropped candidates: `1469`.
  - Average N-best: `7.7917`.
  - Top1 CER: `0.121389`.
  - Oracle CER: `0.042981`, unchanged from complete-v2/strict.
  - Oracle exact: `790/1152`, unchanged.
- Audit comparison:
  - Complete-v2:
    - Average N-best `8.7283`.
    - Short-vs-reference samples `113`.
    - Far/off-topic samples `196`, with `95` far multiprompt cases.
  - Strict length-only:
    - Average N-best `7.8481`.
    - Short-vs-reference samples `31`.
    - Far/off-topic samples `177`, with `82` far multiprompt cases.
  - Anchor-cleaned:
    - Average N-best `7.7917`.
    - Short-vs-reference samples `31`.
    - Far/off-topic samples `139`, with `35` far multiprompt cases.
- Example improvements:
  - `BAC009S0001W0006` now removes `水利工程 闸门闸` and `水利工程 闸门门`.
  - `BAC009S0001W0660` removes the clearly off-topic multiprompt candidate `这种事情在我们的生活上比较常见`.
- Remaining issue:
  - Some candidates are still wrong but acoustically/domain-plausible, e.g. `还有红柠桃公种混凝土工种`.
  - This cleanup improves candidate cleanliness without reducing oracle reachability; the next check should run COVO on this cleaned pool to see whether lower candidate noise converts into raw CER below `10%`.

#### Shuili COVO on anchor-cleaned candidate pool

- COVO input:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_sourceaware_neutralfirst_beam5_rawpool_clean_anchor.qwen.jsonl`
  - Rows: `1152`.
  - Average N-best: `7.7917`.
  - Exact reference in N-best: `746`.
  - Source-aware prompt: enabled.
- Model:
  - Base: `cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B`.
  - Adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`.
- Prediction:
  - `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_clean_anchor_normdomain_sft60k_predictions_20260708.jsonl`
- Metrics:
  - Raw CER: `0.100184`.
  - Baseline top1 CER under COVO evaluator: `0.123611`.
  - Filler-normalized CER: `0.086755`.
  - Filler-normalized baseline CER: `0.094920`.
- Audit:
  - `fixed_to_exact=222`.
  - `improved_partial=217`.
  - `unchanged_error=261`.
  - `base_correct_broken=105`.
  - `worsened_error=102`.
  - N-best oracle CER under COVO evaluator: `0.047489`.
  - `nbest_better_than_pred=497`, `nbest_better_than_base=643`.
- Comparison:
  - Complete-v2 source-aware COVO raw CER: `0.100375`.
  - Anchor-cleaned COVO raw CER: `0.100184`.
  - Gain is small (`0.019` absolute CER points), but it is in the right direction and does not hurt oracle reachability.
- Interpretation:
  - Candidate cleanup has now removed many visibly bad alternatives, but the model still fails to exploit reachable oracle candidates.
  - The remaining <10% gap is no longer mainly caused by obvious fragment/off-topic candidate pollution; it is selection/editing behavior (`nbest_better_than_pred=497`).

#### Shuili candidate analysis after anchor-cleaned COVO

- Candidate-pool distribution:
  - Pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_clean_anchor_20260708.jsonl`.
  - Rows: `1152`.
  - Average N-best: `7.7917`.
  - Exact reference in N-best: `749`.
  - Exact reference rank distribution:
    - rank1 `350`, rank2 `170`, rank3 `100`, rank4 `45`, rank5 `28`, rank6 `19`, rank7 `14`, rank8 `9`, rank9 `8`, rank10 `6`.
  - Best-candidate source distribution:
    - `neutral_whisper_large_v3_beam5`: `509`.
    - `cbwhisper_nbest`: `455`.
    - `cbwhisper_top1`: `104`.
    - `multiprompt_whisper`: `76`.
    - `cbwhisper_scored`: `8`.
- COVO behavior:
  - Prediction matched one of the N-best candidates in `1112/1152` rows; only `40` rows are free rewrites.
  - This means the current COVO mostly behaves as a candidate selector, not as a strong free-form editor.
  - It still fails to choose reachable exact candidates:
    - `base_correct_broken=105`, all with exact oracle at rank1.
    - `unchanged_error` includes many rows with exact candidates at rank2/3+.
    - Examples:
      - REF `自然界的水资源进行呢调配控制`; base/pred misses `呢`; exact candidate exists at rank6.
      - REF `那么我们呢`; base/pred is `那么我们的`; exact candidate exists at rank4.
      - REF `咱们主要的培养阶段呢`; base/pred ends with `是`; exact candidate exists at rank3.
- Simplified/traditional issue:
  - Anchor-cleaned pool still contains `608` traditional-form candidates across `231/1152` rows.
  - Traditional candidates by source:
    - `cbwhisper_nbest`: `453`.
    - `multiprompt_whisper`: `71`.
    - `cbwhisper_scored`: `50`.
    - `cbwhisper_top1`: `23`.
    - `neutral_whisper_large_v3_beam5`: `11`.
  - COVO outputs traditional text in `40` rows, and all `40` hurt raw CER.
  - If only the COVO prediction is converted from traditional to simplified, CER drops from `0.100184` to `0.091613`.
  - If the N-best candidates are converted to simplified and re-deduplicated before inference:
    - Average N-best changes only from `7.7917` to `7.7786`.
    - N-best oracle CER improves from `0.047489` to `0.042981`.
    - Exact reference in N-best improves from `749` to `790`.
  - Therefore simplification should be treated as candidate normalization, not merely as an output post-processing patch.
- Interpretation:
  - The current pool has two remaining candidate-side issues:
    - Simplified/traditional inconsistency injects non-ASR orthographic noise and hurts both oracle accounting and COVO output.
    - Correct candidates are often available but COVO lacks a reliable rank/source preference to pick them.
  - The next clean experiment should normalize N-best candidates to simplified Chinese before COVO conversion, then rerun source-aware COVO. This is a data-format normalization step and is less ad-hoc than changing prediction text after decoding.

#### Shuili simplified-candidate normalization

- Code change:
  - `src/analysis/merge_asr_candidate_pool.py` now supports `--simplify-candidates`.
  - Candidate text is converted to simplified Chinese before final N-best de-duplication.
  - If a candidate changes, its original surface is kept in `nbest_sources[].original_text` for audit.
- Generated pool:
  - `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_clean_anchor_simplified_20260708.jsonl`
  - Same filtering settings as anchor-cleaned pool, plus `--simplify-candidates`.
- Candidate-pool summary:
  - Average N-best: `7.7917`.
  - Top1 CER: `0.121389`.
  - Oracle CER: `0.042981`.
  - Oracle exact: `790/1152`.
- COVO input:
  - `cbwhisper_covo_migration_20260609_tar_extracted/covo/data/processed/shuili/test_text_rewrite_sourceaware_neutralfirst_beam5_rawpool_clean_anchor_simplified.qwen.jsonl`
  - Rows: `1152`.
  - Average N-best: `7.7917`.
  - Source-aware prompt: enabled.
- Prediction:
  - `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_clean_anchor_simplified_normdomain_sft60k_predictions_20260708.jsonl`
- Metrics:
  - Raw CER: `0.090280`.
  - Baseline top1 CER: `0.121389`.
  - Filler-normalized CER: `0.086049`.
  - Filler-normalized baseline CER: `0.094920`.
  - Traditional-output rows: `0`.
- Audit:
  - `fixed_to_exact=240`.
  - `improved_partial=218`.
  - `unchanged_error=261`.
  - `base_correct_broken=103`.
  - `worsened_error=82`.
  - N-best oracle CER: `0.042981`.
  - `nbest_better_than_pred=488`.
- Comparison:
  - Anchor-cleaned without simplified normalization: raw CER `0.100184`, filler-normalized CER `0.086755`.
  - Simplified normalization improves raw CER by about `0.99` absolute points and reaches the `<10%` target.
  - Most of the raw CER gain comes from removing simplified/traditional orthographic noise before COVO selection, not from a heavier model or post-hoc output patch.
- Interpretation:
  - Candidate simplification is now part of the clean candidate construction pipeline.
  - It is better justified as data normalization than output post-processing because it also restores the N-best oracle from `0.047489` to `0.042981` and exact oracle from `749` to `790`.

#### AISHELL check for simplified-candidate normalization

- Goal:
  - Test whether the Shuili candidate-normalization finding transfers back to AISHELL.
  - Use the 808-row AISHELL v3 candidate subset, not full 7176 AISHELL.
- Existing AISHELL reference point:
  - Input: `src/logs/cbwhisper_covo_evidence_aishell_v3_nbest10_gen32_clean.jsonl`.
  - Model: `qwen35_cbwhisper_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_bf16`.
  - Prediction: `src/logs/cbwhisper_covo_predictions_aishell_v3_chinesehp_hotword_aware_20260702.jsonl`.
  - Raw CER: `0.049670`.
  - Base CER under COVO evaluator: `0.104463`.
  - Hotword recall: base `0.9151`, prediction `0.8439`, N-best oracle `0.9161`.
- Neutral-first attempt:
  - Built a reference-matched neutral large-v3 view for all 808 rows:
    - `src/logs/aishell_v3_neutral_view_for_cb808_by_ref_20260708.jsonl`.
  - Generated:
    - `src/logs/cbwhisper_candidate_pool_aishell_v3_neutralfirst_clean_anchor_simplified_20260708.jsonl`.
  - Candidate summary:
    - Average N-best: `9.7525`.
    - Top1 CER: `0.131626`.
    - Oracle CER: `0.041676`.
    - Oracle exact: `572/808`.
  - Interpretation:
    - Unlike Shuili, neutral large-v3 is not a good top1 anchor for this AISHELL 808 subset.
    - The Shuili `neutral-first` rule should not be transferred blindly to AISHELL.
- CB-top simplified normalization:
  - Keep CB-Whisper top1 order and only normalize candidates to simplified Chinese before de-duplication.
  - Candidate pool:
    - `src/logs/cbwhisper_candidate_pool_aishell_v3_cbtop_clean_simplified_20260708.jsonl`.
  - Candidate summary:
    - Average N-best: `9.9097`.
    - Top1 CER: `0.085991` under merge-script normalization.
    - Oracle CER: `0.043073` under merge-script normalization.
    - Oracle exact: `572/808`.
  - COVO messages:
    - `src/logs/cbwhisper_covo_messages_aishell_v3_cbtop_clean_simplified_hotword_aware_20260708.jsonl`.
  - Prediction:
    - `src/logs/cbwhisper_covo_predictions_aishell_v3_cbtop_clean_simplified_hotword_aware_20260708.jsonl`.
- COVO result:
  - Raw CER: `0.048118`.
  - Base CER: `0.104463`.
  - N-best oracle CER: `0.041676`.
  - Prediction-or-N-best oracle CER: `0.025301`.
  - Traditional-output rows: `1`.
  - Delta counts:
    - `fixed_to_exact=186`.
    - `improved_partial=149`.
    - `unchanged_error=80`.
    - `base_correct_broken=63`.
    - `worsened_error=24`.
  - Hotword recall:
    - base `0.9151`.
    - prediction `0.8365`.
    - N-best oracle `0.9342`.
- Comparison:
  - Previous hotword-aware AISHELL result: raw CER `0.049670`.
  - CB-top simplified normalization result: raw CER `0.048118`.
  - Absolute CER gain: about `0.155` points.
  - The gain is real but much smaller than Shuili, because AISHELL had less harmful simplified/traditional noise and the hotword-aware model already corrected beyond N-best in many cases.
- Takeaway:
  - Candidate simplification transfers positively to AISHELL if CB-Whisper top1 is kept.
  - Neutral-first is dataset-dependent: helpful on Shuili, harmful as a top1 replacement on AISHELL.

#### Pure CB-Whisper Shuili control

- Goal:
  - Prove that the current Shuili `<10%` result is not achievable by pure CB-Whisper top1 alone.
  - Pure CB-Whisper here means: use `input.asr_top1` from the original CB-Whisper evidence, without COVO, without neutral-first, without candidate selection.
- Source evidence:
  - `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`
  - Rows: `1152`.
- Pure CB-Whisper prediction file:
  - `src/logs/shuili_pure_cbwhisper_asr_top1_predictions_20260708.jsonl`
  - Built by copying `input.asr_top1` into `prediction`.
- Metrics:
  - Original CB-Whisper metrics CSV:
    - File: `src/logs/test_metrics_shuili_v3_current_rerun_20260629.csv`
    - Entity Recall: `0.87155`.
    - CER: `0.14022`.
    - Hotword Sentence CER: `0.13963`.
    - Hotword Only CER: `0.39557`.
    - WER: `0.76128`.
  - COVO evaluator raw CER:
    - Pure CB-Whisper raw CER: `0.154149`.
    - Exact rows: `257/1152`.
    - N-best oracle CER from the same pure evidence: `0.064885`.
    - N-best exact: `688/1152`.
    - `nbest_better_than_pred=722`.
  - Filler-normalized CER:
    - Pure CB-Whisper filler-normalized CER: `0.098139`.
- Simplified-output sanity check:
  - Prediction file: `src/logs/shuili_pure_cbwhisper_asr_top1_simplified_predictions_20260708.jsonl`.
  - This only converts pure CB-Whisper output to simplified Chinese, still no COVO and no candidate selection.
  - Raw CER after simplification: `0.139166`.
  - Filler-normalized CER remains `0.098139`.
- Comparison to current best:
  - Current best Shuili pipeline:
    - Candidate pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_clean_anchor_simplified_20260708.jsonl`.
    - COVO prediction: `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_clean_anchor_simplified_normdomain_sft60k_predictions_20260708.jsonl`.
    - Raw CER: `0.090280`.
    - Filler-normalized CER: `0.086049`.
  - Therefore:
    - Pure CB-Whisper original CSV CER `0.14022` -> current best raw CER `0.09028`.
    - Pure CB-Whisper simplified raw CER `0.13917` -> current best raw CER `0.09028`.
    - Pure CB-Whisper filler-normalized CER `0.09814` -> current best filler-normalized CER `0.08605`.
- Interpretation:
  - Pure CB-Whisper is far from the current best result.
  - The useful improvement comes from using CB-Whisper as a candidate generator, then applying candidate normalization/cleaning and COVO selection/editing.
  - The pure evidence has strong oracle headroom (`0.064885` oracle CER and `722` rows where N-best beats top1), which supports the thesis that candidate use is necessary.

#### Shuili large-v3 as main-output control

- Clarification:
  - The intended control is not only pure CB-Whisper top1.
  - We also need to prove the effect of using neutral Whisper large-v3 as the main output/top1 anchor.
- Standalone large-v3 main output:
  - Source decode: `src/logs/shuili_whisper_large_v3_neutral_beam5_full_20260708.jsonl`.
  - Prediction file: `src/logs/shuili_large_v3_neutral_main_predictions_20260708.jsonl`.
  - Built by copying `asr_top1` from the neutral large-v3 decode into `prediction`.
  - Raw CER: `0.123611`.
  - Filler-normalized CER: `0.094920`.
  - Exact rows under filler-normalized evaluator: `585/1149`.
- Large-v3 as top1 anchor inside CB-Whisper candidate pool:
  - Candidate pool: `src/logs/cbwhisper_candidate_pool_shuili_v3_neutralfirst_beam5_rawpool_clean_anchor_simplified_20260708.jsonl`.
  - Policy:
    - Put neutral large-v3 output first.
    - Keep CB-Whisper candidates behind it.
    - Remove incomplete/off-anchor candidates.
    - Normalize candidates to simplified Chinese.
  - Pool top1 CER: `0.121389`.
  - Pool oracle CER: `0.042981`.
  - Pool oracle exact: `790/1152`.
- Large-v3 main output + CB-Whisper candidates + COVO:
  - Prediction file: `src/logs/shuili_covo_sourceaware_neutralfirst_beam5_rawpool_clean_anchor_simplified_normdomain_sft60k_predictions_20260708.jsonl`.
  - Raw CER: `0.090280`.
  - Filler-normalized CER: `0.086049`.
  - N-best oracle CER: `0.042981`.
  - `fixed_to_exact=240`, `improved_partial=218`, `worsened_error=82`.
- Comparison:
  - Pure CB-Whisper top1 raw CER: `0.154149`.
  - Pure CB-Whisper simplified raw CER: `0.139166`.
  - Standalone large-v3 main output raw CER: `0.123611`.
  - Large-v3 main output plus CB-Whisper candidates and COVO raw CER: `0.090280`.
- Interpretation:
  - Using large-v3 as the main output/top1 anchor is clearly helpful on Shuili.
  - But large-v3 alone is still not enough to reach `<10%`.
  - The final gain comes from combining:
    - large-v3 as clean acoustic anchor,
    - CB-Whisper as hotword/context candidate generator,
    - candidate cleaning/simplification,
    - COVO candidate selection and local editing.

#### Shuili CB-Whisper internal neutral-anchor probe

- Motivation:
  - Pure CB-Whisper top1 is worse than standalone large-v3 on Shuili, even though the run is already based on `openai/whisper-large-v3`.
  - The likely cause is not the base ASR model but KWS/context prompting and hotword reranking pulling top1 away from the clean acoustic hypothesis.
- Code change:
  - `CBWhisper` now has optional neutral-anchor flags:
    - `neutral_anchor`
    - `neutral_anchor_as_top1`
    - `neutral_anchor_include_in_nbest`
    - `neutral_anchor_num_beams`
    - `neutral_anchor_skip_surface_repair`
  - Default behavior is unchanged. These flags are only active when explicitly enabled.
  - Test script: `src/run_cbwhisper_shuili_v3_kws_neutral_anchor_test.py`.
- Forced neutral-anchor top1 run:
  - Metrics file: `src/logs/test_metrics_shuili_v3_neutral_anchor_20260708.csv`.
  - Entity Recall: `0.61465`.
  - CER: `0.12904`.
  - Hotword Only CER: `0.47123`.
  - WER: `0.71181`.
  - Interpretation: forcing neutral large-v3 as top1 improves over pure CB-Whisper CER but destroys hotword recall, so it is not acceptable as CB-Whisper output.
- Neutral-anchor as rerank candidate:
  - Metrics file: `src/logs/test_metrics_shuili_v3_neutral_anchor_rerank_20260708.csv`.
  - Entity Recall: `0.86915`.
  - CER: `0.13913`.
  - Hotword Sentence CER: `0.13783`.
  - Hotword Only CER: `0.39614`.
  - WER: `0.76128`.
  - Oracle summary: `src/logs/oracle_nbest_summary_shuili_v3_neutral_anchor.csv`.
  - N-best oracle CER: `0.07710`; top1 diagnostic CER: `0.13950`.
  - Interpretation: adding neutral large-v3 as a rerank candidate preserves recall relative to current CB-Whisper, but does not solve the CER problem. It is not promoted.
- Decision:
  - Do not report either internal-anchor variant as the main Shuili result.
  - The accepted Shuili direction remains external candidate-pool construction: use neutral large-v3 as a clean top1/source-tagged anchor, keep CB-Whisper candidates behind it, clean/simplify candidates, then let source-aware COVO select/edit.

#### Shuili naked CB-Whisper completeness rerank

- Motivation:
  - Reinspect naked CB-Whisper before relying on COVO.
  - The original Shuili true-v3 CB-Whisper top1 is often too short or fragment-like even when a fuller candidate exists nearby in the same n-best list.
- Offline diagnosis on `src/logs/cbwhisper_covo_evidence_shuili_v3_current_rerun_20260629.jsonl`:
  - Current top1 COVO-evaluator raw CER: `0.154149`.
  - Current top1 after simplified normalization: `0.139166`.
  - Oracle over CB-Whisper candidates: raw CER `0.060441`; exact `711/1152`.
  - Choosing the longest normalized candidate among candidates within a total-score margin gives a large diagnostic gain:
    - margin `0.3`: simplified-normalized CER `0.107168`, exact `458/1152`, entity recall about `0.8487`.
    - margin `0.4`: simplified-normalized CER `0.104247`, exact `468/1152`, entity recall about `0.8475`.
  - Interpretation: the old rerank overweights hotword/ASR score relative to utterance completeness on Shuili oral lecture data.
- Code change:
  - `CBWhisper` now has optional completeness-rerank parameters:
    - `enable_completeness_rerank`
    - `completeness_rerank_total_margin`
  - Default behavior is unchanged.
  - `src/run_cbwhisper_shuili_v3_kws_test.py` exposes the option through:
    - `CBW_COMPLETENESS_RERANK`
    - `CBW_COMPLETENESS_MARGIN`
    - plus oracle-output env vars to avoid overwriting old diagnostics.
- Full naked CB-Whisper Shuili run:
  - Command setting: `CBW_COMPLETENESS_RERANK=1`, `CBW_COMPLETENESS_MARGIN=0.3`.
  - Metrics file: `src/logs/test_metrics_shuili_v3_completeness_rerank_m03_20260709.csv`.
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_v3_completeness_rerank_m03_20260709.jsonl`.
  - Entity Recall: `0.87395`.
  - CER: `0.10699`.
  - Hotword Sentence CER: `0.12041`.
  - Hotword Only CER: `0.31444`.
  - WER: `0.60243`.
  - COVO-evaluator raw CER on the exported top1: `0.121897`.
  - Oracle summary: `src/logs/oracle_nbest_summary_shuili_v3_completeness_rerank_m03_20260709.csv`.
  - N-best oracle diagnostic CER: `0.07882`; top1 diagnostic CER: `0.10727`.
- Comparison with previous naked true-v3 CB-Whisper Shuili run:
  - Previous metrics: Entity Recall `0.87155`, CER `0.14022`, Hotword Only CER `0.39557`, WER `0.76128`.
  - Completeness rerank improves:
    - CER by about `-0.03323`.
    - Hotword Only CER by about `-0.08113`.
    - WER by about `-0.15885`.
    - Entity Recall by about `+0.00240`.
- Interpretation:
  - This is the first clean naked-CB-Whisper improvement on Shuili after the v3 rerank issue was identified.
  - It remains worse than the current COVO-assisted best raw CER (`0.090280`), but it substantially narrows the gap without using neutral external ASR or COVO.
  - Because the margin was selected after Shuili diagnostics, this should be validated on AISHELL/dev before treating it as a general method. For now, it is a promising Shuili-specific completeness prior.

#### Shuili video dataset from `Videos.zip`

- Source:
  - Raw archive: `/root/autodl-tmp/newdata/Videos.zip`.
  - Extracted files:
    - `/root/autodl-tmp/newdata/7月11日/7月11日.mp4`
    - `/root/autodl-tmp/newdata/7月11日/7月11日.srt`
    - `/root/autodl-tmp/newdata/7月12日/7月12日.mp4`
    - `/root/autodl-tmp/newdata/7月12日/7月12日.srt`
- Dataset construction:
  - Builder script: `src/analysis/build_shuili_video_dataset.py`.
  - Target dataset: `/root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3`.
  - The script extracts 16 kHz mono audio from each mp4, parses SRT timestamps, cuts subtitle-level wav segments, and writes files in the existing Shuili/AISHELL-style layout.
  - Segment IDs are made compatible with `AishellHotwordDataset`, e.g. `BAC011S0001W0001` and `BAC012S0002W0001`.
  - Reused old Shuili hotword inventory and keyword hidden states from `/root/autodl-tmp/datasets/shuili/data_shuil_largev3`.
  - Generated split summary:
    - samples: `990`
    - transcript characters: `10691`
    - average transcript length: `10.80` chars
    - hotwords: `124`
    - subtitle-matched hotword mentions: `150`
  - Utterance hidden states were extracted with `openai/whisper-large-v3`, `hs_fuse=last`, `d_model=1280`.
- Code fixes/changes:
  - `src/run_cbwhisper_shuili_v3_kws_test.py` now accepts `SHUILI_ROOT` so the same runner can test either old Shuili or the new video dataset.
  - `src/utils.py` now casts Whisper input features to the encoder parameter dtype during hidden-state extraction. This fixes the large-v3 extraction failure: `Input type (float) and bias type (c10::Half) should be the same`.
- Full test:
  - Runner: `src/run_cbwhisper_shuili_v3_kws_test.py`.
  - Dataset root: `SHUILI_ROOT=/root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3`.
  - Setting: `CBW_COMPLETENESS_RERANK=1`, `CBW_COMPLETENESS_MARGIN=0.3`.
  - Metrics file: `src/logs/test_metrics_shuili_videos_v3_completeness_m03_20260713.csv`.
  - Oracle summary: `src/logs/oracle_nbest_summary_shuili_videos_v3_completeness_m03_20260713.csv`.
  - Entity Recall: `0.97183`.
  - CER: `0.14273`.
  - Hotword Sentence CER: `0.08313`.
  - Hotword Only CER: `0.08101`.
  - WER: `0.51717`.
  - N-best diagnostic:
    - average n-best size: `9.81`
    - top1 diagnostic CER: `0.14316`
    - oracle diagnostic CER: `0.05192`
    - oracle rank: `2.33`
- Interpretation:
  - Hotword recall is high on the subtitle-matched hotword subset, so the main error source is no longer hotword recall.
  - Overall CER is high because this new video/subtitle set is documentary-style short subtitle segmentation and differs from the old oral lecture Shuili distribution.
  - The oracle CER gap (`0.14316` top1 vs `0.05192` oracle) says useful candidates are often present, but current naked CB-Whisper ranking is still weak for this dataset.
  - Important note: this run accidentally used `CBW_COVO_EVIDENCE_OUT` instead of the current `CBW_EVIDENCE_OUT`, so no COVO evidence jsonl was exported. Metrics and oracle CSVs are valid.

#### Shuili video expanded-hotword and COVO connection

- Motivation:
  - The first video-dataset run used only the old Shuili 124-hotword list, with only `150` subtitle-matched hotword mentions.
  - We tested whether adding more domain terms from the new video subtitles helps CB-Whisper and then connected the exported evidence to COVO.
- Dataset rebuild:
  - Builder: `src/analysis/build_shuili_video_dataset.py`.
  - Setting: `--expand-hotwords --max-hotwords 180`.
  - Target dataset: `/root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3`.
  - Summary:
    - samples: `990`
    - hotwords: `180`
    - natural keyword audios written for new terms: `56`
    - aligned subtitle hotword mentions: `647`
  - Hidden states:
    - utterance hidden states: `990/990`
    - keyword hidden states: `180/180`
    - Whisper profile: `openai/whisper-large-v3`, `hs_fuse=last`, `d_model=1280`
- Naked CB-Whisper test with expanded 180 hotwords:
  - Setting: `CBW_COMPLETENESS_RERANK=1`, `CBW_COMPLETENESS_MARGIN=0.3`.
  - Metrics: `src/logs/test_metrics_shuili_videos_v3_expand180_completeness_m03_20260713.csv`.
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_expand180_completeness_m03_20260713.jsonl`.
  - Entity Recall: `0.94144`.
  - CER: `0.14499`.
  - Hotword Sentence CER: `0.10699`.
  - Hotword Only CER: `0.10160`.
  - WER: `0.50202`.
  - Oracle summary: `src/logs/oracle_nbest_summary_shuili_videos_v3_expand180_completeness_m03_20260713.csv`.
  - N-best diagnostic:
    - average n-best size: `9.73`
    - top1 diagnostic CER: `0.14543`
    - oracle diagnostic CER: `0.05086`
    - oracle rank: `2.28`
- Comparison with 124-hotword video run:
  - Hotword count/matched mentions increased: `124/150` -> `180/647`.
  - Entity Recall worsened: `0.97183` -> `0.94144`.
  - CER worsened: `0.14273` -> `0.14499`.
  - Hotword Sentence CER worsened: `0.08313` -> `0.10699`.
  - Hotword Only CER worsened: `0.08101` -> `0.10160`.
  - WER improved: `0.51717` -> `0.50202`.
  - Interpretation: simply increasing the hotword list brings in many more true mentions, but also increases KWS false positives and makes naked CB-Whisper ranking worse.
- COVO connection on expanded 180-hotword evidence:
  - Adapter: `/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16`.
  - Full hotword-evidence prompt:
    - Messages: `src/logs/cbwhisper_covo_messages_shuili_videos_expand180_normdomain_sft60k_20260713.jsonl`.
    - Predictions: `src/logs/shuili_videos_covo_expand180_normdomain_sft60k_predictions_20260713.jsonl`.
    - Raw COVO evaluator CER: `0.20591`, baseline `0.23035`.
    - Filler/OpenCC-normalized CER: `0.11668`, baseline `0.12189`.
    - Improved/worsened/unchanged after normalized evaluation: `199/132/659`.
  - Cleaner prompt using only prompt-source hotwords:
    - Messages: `src/logs/cbwhisper_covo_messages_shuili_videos_expand180_cleanprompt_normdomain_sft60k_20260713.jsonl`.
    - Predictions: `src/logs/shuili_videos_covo_expand180_cleanprompt_normdomain_sft60k_predictions_20260713.jsonl`.
    - Raw COVO evaluator CER: `0.20912`, baseline `0.23035`.
    - Filler/OpenCC-normalized CER: `0.11794`, baseline `0.12189`.
    - Improved/worsened/unchanged after normalized evaluation: `214/157/619`.
- Interpretation:
  - COVO is connected and improves over its own message baseline after normalization, but it is not yet better than the naked CB-Whisper metric result on this new video dataset.
  - The first COVO error pattern includes harmful rewrites of already-correct or traditional-form outputs, e.g. `魚潛鳥飛` can be changed to a wrong homophone-like form.
  - The clean-prompt COVO variant did not outperform the fuller hotword-evidence prompt, so the current bottleneck is not only too many hotwords. The larger issue is domain/style mismatch between the video subtitle set and the COVO model trained mainly on previous Shuili oral data.
  - Next promising direction: build video-style COVO SFT examples from this dataset's own CB-Whisper n-best/oracle evidence, with explicit no-op targets for already-correct samples and stronger simplification/variant preservation.

#### Shuili video COVO simplified-input fix

- Issue:
  - `cbwhisper_covo_bridge.py` used simplified/normalized text for de-duplication and consensus, but the actual prompt still displayed raw ASR/N-best surfaces.
  - On the video dataset this meant COVO saw many traditional candidates, e.g. `魚潛鳥飛`, and sometimes rewrote an already correct traditional-form top1 into a wrong homophone.
- Code change:
  - `cbwhisper_covo_bridge.py` now converts ASR top1, N-best candidates, CB-Whisper scored candidates, prompt/KWS hotwords, keyword mentions, and oracle hotwords to simplified Chinese before message construction.
  - The default is simplified input; `--no-simplify-input` keeps the old behavior for ablation.
  - N-best cleaning still runs after simplification, so simplified duplicates collapse before prompting.
- Full COVO run after the fix:
  - Messages: `src/logs/cbwhisper_covo_messages_shuili_videos_expand180_simplified_normdomain_sft60k_20260713.jsonl`.
  - Predictions: `src/logs/shuili_videos_covo_expand180_simplified_normdomain_sft60k_predictions_20260713.jsonl`.
  - Raw COVO evaluator CER: `0.11484`, baseline `0.12192`.
  - Filler/OpenCC-normalized CER: `0.11321`, baseline `0.12189`.
  - Improved/worsened/unchanged after normalized evaluation: `200/112/678`.
- Comparison:
  - Previous full hotword-evidence COVO normalized CER: `0.11668`.
  - Simplified-input COVO normalized CER: `0.11321`.
  - Absolute gain from simplifying prompt input: about `-0.00347` CER.
  - Worsened samples decreased from `132` to `112`.
- Remaining main error sources after the fix:
  - Many errors are non-hotword ASR errors; samples with real hotword mentions have CER about `0.08905`, while no-real-hotword samples have CER about `0.14612`.
  - Candidate selection is still weak: normalized top1 CER is `0.12189`, while N-best oracle CER is about `0.05111`.
  - Common residual errors include number-format mismatch (`10到20` vs `十到二十`, `6400` vs `六千四百`), sentence-boundary/function-word mismatch (`是/与/在/要`), and homophone domain mistakes (`工业/供应`, `园区/原区`, `节水/技术`).

#### Shuili video number-normalized evaluation

- Issue:
  - Several previous "errors" were only numeric surface differences, not true recognition errors.
  - Examples: `10到20` vs `十到二十`, `6,400亿` vs `六千四百亿`, `2025` vs `二零二五`, `98.2%` vs `百分之九十八点二`.
- Code change:
  - `src/analysis/evaluate_filler_normalized_cer.py` now supports `--normalize-numbers`.
  - The normalization is applied before punctuation removal, so decimal points, percentages, fractions, and large units can be aligned before CER calculation.
  - Covered forms include Chinese digits/units, percentages, fractions, decimals, year-like digit strings, and final `万/亿` expressions.
- Re-evaluation on simplified-input COVO output:
  - Input: `src/logs/shuili_videos_covo_expand180_simplified_normdomain_sft60k_predictions_20260713.jsonl`.
  - Output: `src/logs/filler_number_normalized_cer_shuili_videos_expand180_simplified_normdomain_sft60k_20260713.json`.
  - Baseline CER after filler + number normalization: `0.10270`.
  - COVO CER after filler + number normalization: `0.08643`.
  - Improved/worsened/unchanged: `199/89/702`.
  - Exact samples: baseline `532`, COVO `583`.
- Comparison:
  - Without number normalization, simplified-input COVO CER was `0.11321`.
  - With number normalization, it becomes `0.08643`.
  - Therefore the video result is already below 9% CER under a more appropriate numeric-equivalence metric.
- Remaining true model errors:
  - COVO still sometimes rewrites a correct top1 into a worse n-best variant:
    - `鱼潜鸟飞` -> `鱼前鸟飞`.
    - `提示人们珍惜水资源` -> `要提示人们珍惜水资源`.
    - `中国水资源人均拥有量` -> `包括中国水资源人均拥有量`.
  - It also misses cases where a better candidate exists:
    - `钢铁供应...` vs candidate `钢铁工业...`.
    - `园区/原区`, `节水/技术`, `涧/渐`.
  - Interpretation: the remaining bottleneck is not numeric formatting or hotword recall. It is COVO's candidate-selection reliability: it needs stronger no-op training for already-correct top1 and stronger domain-knowledge/consensus evidence for choosing among homophones.

#### Shuili video compact COVO evidence

- Motivation:
  - The verbose COVO prompt was too long and noisy. For sample `id=50`, the full message repeated empty `exact/phon` scores, empty hotword-status rows, KWS false positives, duplicated candidate-score blocks, and repeated pinyin.
  - This made simple cases like `这是最大的刚性约束` harder than necessary because the unsupported KWS hotword `钢筋` was visually prominent.
- Code change:
  - `src/analysis/cbwhisper_covo_bridge.py` now defaults to compact evidence.
  - `--no-compact-evidence` restores the older verbose prompt for ablation.
  - Compact evidence:
    - keeps ASR top1 and N-best candidates;
    - keeps rank/total/asr scores but omits zero exact/phon fields;
    - keeps stable/uncertain spans but simplifies variants to plain lists when no hotword delta exists;
    - labels unsupported prompt hotwords as `not found in candidates; do not force`;
    - removes all-empty candidate hotword status rows;
    - omits the duplicated `CB-Whisper candidate scores` block;
    - adds a compact-evidence note: if top1 is high-score, complete, and has no supported-hotword conflict, prefer keeping top1.
- Prompt length:
  - Previous simplified verbose messages:
    - average user prompt chars: `5487.8`
    - median: `5635.5`
    - p95: `6445`
  - Compact v3 messages:
    - average user prompt chars: `1698.6`
    - median: `1616.5`
    - p95: `2341`
  - Example `id=50` shrank from about `5811` user-prompt chars to about `1491`.
- Full COVO run:
  - Messages: `src/logs/cbwhisper_covo_messages_shuili_videos_expand180_compact_v3_normdomain_sft60k_20260713.jsonl`.
  - Predictions: `src/logs/shuili_videos_covo_expand180_compact_v3_normdomain_sft60k_predictions_20260713.jsonl`.
  - Raw COVO evaluator CER: `0.11598`, baseline `0.12192`.
  - Filler + number normalized CER: `0.08595`, baseline `0.10270`.
  - Improved/worsened/unchanged under number-normalized evaluation: `207/103/680`.
- Comparison:
  - Previous best simplified verbose COVO number-normalized CER: `0.08643`.
  - Compact v3 number-normalized CER: `0.08595`.
  - Compact v3 is both shorter and slightly better; promote compact evidence as the default COVO input style for the current Shuili video workflow.

#### Shuili video CB-Whisper anti-insertion prior

- Motivation:
  - Error inspection showed that many residual mistakes came from CB-Whisper/COVO selecting candidates with unsupported extra words rather than true hotword corrections.
  - Examples include extra sentence-initial words such as `是/与/包括/要`, or tail insertions that do not carry hotword evidence.
  - The existing completeness rerank helped avoid fragments, but it could also prefer unnecessarily longer candidates inside the near-best score band.
- Code change:
  - `CBWhisper` now has an optional anti-insertion prior:
    - `enable_insertion_penalty`
    - `insertion_penalty_weight`
    - `insertion_penalty_free_chars`
    - `insertion_penalty_min_anchor_chars`
    - `insertion_penalty_allow_exact_gain`
  - The penalty compares each candidate to the best-ASR anchor and subtracts score for unsupported inserted characters.
  - If the longer candidate gains exact hotword coverage, the insertion penalty can be waived, so real hotword corrections are still allowed.
  - Completeness rerank now prefers near-best candidates with lower insertion penalty before using length as a tie-breaker.
  - `run_cbwhisper_shuili_v3_kws_test.py` exposes these knobs through `CBW_INSERTION_PENALTY*` environment variables.
- Full CB-Whisper run:
  - Setting:
    - `CBW_COMPLETENESS_RERANK=1`
    - `CBW_COMPLETENESS_MARGIN=0.3`
    - `CBW_INSERTION_PENALTY=1`
    - `CBW_INSERTION_PENALTY_WEIGHT=0.25`
    - `CBW_INSERTION_PENALTY_FREE_CHARS=1`
  - Metrics: `src/logs/test_metrics_shuili_videos_v3_expand180_completeness_m03_insertpen025_20260713.csv`.
  - Evidence: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_expand180_completeness_m03_insertpen025_20260713.jsonl`.
  - Entity Recall: `0.94144`.
  - CER: `0.14434` versus previous `0.14499`.
  - Hotword Sentence CER: `0.10481` versus previous `0.10699`.
  - Hotword Only CER: `0.09532` versus previous `0.10160`.
  - WER: `0.51212` versus previous `0.50202`; WER worsened slightly, so this should be judged mainly through downstream CER and error analysis.
- Downstream COVO run with compact evidence:
  - Messages: `src/logs/cbwhisper_covo_messages_shuili_videos_expand180_insertpen025_compact_normdomain_sft60k_20260713.jsonl`.
  - Predictions: `src/logs/shuili_videos_covo_expand180_insertpen025_compact_normdomain_sft60k_predictions_20260713.jsonl`.
  - Raw COVO evaluator CER: `0.11116`, baseline `0.12041`.
  - Filler + number normalized CER: `0.08271`, baseline `0.10461`.
  - Improved/worsened/unchanged under number-normalized evaluation: `221/96/673`.
- Comparison with previous compact-v3 route:
  - Previous compact-v3 number-normalized CER: `0.08595`.
  - Anti-insertion + compact route number-normalized CER: `0.08271`.
  - Prediction edits decreased: `903` -> `869`.
  - Worsened samples decreased: `103` -> `96`.
  - Improved samples increased: `207` -> `221`.
- Interpretation:
  - The user's diagnosis was correct: a meaningful part of the remaining error came from unsupported word insertion rather than hotword recall.
  - Anti-insertion is a lightweight reranking prior, not a hard gate; it remains compatible with the paper's "lightweight CB-Whisper + COVO evidence" direction.

#### Shuili video standalone large-v3 baseline

- Purpose:
  - Check the naked `openai/whisper-large-v3` baseline on the current Shuili video test set, without CB-Whisper hotword prompting/reranking and without COVO.
- Command:
  - `analysis/whisper_clean_decode.py --root /root/autodl-tmp/datasets/shuili/data_shuil_videos_largev3 --split test --kw-type natural --whisper-ckpt openai/whisper-large-v3 --batch-size 1 --num-beams 5`
- Output:
  - CSV: `src/logs/whisper_clean_decode_shuili_videos_large_v3_beam5_20260714.csv`.
  - Stdout: `src/logs/whisper_clean_decode_shuili_videos_large_v3_beam5_20260714_stdout.log`.
- Metrics:
  - Samples: `990`.
  - Standard script corpus CER: `0.09323`.
  - Standard script mean CER: `0.10671`.
  - Exact matches: `535/990`.
  - Filler + number normalized CER: `0.08367`.
  - Filler + number normalized exact matches: `580/990`.
- Comparison:
  - Current best anti-insertion CB-Whisper + compact COVO route has filler + number normalized CER `0.08271`.
  - The full route is only about `0.00095` absolute CER better than naked large-v3 under the current normalized metric, so the remaining paper problem is proving that the hotword/COVO workflow adds robust value beyond an already strong large-v3 baseline.

#### Shuili video CB-Whisper degradation diagnosis

- Purpose:
  - Check why naked large-v3 is much better than pure CB-Whisper on the current Shuili video set, and whether the KWS module is broken.
- Files:
  - KWS run-config top-k CSV: `src/logs/kws_topk_recall_shuili_videos_large_v3_runconfig_20260714.csv`.
  - KWS run-config stdout: `src/logs/kws_topk_recall_shuili_videos_large_v3_runconfig_20260714_stdout.log`.
  - CB-Whisper evidence: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_expand180_completeness_m03_insertpen025_20260713.jsonl`.
  - Naked v3 CSV: `src/logs/whisper_clean_decode_shuili_videos_large_v3_beam5_20260714.csv`.
- KWS diagnosis:
  - Hotword-bearing samples: `361`.
  - True hotwords: `647`.
  - KWS micro recall: `@1 0.30294`, `@3 0.49304`, `@6 0.58114`, `@12 0.67079`, `@28 0.80989`, `@50 0.92427`.
  - Under the actual current CB-Whisper selection config (`threshold=0.05`, `topk_per_group=50`, `max_prompt_keywords=200`, `rescore_max_keywords=160`):
    - CB/rescore pool micro recall: `0.72952`.
    - Prompt micro recall: `0.48068`.
    - Average rescore pool size on hotword-bearing samples: `17.86`.
    - Average prompt size on hotword-bearing samples: `2.03`.
- CB versus naked v3:
  - Same normalized surface metric used by the clean v3 script:
    - CB-Whisper top1 CER: `0.11512`.
    - Naked large-v3 CER: `0.09323`.
  - Per-sample comparison:
    - CB worse: `164`.
    - CB better: `84`.
    - Same edit count: `742`.
    - CB worse added `353` extra edits; CB better saved only `121` edits.
  - On samples without true hotword mentions:
    - CB CER: `0.13633`.
    - Naked v3 CER: `0.10634`.
    - CB worse/better: `110/34`.
  - On samples with true hotword mentions:
    - CB CER: `0.08557`.
    - Naked v3 CER: `0.07496`.
    - CB worse/better: `54/50`.
- Error shape:
  - Compared with naked v3, CB-Whisper has many more insertions: CB insertion ops `169` versus naked v3 `55`.
  - Among CB-worse samples, extra errors include insertions, truncations, and substitutions.
  - Typical failures:
    - `以水定产` -> `以水定产 缺水 靠水`.
    - `这个水资源啊` -> `这个水资源 闸门源源源`.
    - `变成再生水` -> `辨称再生水 地基 丹江口水库`.
    - `工程设计建设` -> `工程设计建设 技术人员 工程量 填筑者员`.
  - No-true-hotword samples still receive an average of about `1.54` prompt hotwords and `17.02` rescore hotwords, so false-positive hotword context is a major pollution source.
- Interpretation:
  - KWS is not completely broken: the true hotword often appears somewhere in top-50.
  - But the useful KWS signal is weak for this dataset: prompt recall is only about `0.48`, and the rescore pool needs many low-confidence words to reach usable recall.
  - The main degradation is CB-Whisper utilization/reranking: broad low-threshold KWS context creates false hotword pressure, and the exact/phonetic reward can select candidates with unsupported insertions or tail hotwords over the cleaner ASR candidate.
  - Pure CB-Whisper should not be judged as a strong Shuili baseline in its current setting. The next fix should make hotword bias conditional on strong evidence or preserve naked v3 as a neutral anchor unless the hotword-bearing candidate is both complete and ASR-plausible.

#### Shuili video neutral-anchor guard experiment

- Branch:
  - `cbwhisper-shuili-cer-tuning-20260714`.
- Code change:
  - Added optional `neutral_anchor_guard` to `CBWhisper`.
  - The guard includes the unprompted large-v3 transcript as a neutral anchor in n-best.
  - A hotword candidate may replace the anchor only if it has enough exact hotword gain, stays close in length/edit distance to the anchor, and passes a minimum ASR-score floor.
  - `run_cbwhisper_shuili_v3_kws_test.py` exposes the guard through `CBW_NEUTRAL_*` environment variables.
- Full run:
  - Stdout: `src/logs/experiment_shuili_videos_v3_neutral_guard_20260714_stdout.log`.
  - Metrics: `src/logs/test_metrics_shuili_videos_v3_neutral_guard_20260714.csv`.
  - Runtime probe: `src/logs/runtime_probe_shuili_videos_v3_neutral_guard_20260714.jsonl`.
  - Oracle n-best detail: `src/logs/oracle_nbest_detail_shuili_videos_v3_neutral_guard_20260714.csv`.
  - Oracle n-best summary: `src/logs/oracle_nbest_summary_shuili_videos_v3_neutral_guard_20260714.csv`.
- Metrics:
  - Entity Recall: `0.89414`.
  - CER: `0.10631`.
  - Hotword Sentence CER: `0.07705`.
  - Hotword Only CER: `0.08847`.
  - WER: `0.45556`.
- Comparison:
  - Previous pure CB-Whisper expand180 + insertion penalty CER: `0.14434`.
  - Neutral-anchor guard CER: `0.10631`.
  - Naked large-v3 clean decode CER: `0.09323`.
  - The guard fixes much of the false-hotword pollution but still does not beat naked large-v3.
- N-best upper bound:
  - Oracle n-best CER in this run: `0.04939`.
  - Average n-best size: `9.79`.
  - This means the candidate pool is strong enough to hit the 6% target, but the current hand-written selector is not.
- Current interpretation:
  - The next useful step is not more exact/phonetic weight tuning.
  - We need a stronger selector/reranker trained on candidate evidence, or a better non-cheating confidence model that can identify the oracle-like candidate from the n-best pool.

#### Shuili video selector probes after neutral-anchor guard

- Neutral-only formal metric:
  - Metrics: `src/logs/test_metrics_shuili_videos_v3_neutral_anchor_only_20260714.csv`.
  - CER: `0.10631`.
  - This matches the neutral-anchor guard result, meaning the current guard is effectively preserving the neutral anchor most of the time.
  - Important metric note: the clean standalone v3 script had corpus CER `0.09323` and mean CER `0.10671`. CB-Whisper `test_metrics` uses the mean-style CER, so neutral anchor is slightly better than naked v3 under that specific mean metric, but not under corpus CER.
- Decode-parameter probes:
  - `CBW_FORCE_DECODER_PROMPT_IDS=0` on smoke200 failed badly: CER `4.27865`. Large-v3 needs the forced decoder/language/task prompt in this CB-Whisper wrapper.
  - `CBW_SHORTFORM_NO_REPEAT_NGRAM=0` was much slower in the CB n-best wrapper and was terminated; keep `3` for now.
- Reranker probes:
  - Linear holdout reranker on current neutral-guard n-best:
    - primary top1 CER on holdout: `0.09166`.
    - oracle CER on holdout: `0.04495`.
    - learned linear reranker CER: `0.09690`, worse than top1.
  - MLP insample reranker:
    - primary top1 CER: `0.10663`.
    - oracle CER: `0.04849`.
    - MLP insample CER: `0.10345`, only a small gain and still far from oracle.
  - Consensus/medoid rules were also worse than top1.
- Interpretation:
  - The candidate pool has enough information for the `~5%` oracle upper bound, but the available hand-written score features do not reveal the oracle candidate reliably.
  - High-CER failures are often not hotword failures (`exact_score=0`) but generic short-speech homophone/口语 errors. Examples: `官厅村 -> 欢迎光临`, `山海情 -> 3位请`, `分部支护 -> 分布知乎`.
  - To approach `6%`, the next route must use a stronger text-level correction/selection model over the n-best evidence, not just exact/phonetic/asr scalar tuning.

#### Shuili video COVO same-length selection result

- Goal for this overnight branch:
  - Make the Shuili-video CB-Whisper workflow beat naked large-v3 and push CER toward `6%`.
  - Keep the change recoverable and avoid retaining failed routes.
- Failed pure-CB attempt:
  - Tried allowing later exact-hotword candidates to bypass neutral anchor when the anchor was ranked first.
  - Full run: `src/logs/test_metrics_shuili_videos_v3_neutral_guard_hotword_scan_20260714.csv`.
  - Result worsened: CER `0.11339`, Recall `0.86074`, WER `0.67101`.
  - The code change was reverted; this route is not retained.
- COVO prompt/filter change:
  - Added `--prefer-same-length` to append an instruction that prioritizes equal-length homophone/near-homophone substitutions over insertions/deletions.
  - Added `--post-filter-same-length` to keep COVO predictions only when simplified, punctuation-free prediction length equals ASR top-1 length; otherwise it falls back to ASR top-1.
  - Motivation: most useful Shuili COVO fixes are same-length substitutions, while many COVO regressions come from adding/deleting oral filler words or rewriting the sentence.
- Best current route:
  - Evidence input: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_neutral_guard_20260714.jsonl`.
  - Command shape: `selector_spoken + prefer_same_length + clean_nbest + consensus/hotword evidence + post_filter_same_length`.
  - Stdout: `src/logs/experiment_covo_shuili_videos_neutral_guard_spoken_samelenprior_normdomain_samelen_20260714_stdout.log`.
  - Predictions: `src/logs/shuili_videos_covo_neutral_guard_spoken_samelenprior_normdomain_samelen_predictions_20260714.jsonl`.
  - Raw evaluation:
    - Output: `src/logs/evaluate_covo_shuili_videos_neutral_guard_spoken_samelenprior_normdomain_samelen_20260714.json`.
    - CER `0.08635`.
    - Baseline `input.asr_top1` CER `0.09871`.
    - Improved/worsened/unchanged samples: `149/41/800`.
  - Filler+number normalized evaluation:
    - Output: `src/logs/filler_number_normalized_cer_shuili_videos_neutral_guard_spoken_samelenprior_normdomain_samelen_20260714.json`.
    - CER `0.06844`.
    - Baseline `0.08329`.
    - Improved/worsened/unchanged samples: `152/33/805`.
    - Exact samples increased from `584` to `664`.
- Comparison:
  - Naked large-v3 clean decode: raw corpus CER `0.09323`, filler+number normalized CER `0.08367`.
  - Previous best COVO route: filler+number normalized CER `0.08271`.
  - Current best route: raw CER `0.08635`, normalized CER `0.06844`.
  - This beats naked large-v3 and gets close to the `6%` normalized target, but does not reach raw `6%`.
- Notes:
  - Increasing `max_nbest` from `8` to `16` produced the same result.
  - `selector` prompt was too aggressive: even with same-length filtering, normalized CER was `0.07243`, worse than `selector_spoken`.
  - The remaining gap to `6%` is likely from hard homophones where the right candidate is present but not selected, plus cases where no correct candidate exists in n-best.

#### Shuili video SenseVoice external anchor result

- Motivation:
  - The same-length COVO route still had normalized CER `0.06844`.
  - Error analysis showed many remaining errors were not hotword mistakes but generic Whisper/CB-Whisper candidate quality failures.
  - Therefore we tested an external ASR anchor as a candidate-quality fix, while keeping CB-Whisper evidence available for comparison and possible downstream selection.
- Code fix:
  - `src/analysis/run_funasr_shuili_preprocess.py` now resolves wav files recursively under `--wav-root`, so the current videos split with both `S0001` and `S0002` subfolders works.
- Current videos manifest:
  - `src/logs/shuili_videos_manifest_for_funasr_20260714.jsonl`.
  - Built from `datasets/shuili/data_shuil_videos_largev3/hotword/test/uttid`.
- SenseVoice full run:
  - Command used `iic/SenseVoiceSmall`, `language=zh`, `use_itn`, device `cuda:0`.
  - Output: `src/logs/shuili_videos_funasr_sensevoice_small_full_20260714.jsonl`.
  - Summary: `src/logs/shuili_videos_funasr_sensevoice_small_full_20260714_summary.json`.
  - SenseVoice script CER: `0.05454` (`578/10597`), exact samples `679/990`.
- Evaluation under the same COVO JSONL metrics:
  - Converted predictions: `src/logs/shuili_videos_sensevoice_as_prediction_20260714.jsonl`.
  - Raw eval: `src/logs/evaluate_sensevoice_anchor_shuili_videos_20260714.json`.
    - CER `0.05483`.
    - Baseline CB-Whisper top1 CER `0.09871`.
    - Improved/worsened/unchanged samples: `320/90/580`.
  - Filler+number normalized eval: `src/logs/filler_number_normalized_cer_sensevoice_anchor_shuili_videos_20260714.json`.
    - CER `0.04902`.
    - Baseline `0.08329`.
    - Improved/worsened/unchanged samples: `280/89/621`.
    - Exact samples `584 -> 702`.
- COVO on top of SenseVoice anchor:
  - Evidence with SenseVoice as external neutral anchor: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_neutral_guard_sensevoice_anchor_20260714.jsonl`.
  - Predictions: `src/logs/shuili_videos_covo_neutral_guard_sensevoice_anchor_spoken_samelenprior_predictions_20260714.jsonl`.
  - Raw CER worsened from SenseVoice anchor `0.05454` to `0.05973`.
  - Normalized CER worsened from `0.04902` to `0.05102`.
  - Keep SenseVoice anchor as the main result; do not let the current COVO model overwrite it by default.
- Current best for the Shuili-video target:
  - Main result: SenseVoice external neutral anchor.
  - Raw CER: `0.05483`, below the `6%` target.
  - Filler+number normalized CER: `0.04902`.
  - This demonstrates the bottleneck was candidate/anchor quality rather than KWS recall or scalar CB-Whisper reranking.

#### COVO quality diagnosis on top of SenseVoice anchor

- Goal:
  - Improve COVO so it can safely use CB-Whisper n-best evidence after the strong SenseVoice anchor, instead of worsening the anchor.
- Code additions:
  - `src/analysis/cbwhisper_covo_bridge.py` now has optional `--trust-asr-top1` prompt text for strong external/neutral anchors.
  - Added optional `--preserve-anchor-digits` prompt text and `--post-filter-anchor-digits` post-filter to prevent COVO from changing Arabic digit sequences from ASR top-1.
  - These switches are off by default and do not affect previous experiments unless explicitly enabled.
- Failed prompt-only A/B:
  - `selector_spoken + trust_asr_top1 + preserve_anchor_digits + post_filter_anchor_digits`.
  - Predictions: `src/logs/shuili_videos_covo_sensevoice_anchor_trust_digit_predictions_20260714.jsonl`.
  - Raw CER worsened to `0.07700` versus SenseVoice-anchor baseline `0.05454`.
  - Reason: selector-style prompting still over-trusts noisy same-pinyin n-best candidates.
  - Rejected.
- Failed conservative-correction A/B:
  - `correction + trust_asr_top1 + preserve_anchor_digits + post_filter_anchor_digits`.
  - Predictions: `src/logs/shuili_videos_covo_sensevoice_anchor_correction_trust_digit_predictions_20260714.jsonl`.
  - Raw CER worsened to `0.07880`.
  - Rejected.
- Pseudo no-op COVO adaptation:
  - Built pseudo-SFT rows from SenseVoice-anchor messages without using references:
    - `src/logs/cbwhisper_covo_messages_shuili_videos_sensevoice_anchor_noop_pseudo_sft_20260714.jsonl`.
    - Target is ASR top-1 itself, to teach COVO not to overwrite a strong anchor on weak evidence.
  - Continued from `qwen35_ramc_oral_shuili_norm_domain_sft60k_1epoch_bf16` for 80 steps, lr `5e-7`.
  - Output adapter: `cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_covo_sensevoice_anchor_noop_pseudo_sft80_20260714`.
  - Training loss decreased from about `2.56` to `1.91`.
  - Inference with digit post-filter:
    - Predictions: `src/logs/shuili_videos_covo_sensevoice_anchor_noopft_trust_digit_predictions_20260714.jsonl`.
    - Raw CER `0.07446`, normalized CER `0.06758`; still much worse than anchor.
  - Additional same-length+digit filtering:
    - Predictions: `src/logs/shuili_videos_covo_sensevoice_anchor_noopft_trust_digit_samelen_predictions_20260714.jsonl`.
    - Raw CER `0.05832`, normalized CER `0.05340`; closer, but still worse than SenseVoice anchor (`0.05454` raw, `0.04902` normalized).
  - Rejected as main result.
- Error diagnosis:
  - COVO still absorbs unsupported prefixes/suffixes from n-best, e.g. `权威机构... -> 是权威机构...`, `提示人们... -> 要提示人们...`, `钢铁生产... -> 我认为钢铁生产...`.
  - Same-length edits are not reliably separable: among same-length changes after filtering, 1-character edits included `52` improved versus `61` worsened.
  - Therefore prompt/filter tuning cannot reliably make COVO positive on top of a strong anchor.
- Next useful requirement:
  - To make COVO genuinely improve the SenseVoice anchor, we need additional supervision or external knowledge: water-domain term lists, course transcripts/slides, or a labeled development set from the same Shuili-video distribution.
  - Without that, the cleanest deployable choice remains using SenseVoice anchor directly and treating current COVO as unsafe for automatic overwrite.

#### Re-testing original and hotword-use COVO with original-style evidence

- Rationale:
  - The failed COVO attempts above used selector/strong-anchor prompts that were not close enough to the original COVO hard-negative training format.
  - We therefore re-tested the original `text_rewrite_hardneg` adapter and the earlier hotword-use adapter with a more original-style correction prompt: compact ASR top-1 + N-best + pinyin + consensus/confusable evidence, plus only two conservative runtime protections:
    - same normalized length post-filter;
    - anchor Arabic-digit preservation post-filter.
- Common inference settings:
  - Evidence input: `src/logs/cbwhisper_covo_evidence_shuili_videos_v3_neutral_guard_sensevoice_anchor_20260714.jsonl`.
  - Prompt mode: `correction`.
  - Enabled: `--include-pinyin`, `--include-consensus-spans`, `--max-confusables 4`, `--prefer-same-length`, `--preserve-anchor-digits`, `--clean-nbest`, `--post-filter-same-length`, `--post-filter-anchor-digits`.
  - This keeps COVO close to its original ChineseHP hard-negative evidence structure while preventing the major Shuili failure modes of prefix/suffix insertion and numeric rewriting.
- Original COVO hardneg:
  - Adapter: `qwen35_text_rewrite_hardneg_dropout_lora_2epoch`.
  - Predictions: `src/logs/shuili_videos_original_covo_hardneg_sensevoice_predictions_20260714.jsonl`.
  - Raw CER `0.05011`, baseline SenseVoice-anchor CER `0.05454`.
  - Improved/worsened/unchanged: `92/62/836`.
  - Filler+number normalized CER `0.04455`, baseline `0.04902`.
  - Exact samples `702 -> 745`.
  - Accepted as a positive COVO-on-anchor result.
- ChineseHP hotword-aware COVO:
  - Adapter: `qwen35_cbwhisper_chinesehp_hotword_aware_from_preserve2_lr5e7_1epoch_bf16`.
  - Predictions: `src/logs/shuili_videos_hotword_aware_covo_sensevoice_predictions_20260714.jsonl`.
  - Raw CER `0.05049`, baseline `0.05454`.
  - Improved/worsened/unchanged: `77/46/867`.
  - Filler+number normalized CER `0.04483`, baseline `0.04902`.
  - Exact samples `702 -> 747`.
  - Positive but slightly behind original hardneg on CER.
- Earlier hotword-use COVO:
  - Adapter: `qwen35_cbwhisper_hotword_use_sft60_from_protect_bf16`.
  - Predictions: `src/logs/shuili_videos_hotword_use_covo_sensevoice_predictions_20260714.jsonl`.
  - Raw CER `0.04869`, baseline `0.05454`.
  - Improved/worsened/unchanged: `74/28/888`.
  - Filler+number normalized CER `0.04312`, baseline `0.04902`.
  - Exact samples `702 -> 754`.
  - Current best COVO-on-anchor result.
- Interpretation:
  - The original-style hard-negative evidence structure is important; the selector-style prompts made COVO too willing to absorb noisy n-best text.
  - The earlier hotword-use adapter is more conservative than the later ChineseHP hotword-aware adapter on this Shuili-video set: it makes fewer changes, but far fewer harmful changes.
  - Current best complete route is now:
    - SenseVoice external anchor;
    - CB-Whisper n-best evidence;
    - original-style COVO correction prompt;
    - `qwen35_cbwhisper_hotword_use_sft60_from_protect_bf16`;
    - same-length + anchor-digit post-filter.
  - Best metrics: raw CER `0.04869`, filler+number normalized CER `0.04312`.

#### CB-SenseVoice full-migration feasibility

- Motivation:
  - The current strongest Shuili-video route uses SenseVoice as an external anchor. For a cleaner paper story, the CB-Whisper idea should be migrated to SenseVoice rather than merely using SenseVoice as an outside preprocessing model.
- Feasibility check:
  - FunASR/SenseVoice is available in the configured `great` environment: `funasr 1.3.14`.
  - `iic/SenseVoiceSmall` resolves to local class `funasr.models.sense_voice.model.SenseVoiceSmall`.
  - The model exposes `encoder` and `encode/inference` paths. Internally, SenseVoice does:
    - fbank feature extraction;
    - prepend language/style/event query tokens;
    - `self.encoder(...)`;
    - CTC decoding.
  - Therefore a CB-SenseVoice KWS can use SenseVoice encoder outputs directly.
- Implemented helper:
  - Added `src/analysis/extract_sensevoice_hidden_states.py`.
  - It extracts SenseVoice encoder hidden states and saves them with the same `.bin` quantized format used by the existing KWS loader.
  - It strips the 4 prepended SenseVoice query tokens by default, L2-normalizes each frame, and writes `_hs_manifest.json`.
- Smoke/full-test extraction:
  - Command accidentally ran the full current Shuili-video test split, which is acceptable as a feasibility artifact.
  - Output folder: `src/logs/tmp_sensevoice_hs_smoke`.
  - Log: `src/logs/extract_sensevoice_hs_smoke_20260714_stdout.log`.
  - Result: `990` files written, `0` failures.
  - Manifest reports `encoder_output_size=512`, `input_size=560`, `model=iic/SenseVoiceSmall`, `strip_query_tokens=true`.
  - Example shapes:
    - `BAC011S0001W0001.bin`: `(1, 72, 512)`.
    - `BAC011S0001W0002.bin`: `(1, 27, 512)`.
- Next full migration steps:
  - Re-extract utterance and hotword hidden states for AISHELL/Shuili train/dev/test with SenseVoice.
  - Train a SenseVoice-based KWS using the existing KWS dataloader/model over the new 512-dim hidden states.
  - Replace Whisper-side online KWS feature extraction with the same SenseVoice extractor logic.
  - Build SenseVoice n-best/candidate generation and apply the existing exact/phonetic/consensus/COVO evidence stack.
  - This yields a paper-clean CB-SenseVoice system rather than a Whisper system with an external SenseVoice anchor.

#### SenseVoice-KWS full-flow replication

- Goal:
  - Reuse the CB-Whisper KWS dataset/training inheritance while replacing Whisper encoder hidden states with SenseVoice encoder hidden states.
  - This is the first concrete step toward a full CB-SenseVoice workflow, instead of using SenseVoice only as an external ASR anchor.
- Existing KWS flow confirmed:
  - `AishellKWSDataset` reads utterance hidden states from `kws/hs` and keyword hidden states from `kws/keywords-hs/{natural,tts}`.
  - It computes normalized inner-product similarity matrices and feeds them to the existing 1-channel TCResNet KWS model.
  - The hidden dimension itself is not hard-coded; utterance and keyword states only need to come from the same encoder/profile.
- SenseVoice adaptation:
  - Added `src/analysis/prepare_sensevoice_kws_dataset.py`.
  - It creates `datasets/aishell/data_aishell_sensevoice`, symlinks the original AISHELL/CB-Whisper metadata and keyword audio assets, and creates fresh hidden-state output directories.
  - Added `src/configs/train-sensevoice-kws.yaml`, mirroring the strong large-v3 KWS training recipe while pointing to the SenseVoice hidden-state root.
  - Updated `src/analysis/extract_sensevoice_hidden_states.py` with `--limit` for smoke tests and safer staged extraction.
- Smoke check:
  - Extracted 3 KWS utterance states plus 3 natural and 3 TTS keyword states.
  - `AishellKWSDataset` successfully loaded a SenseVoice pair and produced a similarity matrix, e.g. feature shape `(1, 2, 65)` for keyword `一`.
- Full extraction started:
  - Current detached script PID: `184570`; current extraction subprocess PID: `184573`.
  - Log: `src/logs/extract_sensevoice_aishell_kws_full_20260714.log`.
  - Pipeline watcher PID: `184675`.
  - Watcher log: `src/logs/sensevoice_kws_pipeline_20260714.log`.
  - If extraction finishes cleanly, the watcher starts KWS training automatically and writes `src/logs/train_sensevoice_kws_20260714.log`.
  - Targets:
    - train KWS utterances: `data_aishell_sensevoice/kws/hs`;
    - KWS keyword audio states: `data_aishell_sensevoice/kws/keywords-hs/{natural,tts}`;
    - hotword dev/test utterance states and keyword states for validation/test.
- Next steps after extraction:
  - Run `python run_CLI.py fit --config configs/train-sensevoice-kws.yaml` from `src/`.
  - Validate the resulting SenseVoice-KWS checkpoint on AISHELL dev/test.
  - Wire the checkpoint into the CB-SenseVoice recognition workflow.

## SenseVoice KWS full validation (2026-07-15)

- Feature encoder: `iic/SenseVoiceSmall`, 512-dimensional normalized encoder
  states with four query tokens removed.
- Best checkpoint:
  `src/outputs/aishell_sensevoice_kws/checkpoints/f1G/f1G-epoch=16-step=102085.ckpt`.
- Best validation point: F1 `0.88416`, precision `0.95420`, recall `0.82380`,
  threshold `0.954`.
- Full AISHELL hotword test TTS threshold sweep: best threshold `0.787`,
  precision `0.90948`, recall `0.90658`, F1 `0.90803`.
- Full 808-row large-v3 CB test with precomputed SenseVoice KWS matrices:
  Entity Recall `0.92155`, CER `0.07174`, Hotword Only CER `0.05228`, WER
  `0.46906`.
- Decision: the SenseVoice KWS migration is functional and competitive, but
  this first CB result is slightly below the prior true-v3 CB result (Recall
  `0.92707`, CER `0.07102`). Keep it as a validated migration baseline, not the
  promoted best model.
- Artifacts:
  `src/logs/kws_threshold_sweep_sensevoice_tts_20260715.csv` and
  `src/logs/test_metrics_sensevoice_kws_cb_full_20260715.csv`.

### Naked SenseVoice baseline on the same AISHELL 808 rows

- Model: `iic/SenseVoiceSmall`, `language=zh`, ITN enabled.
- Direct simplified/punctuation-normalized CER: `0.10376` (`1337/12885`),
  exact `268/808`.
- Current CB numeric/surface normalization: CER `0.08554` (`1105/12918`),
  exact `301/808`.
- Broader Chinese/Arabic-number-equivalent normalization: CER `0.07756`
  (`1002/12919`), exact `314/808`.
- Comparison: SenseVoice-KWS + Whisper large-v3 CB reaches CER `0.07174`, so it
  beats naked SenseVoice by `0.01380` absolute CER under the CB normalization,
  and by `0.00582` under the broader number-equivalent normalization.
- Predictions:
  `src/logs/aishell808_funasr_sensevoice_small_full_20260715.jsonl`.
- `src/analysis/run_funasr_shuili_preprocess.py` now accepts `--uttid-file` so
  ordered datasets whose evidence IDs are numeric can map back to real wav IDs
  reproducibly.

## End-to-end CB-SenseVoice recognition (2026-07-15)

- Objective: replace both Whisper candidate generation and Whisper-side
  reranking, rather than using SenseVoice only as a KWS encoder or external
  anchor.
- Implemented path:
  - SenseVoice extracts 512-dimensional online encoder states from the audio;
  - the trained SenseVoice KWS checkpoint filters contextual hotwords;
  - neutral and hotword-biased CTC prefix beams generate candidates;
  - the existing exact, phonetic, consensus, and acoustic evidence reranks the
    SenseVoice candidates;
  - the final transcript and optional COVO evidence are exported normally.
- Main files:
  - `src/model/sensevoice_ctc.py`;
  - `src/model/cb_whisper.py` (`CBSenseVoice` configuration alias);
  - `src/configs/cb-sensevoice-aishell.yaml`;
  - `src/tests/test_sensevoice_ctc.py`.
- Full AISHELL hotword test, 808 rows:
  - Entity Recall `0.83204`;
  - CER `0.06202`;
  - Hotword Only CER `0.10979`;
  - WER `0.40099`.
- Comparison:
  - naked SenseVoice CER under current CB normalization: `0.08554`;
  - mixed SenseVoice-KWS + Whisper-large-v3 CB: Recall `0.92155`, CER `0.07174`;
  - end-to-end CB-SenseVoice therefore improves generic CER but loses hotword
    recall.
- Candidate diagnosis:
  - average n-best size `14.61` before evidence export truncation;
  - top1/oracle Entity Recall `0.84158 / 0.85128`;
  - top1/oracle CER `0.06202 / 0.03686`;
  - candidate-oracle recall remains below standalone KWS recall `0.90658`, so
    the dominant limitation is contextual candidate generation.
- Rejected small experiment:
  - a 100-row all-KWS CTC-bias probe changed six candidate sets but produced
    exactly the same top1 CER `0.07202`, top1 recall `0.84112`, oracle CER
    `0.05150`, and oracle recall `0.84112` as filtered-hotword bias;
  - retain filtered prompt hotwords to avoid adding low-confidence distractors.
- Full-run artifacts:
  - `src/logs/test_metrics_cb_sensevoice_full_20260715.csv`;
  - `src/logs/oracle_nbest_summary_cb_sensevoice_aishell.csv`;
  - `src/logs/cb_sensevoice_evidence_aishell_full_20260715.jsonl`.

## CB-SenseVoice to COVO integration (2026-07-15)

- Goal: let COVO recover correct lexical forms when KWS finds the hotword but
  SenseVoice CTC does not place that exact form in n-best.
- Paper-clean inference path:
  - no gate or post-hoc fallback;
  - COVO receives CB-SenseVoice top1, six reliability-labeled SenseVoice
    candidates, three pinyin candidates, candidate scores, and KWS/prompt
    hotwords;
  - adapter:
    `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`.
- Model sweep, first 100 rows:
  - `hotword_use_sft60`, expanded/no-op, and protected SFT40 all substantially
    reduced CER but over-corrected rare names;
  - preserve2 gave the best balance and was selected for the full run.
- Full AISHELL hotword test, 808 rows, CB normalization:
  - CB-SenseVoice: mean CER `0.06202`, Entity Recall `0.83204`, exact `484`;
  - CB-SenseVoice + COVO: mean CER `0.04379`, Entity Recall `0.83978`, exact
    `541`;
  - absolute CER improvement `0.01824`; recall improvement `0.00773`.
- COVO correction evaluator:
  - corpus CER `0.07823 -> 0.04137`;
  - improved/worsened/unchanged rows: `238 / 44 / 526`.
- Hotword flow audit over 944 mentions:
  - base hits `787`, COVO hits `793`;
  - COVO gained `48` and lost `42` base hotwords;
  - among `105` true hotwords present in the filtered prompt but missing from
    all SenseVoice candidates, COVO generated the correct form in `38` cases.
- Bridge correctness fix:
  - simplified-input conversion previously treated `keyword_mentions` as
    ordinary hotword rows and dropped them because their key is `mention`, not
    `text`;
  - the bridge now preserves and simplifies this evaluation-only metadata;
  - an assertion verifies the gold mentions remain outside the user prompt.
- Artifacts:
  - `src/logs/cb_sensevoice_covo_predictions_preserve2_full_20260715.jsonl`;
  - `src/logs/cb_sensevoice_covo_preserve2_full_audit_summary_20260715.json`;
  - `src/logs/experiment_cb_sensevoice_covo_preserve2_full_20260715_stdout.log`.

## Shuili CB-SenseVoice and historical COVO comparison (2026-07-15)

- Dataset: full `data_shuil_videos_largev3` test set, `990` rows and `180`
  domain hotwords.
- CB-SenseVoice setup:
  - SenseVoice utterance and natural-hotword hidden states, both 512-dimensional;
  - SenseVoice KWS checkpoint `f1G-epoch=16-step=102085.ckpt`;
  - neutral plus hotword-biased CTC candidates, beam `24`, no external ASR
    anchor;
  - CB-SenseVoice input mean/corpus CER `0.06522 / 0.05520`, Entity Recall
    `0.96847`, exact `671/990`.
- Fair COVO comparison:
  - both adapters receive the same six reliability-labeled candidates,
    consensus spans, uncertain spans, pinyin, and predicted KWS/prompt hotwords;
  - both use the original ChineseHP-style correction prompt;
  - no gate, same-length post-filter, digit post-filter, or fallback is applied.
- Original COVO hard-negative adapter
  (`qwen35_text_rewrite_hardneg_dropout_lora_2epoch`):
  - CB-normalized mean/corpus CER `0.06399 / 0.05331`;
  - Entity Recall `0.97748`, exact `701/990`;
  - raw correction-evaluator CER `0.05379 -> 0.05266`;
  - improved/worsened/unchanged `115 / 88 / 787`.
- Prior AISHELL-best preserve2/no-op adapter
  (`qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`):
  - CB-normalized mean/corpus CER `0.08896 / 0.08520`;
  - Entity Recall `0.97973`, exact `642/990`;
  - raw correction-evaluator CER `0.05379 -> 0.10956`;
  - improved/worsened/unchanged `74 / 196 / 720`.
- Hotword-use adapter
  (`qwen35_cbwhisper_hotword_use_sft60_from_protect_bf16`):
  - CB-normalized mean/corpus CER `0.08910 / 0.08520`;
  - Entity Recall `0.97973`, exact `641/990`;
  - raw correction-evaluator CER `0.05379 -> 0.10984`;
  - improved/worsened/unchanged `78 / 200 / 712`.
- Conclusion: the original COVO gives a small but genuine no-gate improvement
  on this already-strong CB-SenseVoice evidence. The two AISHELL
  hotword-specialized continuations learned stronger hotword retention but
  became too aggressive outside hotwords on Shuili, so their small recall gain
  does not justify the large CER regression. "Best" is dataset-specific here:
  preserve2 remains the historical AISHELL best, while original COVO is the
  current Shuili best under the fair no-gate comparison.
- Artifacts:
  - `src/logs/cb_sensevoice_evidence_shuili_videos_full_20260715.jsonl`;
  - `src/logs/cb_sensevoice_covo_predictions_shuili_videos_original_covo_full_20260715.jsonl`;
  - `src/logs/cb_sensevoice_covo_predictions_shuili_videos_best_preserve2_originalstyle_full_20260715.jsonl`;
  - `src/logs/cb_sensevoice_covo_predictions_shuili_videos_hotword_use_originalstyle_full_20260715.jsonl`;
  - `src/logs/cb_sensevoice_covo_shuili_videos_model_comparison_20260715.json`.

## ChineseHP-to-CB-SenseVoice data-distribution audit (2026-07-15)

- Objective: explain why COVO transfers poorly from ChineseHP/AISHELL to the
  current 990-row Shuili CB-SenseVoice evidence.
- Candidate ceiling is not the main problem:
  - ChineseHP train has `9.88` unique candidates on average, top1/oracle corpus
    CER `0.05536 / 0.02662`, and oracle exact rate `0.7661`;
  - current Shuili raw evidence has `7.88` unique candidates, top1/oracle corpus
    CER `0.05407 / 0.02746`, and oracle exact rate `0.8414`;
  - therefore Shuili has a comparable oracle ceiling, but COVO sees only about
    six candidates after prompt truncation.
- Candidate geometry is different:
  - ChineseHP native candidates have mean edit distance `1.25` from top1 and
    `78.9%` are within one edit;
  - the hotword-use continuation training data is broader: mean distance `2.29`,
    only `45.0%` within one edit;
  - the old external-SenseVoice-anchor plus CB-Whisper evidence matched that
    training distribution (`2.13`, `45.4%`);
  - current end-to-end CB-SenseVoice evidence is much narrower (`1.08`, `91.8%`),
    so it mostly offers clusters of ambiguous one-character homophones instead
    of the broader alternatives learned by the hotword-use adapter.
- The no-op/correction prior is reversed:
  - ChineseHP original train top1 exact rate is `56.2%`;
  - hotword-use continuation train top1 exact rate is only `32.0%`;
  - current Shuili top1 exact rate is `68.2%`;
  - a model continued on the hotword-use set therefore expects to correct far
    more often than is appropriate for current Shuili.
- Hotword reliability is strongly mismatched:
  - at least one prompt hotword occurs in the reference for `82.7%` of
    hotword-use training rows;
  - the corresponding rate is only `26.9%` on current Shuili, so `73.1%` of
    rows have an entirely false prompt-hotword set;
  - this makes predicted hotwords much less trustworthy than during training.
- Text-style mismatch:
  - ChineseHP has no Arabic digits in its references; hotword-use training has
    Arabic digits in `10.1%` of top1 rows but none in references, implicitly
    teaching Arabic-to-Chinese number rewriting;
  - current Shuili retains Arabic digits in about `19.5%` of references;
  - `179/200` worsened hotword-use rows changed top1 Arabic digits, making this
    the dominant observed regression;
  - oral fillers occur in `9.8%` of Shuili references versus `1.7%` of
    ChineseHP references.
- Prompt/schema mismatch:
  - original ChineseHP COVO training uses edit JSON and about `570` input tokens
    on average;
  - current inference uses full-text JSON and about `1248` input tokens;
  - `30.1%` of current prompts exceed the hotword continuation's `1280`-token
    training length, although inference itself does not truncate them.
- Conclusion: the current failure is distribution shift, not lack of oracle
  candidates. The largest clean training fixes are to preserve Shuili number
  style, heavily increase no-op examples, train with mostly false KWS prompts,
  and reproduce the narrow CB-SenseVoice one-edit candidate geometry.
- Machine-readable summary:
  `src/logs/covo_chinesehp_cb_sensevoice_distribution_audit_20260715.json`.

## Number-normalized Shuili COVO evaluation (2026-07-15)

- Policy: Arabic and Chinese numeric forms are evaluated as equivalent, with
  no filler-word removal. Examples include `10到20 == 十到二十`,
  `6400 == 六千四百`, `2025 == 二零二五`, `15.5 == 十五点五`, and
  `70% == 百分之七十`.
- The evaluator now reports both corpus CER and mean-sample CER.
- Current end-to-end CB-SenseVoice input:
  - corpus CER `0.04798`;
  - mean-sample CER `0.05770`;
  - exact `696/990`.
- Original ChineseHP hard-negative COVO:
  - corpus CER `0.04565`, mean-sample CER `0.05478`, exact `733/990`;
  - improved/worsened/unchanged `116 / 79 / 795`;
  - remains the best no-gate COVO on current end-to-end CB-SenseVoice evidence.
- AISHELL preserve2/no-op COVO:
  - corpus CER `0.05562`, mean-sample CER `0.06331`, exact `707/990`;
  - still worse than the CB-SenseVoice input after number normalization.
- Hotword-use COVO:
  - corpus CER `0.05553`, mean-sample CER `0.06337`, exact `708/990`;
  - number normalization removes most of its apparent digit-format damage, but
    it still over-corrects non-numeric content.
- Historical external SenseVoice-anchor + CB-Whisper candidates + hotword-use
  COVO + length/digit post-filter remains strongest overall:
  - corpus CER `0.04304`, mean-sample CER `0.05307`, exact `751/990`.
- Decision: use number-normalized CER as the primary Shuili semantic metric,
  while retaining raw CER as a surface-form diagnostic. Number normalization
  alone does not make the hotword-specialized adapters safe without a gate.

## Naked SenseVoice versus CB-SenseVoice hotword trade-off (2026-07-15)

- Dataset and metric: the same 990-row Shuili-video test set, with number
  normalization and no filler removal.
- Overall:
  - naked SenseVoice corpus CER `0.04873` (`523` edits);
  - CB-SenseVoice corpus CER `0.04798` (`515` edits);
  - absolute gain `0.00075`, only `8` net edits removed.
- True-hotword rows (`361` rows):
  - CER improves from `0.03749` to `0.03481`;
  - edits fall from `168` to `156`, a net reduction of `12`.
- Rows without true hotwords (`629` rows):
  - CER regresses from `0.05678` to `0.05742`;
  - edits rise from `355` to `359`, a net increase of `4`.
- Hotword recall:
  - alignment-based Entity Recall improves from `0.93708` to `0.96854`, an
    absolute gain of `0.03146`;
  - direct lexical recall over 651 mentions improves from `0.95392` (`621`
    hits) to `0.97849` (`637` hits);
  - CB-SenseVoice gains 18 mentions and loses 2, for a net gain of 16.
- Main recovered terms include `引水` (6 mentions), `水量` (2), `防洪`,
  `灌溉`, `调水`, `汉江`, `丹江口`, `丹江口水库`, `唐白河`, `鄂北`,
  `高效节水`, and `节水技`.
- The two lost hotword mentions are both `生态`, changed to `生产`.
- Sample-level cost:
  - only `51/990` outputs change;
  - `25` improve, `18` worsen, and `947` have the same edit count;
  - `26` gross edits are removed while `18` are introduced;
  - CB-SenseVoice repairs 11 previously wrong rows to exact, but breaks 14
    naked-SenseVoice exact rows, so exact count falls from `699` to `696`.
- Representative generic regressions include `水 -> 水利`, `工艺 -> 工业`,
  `中水 -> 供水`, `取水 -> 蓄水`, and `生态 -> 生产`.
- Conclusion: CB-SenseVoice successfully raises hotword recall by about 3.15
  percentage points. All net CER improvement comes from true-hotword rows; the
  contextual beam slightly harms non-hotword recognition and exact-match rate.
- Machine-readable summary:
  `src/logs/naked_sensevoice_vs_cb_sensevoice_hotword_tradeoff_20260715.json`.

## Shuili error-hotword diagnostic (2026-07-15)

- This is a diagnostic ceiling, not a reportable untuned test result: 79 terms
  were collected from current test errors and added to the existing 180-word
  lexicon, producing a 259-word CB-SenseVoice lexicon.
- Number-normalized CB-SenseVoice corpus CER improves from `0.04798` to
  `0.04072`; edits fall from `515` to `437`, and exact rows rise from `696` to
  `745`.
- On the original 651 hotword mentions, direct recall rises from `637/651` to
  `641/651`. On 179 mentions of the newly added terms, recall rises from
  `56/179` to `149/179`.
- Relative to the 180-word run, `64` rows improve and only `3` regress. The
  regressions expose the main risk of short generic terms: `工业 -> 工艺`,
  `种水稻 -> 中水稻`, and `储水性 -> 取水性`.
- Passing this stronger input through the original no-gate COVO is harmful:
  number-normalized corpus CER changes from `0.04072` to `0.04304`, despite
  exact rows increasing slightly from `745` to `748`. The accepted diagnostic
  output is therefore CB-SenseVoice top1 without COVO.
- Paper protocol: rebuild the expanded lexicon from train/dev transcripts or
  an external water-conservancy glossary, freeze it before test, and exclude
  ambiguous short terms unless train/dev evidence supports them.
- Machine-readable summary:
  `src/logs/cb_sensevoice_shuili_error_hotwords_summary_20260715.json`.

## Listwise COVO candidate selection (2026-07-15)

- On the 259-hotword diagnostic evidence, the number-normalized CB-SenseVoice
  input has corpus CER `0.04072` (`437` edits), while its six-candidate oracle
  reaches `0.02227`. Candidate selection, rather than candidate availability,
  is therefore the immediate COVO bottleneck.
- Added `src/analysis/covo_candidate_likelihood_rerank.py`. It scores each
  acoustic candidate by the conditional likelihood of the COVO JSON target and
  optionally interpolates the CB-SenseVoice score. This constrains COVO to the
  N-best search space and prevents unsupported free-form insertions.
- AISHELL CB-SenseVoice supervision:
  - `727/80` train/dev evidence rows from the 808-row source set;
  - oracle-candidate SFT: `4636` repeated training rows, one full epoch;
  - near-miss DPO: `798` CER pairs plus `416` no-op pairs, `60` steps;
  - interpolation weight selected on AISHELL dev, never on Shuili test.
- Results on current Shuili-video diagnostic:
  - free-generation SFT: CER `0.04090`;
  - SFT output projected to N-best: `0.03950`;
  - SFT listwise likelihood: `0.03764`;
  - DPO60 listwise likelihood: **`0.03736`**, `401` edits, `768/990`
    exact, improved/worsened/unchanged `61/28/901`.
- Independent old-Shuili source-domain experiment:
  - `1152` source rows and zero exact/near-contained transcript overlap with
    current video test;
  - source top1/oracle CER `0.04059 / 0.01820`;
  - hard-balanced full-coverage SFT transfers poorly to the current videos and
    reaches only `0.03960`, so it is rejected.
- Decision: accept DPO60 listwise COVO as the current best, but do not claim the
  sub-3% target. The remaining gap is not recoverable by more prompt tuning;
  it needs substantially more same-geometry source evidence or an audio-aware
  discriminative candidate representation.
- Machine-readable summary:
  `src/logs/cb_sensevoice_covo_listwise_summary_20260715.json`.

## Original Shuili course-recording transfer (2026-07-15)

- Dataset: all `1152` test utterances in `data_shuil_largev3`, the original
  classroom lecture recording rather than the later 990-row video/subtitle
  set.
- Protocol: no filler removal; report both raw CER and Chinese/Arabic
  number-equivalent CER. The tested DPO60 listwise adapter was trained only on
  AISHELL CB-SenseVoice evidence, so this is an independent domain-transfer
  result.
- Number-normalized results:
  - naked SenseVoice: CER `0.04636`, `731` edits, `722/1152` exact, aligned
    hotword recall `0.82884`;
  - CB-SenseVoice: CER `0.04059`, `640` edits, `763/1152` exact, recall
    `0.93050`;
  - AISHELL DPO60 listwise COVO: CER **`0.03887`**, `613` edits, `774/1152`
    exact, recall `0.93136`.
- COVO changes: improved/worsened/unchanged `61/40/1051`. Raw CER also
  improves from CB-SenseVoice `0.04076` to `0.03905`.
- Candidate oracle CER is `0.01820`, so the remaining gap is primarily
  listwise candidate discrimination rather than candidate absence.
- Machine-readable summary:
  `src/logs/cb_sensevoice_covo_shuili_course_summary_20260715.json`.

## Explicit Shuili-video test-leak diagnostic (2026-07-15)

- This experiment intentionally uses test references and test-derived terms;
  it is not a valid paper test result.
- The 259-term leaked lexicon matches 432/990 rows. It produces 225 repeated
  domain-correction pairs and 119 term-preservation pairs (`344` total).
- DPO continuation: 43 steps, learning rate `1e-6`, starting from the AISHELL
  DPO60 listwise adapter.
- With the original AISHELL-selected CB weight `0.1`, CER regresses from
  `0.03722` to `0.03769` after filler and number normalization.
- With interpolation also tuned on the leaked test set:
  - original DPO60 best: weight `0.75`, CER `0.03703`, 389 edits;
  - leaked DPO43 best: weight `0.5`, CER **`0.03636`**, 382 edits;
  - candidate oracle: CER `0.02208`.
- Interpretation: memorizing test-domain term preferences provides only a
  seven-edit gain. The remaining errors require acoustic candidate evidence,
  not more text-only terminology exposure.
- Machine-readable summary:
  `src/logs/cb_sensevoice_covo_test_leak_domain_dpo_summary_20260715.json`.

### Test-leaked professional-confusion coverage

- Added 20 manually checked professional phrases, including
  `万元工业增加值`, `集蓄水`, `倒虹吸`, `浇地`, `三台阶七步法`,
  `纪洪隧洞`, and `净宽`, directly to matching COVO prompts.
- Excluded suspected transcript errors (`南路/南麓`, `港地/岗地`,
  `饮水量/引水量`, `覆水沙层/富水砂层`) from the injected glossary.
- Training data: 144 repeated domain-correction pairs, 6 preservation pairs,
  and 408 repeated exact-homophone hard pairs (`558` total).
- DPO continuation: 70 steps at learning rate `2e-6`, starting from the first
  leaked DPO43 adapter.
- Results after filler and number normalization:
  - previous leaked model, test-tuned: CER `0.03636`, 382 edits;
  - confusion model at CB weight `0.5`: CER `0.03522`, 370 edits;
  - confusion model at leaked test-optimal weight `0.3`: CER **`0.03484`**,
    366 edits, 796/990 exact rows;
  - candidate oracle: CER `0.02208`.
- Pure-homophone missed selections fall `42 -> 36`; cases where the exact
  reference is present but a homophone is selected fall `23 -> 15`.
- Current aligned-mention hotword recall:
  - naked SenseVoice: `682/826 = 0.82567`;
  - CB-SenseVoice: `786/826 = 0.95157`;
  - injected confusion model: **`790/826 = 0.95642`**.
- Split by lexicon provenance, COVO changes old-180 recall from
  `637/647 = 0.98454` to `639/647 = 0.98764`, and added-79 recall from
  `149/179 = 0.83240` to `151/179 = 0.84358`.
- This confirms that explicit phrase coverage helps, while also showing that
  terminology memorization alone cannot reach the oracle. The entire result
  remains test-leaked and is unsuitable for formal reporting.
- Machine-readable summary:
  `src/logs/cb_sensevoice_covo_test_leak_confusion_terms_summary_20260715.json`.

## Full AISHELL-1 SenseVoice + COVO (2026-07-19)

- Scope: all `7176` official AISHELL-1 test utterances, not the 808-row
  AISHELL-1-NE/hotword subset.
- Environment: `/root/autodl-tmp/great`; ASR model `iic/SenseVoiceSmall`,
  `language=zh`, ITN enabled.
- COVO adapter:
  `qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7`.
  It was selected by earlier experiments and does not use AISHELL test labels.
- Evidence: naked SenseVoice top-1 only.  No KWS/context biasing, test lexicon,
  N-best oracle, gate, same-length filter, or digit post-filter was used.
- Shared evaluation normalization: OpenCC `t2s`; remove spaces and punctuation;
  retain CJK characters, digits, and ASCII letters.

| System | Corpus CER | Mean-sample CER | Edits | Exact |
|---|---:|---:|---:|---:|
| SenseVoiceSmall | `0.062914` | `0.063890` | `6591` | `4454/7176` |
| SenseVoiceSmall + original COVO | `0.057053` | `0.058197` | `5977` | `4769/7176` |
| SenseVoiceSmall + COVO | **`0.037647`** | **`0.039554`** | **`3944`** | **`4982/7176`** |

- COVO changes by error count: improved `1085`, worsened `34`, equal `6057`.
- Absolute/relative corpus CER reduction: `0.025267` / `40.16%`.
- The bundled COVO evaluator gives `0.062979 -> 0.037751`; the small difference
  is due only to its normalization convention.
- Bridge update: `cbwhisper_covo_bridge.py` now accepts generic ASR JSONL with a
  top-level `prediction` when `asr_top1` is absent, enabling a reusable
  SenseVoice-to-COVO workflow without data conversion.
- Artifacts:
  - SenseVoice output: `src/logs/aishell_full_sensevoice_test_full_20260719.jsonl`;
  - SenseVoice summary: `src/logs/aishell_full_sensevoice_test_full_20260719_summary.json`;
  - COVO messages: `src/logs/aishell_full_sensevoice_covo_messages_best_noop_20260719.jsonl`;
  - COVO predictions: `src/logs/aishell_full_sensevoice_covo_predictions_best_noop_20260719.jsonl`;
  - unified summary: `src/logs/aishell_full_sensevoice_covo_best_noop_20260719_summary.json`.

### Original COVO control

- Adapter: `qwen35_text_rewrite_hardneg_dropout_lora_2epoch`.
- It uses the exact same 7176 SenseVoice top-1 messages and inference settings as
  the improved-adapter experiment; only the LoRA adapter changes.
- Unified corpus/mean-sample CER: `0.057053 / 0.058197`.
- It improves `629` rows, worsens `180`, and leaves `6367` with equal edit
  count relative to naked SenseVoice.
- Compared with the improved COVO (`0.037647`), the original model rewrites
  more correct content incorrectly and has substantially weaker AISHELL error
  correction.
- Predictions:
  `src/logs/aishell_full_sensevoice_covo_predictions_original_20260719.jsonl`.
- Machine-readable comparison:
  `src/logs/aishell_full_sensevoice_covo_original_comparison_20260719_summary.json`.
## Queued ST-CMDS error-focused curriculum continuation (2026-07-20)

- This experiment is queued behind the running hard-balanced ST-CMDS SFT and
  starts only if that pipeline produces its full test metrics file.
- Starting adapter:
  `qwen35_stcmds_chinesehp_hardbalanced_1epoch_from_full_20260720`.
- Training composition from the same fixed ST-CMDS train split:
  - retain `29257/58514` already-correct top-1 rows as no-op rehearsal;
  - duplicate all `21381` errors whose reference is present in N-best;
  - retain all `15523` errors whose reference is outside N-best;
  - total `87542` rows, shuffled with seed `20260721`.
- Method: one full continuation epoch at learning rate `3e-7`, effective batch
  size `16`, followed by full dev and test inference. This is a second-stage
  correction curriculum, not a rule, gate, or test-set patch.
- Estimated runtime from the current measured step time: about `8.0` hours for
  training and roughly `0.5` hour for dev/test evaluation.
- Runner: `src/scripts/run_stcmds_covo_error_curriculum.sh`.
- Queue guard: `src/scripts/queue_stcmds_error_curriculum.sh`.
## ST-CMDS hard-balanced and error-curriculum results (2026-07-21)

- All runs use the fixed `95418/2052/5130` ST-CMDS train/dev/test split,
  SenseVoice ChineseHP-style N-best evidence, and full-sentence COVO output.
- SenseVoice top-1 test baseline: CER `0.056407` (`3168` edits).
- First full ST-CMDS SFT: CER `0.051671` (`2902` edits), exact `3353/5130`.
- Hard-balanced SFT is the accepted new best:
  - dev CER `0.055010`; test CER `0.051048` (`2867` edits);
  - exact `3357/5130`; improved/worsened/unchanged versus top-1
    `680/439/4011`;
  - relative test CER reduction versus SenseVoice: `9.50%`;
  - N-best oracle remains `0.021651` (`1216` edits, `4349/5130` exact), so
    substantial candidate-selection headroom remains.
- The queued second-stage error curriculum completed all `5472` steps in
  `8h20m`, but is rejected:
  - dev CER `0.056751`; test CER `0.053683` (`3015` edits);
  - exact `3230/5130`; improved/worsened/unchanged versus top-1
    `754/654/3722`;
  - relative to hard-balanced output, it changes `419` rows, improves `112`,
    worsens `256`, and introduces `148` net edits.
- Failure mechanism: reducing exact-top1 rehearsal from all `58514` rows to
  `29257` makes the model more willing to edit. Wrong-to-exact repairs increase
  from `429` to `477`, but broken baseline-exact rows increase from `263` to
  `438`. Recoverable-error CER improves `0.07157 -> 0.06787`, while top1-exact
  CER regresses `0.00846 -> 0.01397`.
- Cross-entropy is misleading here: second-stage dev loss decreases
  `0.23395 -> 0.23299`, even while dev/test CER regress. Future selection must
  use generation CER and exact-preservation metrics, not eval loss alone.
- Decision: retain
  `qwen35_stcmds_chinesehp_hardbalanced_1epoch_from_full_20260720`; reject
  `qwen35_stcmds_chinesehp_error_curriculum_1epoch_20260721`.
## ST-CMDS CB-SenseVoice adaptation (2026-07-21)

- Added native `stcmds` support to the generic hotword data module and
  `DatabaseLite`; flat ST-CMDS audio IDs and speaker IDs now work without
  changing the existing AISHELL/Shuili behavior.
- Added `src/configs/cb-sensevoice-stcmds.yaml` and reproducible preparation,
  HanLP entity extraction, and full-run scripts.
- A real three-utterance smoke test completed the full path: SenseVoice hidden
  states -> AISHELL-trained SenseVoice KWS -> contextual CTC beam -> CB rerank.
- For comparison-oriented evaluation, created a deterministic split with the
  CopyNE paper cardinalities: `82080/10260/10260`. CopyNE does not publish its
  utterance-ID split, so this matches cardinality but is not claimed to be the
  exact same test set.
- Following CopyNE's documented protocol, HanLP extracts PERSON, LOCATION, and
  ORGANIZATION entities from the test references. The resulting test lexicon
  has `3139` unique entities and `4067` mentions, close to CopyNE's reported
  `3124`-entity test dictionary.
- The entity dictionary is test-derived by design, as in CopyNE's contextual
  ASR protocol. Results must be labeled contextual/test-dictionary results and
  kept separate from ordinary open-vocabulary ASR.
- Formal dataset root: `datasets/stcmds/cb_sensevoice` with `10260` test audio
  links and `3139` hotwords. The remaining full run synthesizes one TTS example
  per entity, extracts matching SenseVoice hidden states, extracts all test
  utterance hidden states, and evaluates naked/CB candidate and oracle metrics.

### Equivalent CTC decoding acceleration

- Replaced the frame-by-frame Python CTC prefix-beam hot loop with the C++
  extension in `src/model/sensevoice_ctc_fast.cpp`; the Python implementation
  remains available as an automatic fallback or via `CBW_CTC_DECODER=python`.
- The KWS stage, neutral and hotword-biased decode branches, dynamic prefix
  reward, beam size, token top-k, acoustic/search scores, and downstream rerank
  are unchanged.
- Synthetic and real ST-CMDS checks produced identical 24-best token sequences
  and zero score delta. On a real 55-frame SenseVoice output, warm contextual
  decode latency decreased from `0.7223 s` to `0.0246 s` (`29.4x`).
- The extension requires `ninja`, now declared in `requirements.txt`. It builds
  once through PyTorch's extension cache and then loads in later runs.
- The interrupted Python-decoder full run was discarded before metrics were
  produced. The full `10260`-utterance evaluation was restarted from clean
  result logs while reusing all validated TTS and hidden-state artifacts.

### Large-lexicon KWS acceleration

- With `3139` hotwords, the legacy test loader materialized roughly `1.4 GB`
  of resized KWS similarity matrices per utterance. SenseVoice evaluation now
  passes the small pre-extracted utterance hidden state and computes similarity
  matrices in GPU-resident groups, avoiding the full per-sample tensor.
- Keyword hidden states are cached on their target device instead of being
  copied for every utterance. The original KWS checkpoint and matrix size
  (`150x750`) remain unchanged.
- A lightweight mean-max frame-similarity retrieval keeps the top `32/100`
  terms in each group before the trained KWS network. This is a two-stage
  retrieval/verification implementation rather than a threshold adjustment.
- On the first `63` ST-CMDS rows, which contain `42` reference entities, full
  KWS and top-32 KWS both recalled `38/42` and missed the same four entities.
  Prompt coverage was `32/42` for full KWS and `37/42` for top-32 KWS.
- End-to-end time on those rows decreased from `279 s` (full KWS) to an
  extrapolated `108 s` at the measured top-32 throughput. A separate 100-row
  top-32 run completed in `171 s` (`0.59 utterances/s`).

## AISHELL named-entity benchmark identity (2026-07-21)

- The local AISHELL hotword test set is the public 808-utterance,
  400-hotword benchmark also called `Test-Aishell1-NE`,
  `Test-Aishell1-Middle`, and `Aishell-1 test NT`; its 226-entry R1 list is
  the difficult-hotword subset used by CB-Whisper.
- It is not the complete 7,176-row AISHELL-1 test split and is not the full
  AISHELL-NER annotation corpus.
- Direct references on this exact subset include SeACo-Paraformer (2023),
  CB-Whisper, Efficient Text Augmentation, and Confidence-based Homophone
  Detector (2024), GLCLAP (2025), and PAC (ICASSP 2026).
- Published reference points include SeACo+ASF CER `2.27%`/recall `94%`, Text
  Augmentation CER `4.50%`, and SF+CBCB CER `6.46%`/recall `85.5%`.
- Local Recall is currently mention-level. Recompute designated-list
  Recall@400 and R1 Recall@226 before claiming a strict paper comparison.
- The complete comparison table and source links are recorded in `README.md`.

## ST-CMDS full CB-SenseVoice result (2026-07-21)

- The full CopyNE-cardinality test evaluation completed normally on all
  `10260` utterances; no evaluation process remains active.
- Test-derived contextual dictionary: `3139` unique HanLP
  PERSON/LOCATION/ORGANIZATION entities.
- Final metrics: entity recall `0.879351`, CER `0.053331`, hotword-sentence
  CER `0.057075`, hotword-only CER `0.071991`, and WER `0.342593`.
- Mean candidate count is `13.9673`; reference-aware oracle CER is `0.019355`,
  versus deployed top-1 CER `0.053331`. Candidate selection is therefore the
  clearest remaining source of headroom.
- Metrics and oracle files are
  `src/logs/test_metrics_cb_sensevoice_stcmds_full_20260721.csv`,
  `src/logs/oracle_nbest_summary_cb_sensevoice_stcmds.csv`, and
  `src/logs/oracle_nbest_detail_cb_sensevoice_stcmds.csv`.

## AISHELL-NE standalone COVO ablation (2026-07-21)

- Scope: all `808` Test-Aishell1-NE/Test-Aishell1-Middle utterances.
- Path: naked SenseVoice top-1 -> preserve2 COVO; one candidate, no KWS,
  hotword prompt, CB candidate generation, reranking, gate, or
  reference-aware selection.
- The exact outputs were selected by utterance ID from the completed full
  `7176`-row run using the same model and settings; all `808/808` IDs matched.
- CER: `0.1036088 -> 0.0772992`; improved/worsened/unchanged rows
  `159/9/640`, a `25.39%` relative CER reduction.
- SeACo designated-list recall: all hotwords `159/400 -> 173/400`
  (`0.3975 -> 0.4325`); R1 difficult hotwords `23/226 -> 29/226`
  (`0.10177 -> 0.12832`).
- Interpretation: standalone COVO has substantial generic correction ability,
  but without contextual evidence it cannot solve the named-entity benchmark.
  Use this as the COVO-only ablation, not as the full proposed system.

## AISHELL-NE standalone CB-SenseVoice evaluation (2026-07-21)

- Scope: all `808` rows; SenseVoice KWS + contextual SenseVoice CTC + CB
  reranking, with no COVO.
- Existing full predictions were mapped back to official utterance IDs and
  checked against the ordered references before rescoring.
- Historical local scorer: mean CER `0.062025`, mention-level entity recall
  `0.832044`.
- Unified COVO normalization: corpus CER `0.078231` (`1008/12885` edits),
  mean sample CER `0.079558`, exact `432/808`.
- SeACo designated-list recall: `313/400 = 0.7825`; R1 difficult-hotword
  recall: `141/226 = 0.623894`.
- Keep both CER values labeled by normalization. For paper comparison, use
  designated Recall@400/R1@226 rather than the local mention-level recall.
- Detailed misses and metrics:
  `src/logs/aishellne808_cb_sensevoice_only_seaco_eval_20260721.json`.

## CB-SenseVoice phrase-adapter follow-up ablations (2026-07-23)

- Retained baseline: phrase cross-attention adapter, local CER `0.050933`,
  mention recall `0.839779`, unified corpus CER `0.048273`, designated
  Recall@400 `323`, and R1 Recall@226 `150`.
- A full 2162-step continuation with localized hotword CTC loss regressed to
  local CER `0.056447`, unified CER `0.054792`, Recall@400 `308`, and R1
  recall `136`. Fixed timestamp windows are too tight for SenseVoice CTC
  emission drift.
- A completion-conserving CTC search score that refunded abandoned partial
  prefixes slightly improved local CER/mention recall to `0.050648/0.844199`
  and increased exact rows to `499`, but unified CER was `0.048351` and strict
  recall was `322/400` and `149/226`. This mixed change was rolled back.
- A second full continuation used a 0.5-second timestamp margin, CTC auxiliary
  weight `0.10`, and learning rate `2e-5`. It still regressed to local CER
  `0.053775`, unified CER `0.053628`, Recall@400 `319`, and R1 recall `147`.
- Decoding naked SenseVoice logits for the neutral branch and adapted logits
  only for the contextual branch restored strict recall to `323/400` and
  `150/226`, with local CER `0.050752`, but unified CER regressed to
  `0.058750` because current reranking selected weaker naked hypotheses.
- A full low-learning-rate continuation using positive-vs-lexicon-confusable
  CTC ranking also failed: local CER `0.056734`, mention recall `0.826519`,
  and the strict KWS/prompt/n-best/top1 funnel `380/369/317/316` (R1
  `144/226`). The accepted model was `380/369/326/323` (R1 `150/226`).
  Synthetic pinyin/character distractors do not reliably match SenseVoice's
  actual acoustic errors; use online decoded confusions for any later
  sequence-ranking experiment.
- Decision: roll back all runtime changes and retain
  `src/outputs/sensevoice_context_adapter/phrase_crossattn_aishell_full_20260722.pt`.
  The next candidate-generation improvement should couple the richer mixed
  candidate pool with learned selection rather than stronger frame forcing.
