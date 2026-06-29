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
