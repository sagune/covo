# Experiment Log

Use this file as the durable ledger for non-trivial runs. Keep short failed
experiments too; they are useful for deciding what not to repeat.

## Template

```text
Date:
Commit:
Dataset:
ASR source:
Correction method:
Model / checkpoint:
Config:

Metrics:
- Baseline CER:
- Final CER:
- Improved samples:
- Worsened samples:
- Entity Recall:
- Hotword Only CER:

Notes:
- What changed?
- What got better?
- What got worse?
- Decision: keep / revise / reject
- Next step:
```

## Runs

### 2026-05-20 - Repository bootstrap

Commit range:

```text
cac8a3d -> e11e08d
```

Summary:

- Added project plan, evidence schema, data conversion, verifier/fallback, evaluation, Whisper-v3 entry point, LoRA skeleton, DeepSeek teacher skeleton, rule baseline, and CI.
- No model training has been run yet.

Current validation:

```text
python -m unittest discover -s tests
python -m compileall src scripts tests
```

### 2026-05-20 - Config smoke run

Commit range:

```text
after e11e08d
```

Dataset:

```text
ChineseHP AISHELL-1 sample, first 20 records
```

Config:

```text
configs/sample_prepare_chinesehp.json
```

Outputs:

```text
outputs/pipeline/chinesehp_sample_sft.jsonl
outputs/pipeline/chinesehp_sample_sft.clean.jsonl
outputs/pipeline/chinesehp_sample_qwen_messages.jsonl
outputs/logs/sample_prepare_chinesehp.json
```

Metrics:

- Samples: 20
- Avg baseline CER: 0.03231617647058824
- Empty edit rate: 0.65
- Avg edit count: 0.35

Decision: keep.

### 2026-06-08 - Hard-negative disagreement text rewrite

Commit:

```text
10444eb Add hard-negative ASR rewrite data builder
```

Dataset:

```text
ChineseHP AISHELL-1 test, 7174 records
```

ASR source:

```text
Whisper N-best evidence already prepared in data/processed/chinesehp_aishell1
```

Correction method:

```text
Qwen3.5-4B LoRA text rewrite.
Input uses ASR top-1, N-best, pinyin, consensus spans, and automatically mined
confusable candidates from N-best. Training uses evidence dropout.
```

Training:

```text
train: train_text_rewrite_hardneg_dropout.qwen.jsonl
dev:   dev_text_rewrite_hardneg.qwen.jsonl
test:  test_text_rewrite_hardneg.qwen.jsonl
LoRA: r=16, alpha=32, dropout=0.05
effective batch: 28
epochs: 2
final train loss: 0.186
final dev eval loss: 0.1496
```

Metrics:

| Method | Checkpoint | CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ASR top-1 | none | 0.05835 | - | - | - | - |
| ASR-only direct LoRA | 1 epoch | 0.03765 | 0.05835 | 1874 | 361 | 4939 |
| Consensus text rewrite | 1 epoch | 0.02945 | 0.05835 | 2303 | 290 | 4581 |
| Hard-negative + dropout | checkpoint-4000 | 0.03023 | 0.05835 | 2324 | 364 | 4486 |
| Hard-negative + dropout | final 2 epoch | 0.02777 | 0.05835 | 2384 | 296 | 4494 |

Outputs:

```text
outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch/
outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch/test_predicted_text_ckpt4000.jsonl
outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch/test_predicted_text_final.jsonl
```

Notes:

- Final 2-epoch hard-negative model improves over the previous consensus-only model: CER 0.02945 -> 0.02777.
- Checkpoint-4000 has more improved samples than consensus but also much more worsened samples, so it is not preferred.
- The second epoch is useful in this setting: final CER improves and worsened samples drop from 364 to 296.
- Decision: keep final 2-epoch hard-negative + evidence-dropout model as current best.
- Next step: analyze remaining worsened cases and decide whether a lightweight confidence gate is worth adding.

### Continuation From 0.02777 Model

Goal: continue training from the current best 2-epoch hard-negative LoRA to see whether a smaller, targeted continuation pass can reduce remaining worsened cases.

Training:

```text
base adapter: outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
train: train_text_rewrite_hardneg_continuation.qwen.jsonl
rows: 52000
sampling focus: random base rows + no-change confusable rows + ref-absent free-correction rows + ref-present N-best correction rows
max_steps: 1300
effective batch: 28
learning rate: 5e-5
final train loss: 0.1543
```

Metrics:

| Method | Checkpoint | CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hard-negative + dropout | final 2 epoch | 0.02777 | 0.05835 | 2384 | 296 | 4494 |
| Continuation from 0.02777 | checkpoint-650 | 0.02928 | 0.05835 | 2367 | 382 | 4425 |
| Continuation from 0.02777 | final 1300 | 0.02869 | 0.05835 | 2376 | 349 | 4449 |

Outputs:

```text
outputs/qwen35_text_rewrite_hardneg_continuation_from277_1300/
outputs/qwen35_text_rewrite_hardneg_continuation_from277_1300/test_predicted_text_ckpt650.jsonl
outputs/qwen35_text_rewrite_hardneg_continuation_from277_1300/test_predicted_text_final.jsonl
```

Notes:

- The continuation pass did not improve over the current best. Both checkpoint-650 and final-1300 increased worsened samples.
- The final-1300 checkpoint is better than checkpoint-650, but still worse than the original 2-epoch model.
- Decision: keep `outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch` as the current best model.
- Interpretation: the targeted continuation set increased the model's willingness to edit, but did not improve edit discrimination enough. The next improvement should use preference-style training or explicit no-change contrastive supervision rather than plain SFT continuation.

### DPO Preference Probe From 0.02777 Model

Goal: test whether preference learning can reduce over-correction by training the current best adapter to prefer reference/keep outputs over tempting wrong alternatives.

Training:

```text
base adapter: outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
train: train_text_preference_dpo_20000.jsonl
rows: 20000
pair mix: 9000 no-change vs confusable, 5000 ref-in-N-best vs ASR, 6000 free-correction vs ASR/confusable
max_steps: 200
learning rate: 1e-5
beta: 0.1
```

Partial test probe:

| Method | Test subset | CER | Baseline CER | Improved | Worsened | Unchanged | Parse warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DPO from 0.02777 | first 3403 | 0.05431 | 0.05411 | 49 | 33 | 3321 | 3195 |

Outputs:

```text
outputs/qwen35_text_dpo_from277_200/
outputs/qwen35_text_dpo_from277_200/test_predicted_text_final.jsonl
```

Notes:

- The run was stopped after the partial test probe because the model usually emitted plain Chinese text instead of the required `{"text":"..."}` JSON object.
- The high parse-warning count caused predictions to fall back to ASR top-1, so CER returned close to the ASR baseline.
- Interpretation: a pure DPO loss over chosen/rejected final texts does not preserve output format because the shared JSON wrapper contributes little to the preference margin. Future DPO attempts need an SFT-format anchor, a smaller learning rate/beta, or a mixed DPO+SFT objective.

### DPO + SFT Anchor Probe

Goal: fix the DPO formatting failure by adding a chosen-response SFT loss to the DPO objective.

Training:

```text
base adapter: outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
train: train_text_preference_dpo_20000.jsonl
max_steps: 60
learning rate: 5e-6
beta: 0.05
sft_weight: 0.2
```

Partial test probe:

| Method | Test subset | CER | Baseline CER | Improved | Worsened | Unchanged | Parse warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard-negative + dropout | first 512 | 0.02605 | 0.04727 | 135 | 18 | 359 | 0 |
| Pure DPO from 0.02777 | first 512 | 0.04633 | 0.04727 | 9 | 4 | 499 | 478 |
| DPO + SFT anchor | first 512 | 0.02820 | 0.04727 | 139 | 34 | 339 | 0 |

Outputs:

```text
outputs/qwen35_text_dpo_sftanchor_from277_60/
outputs/qwen35_text_dpo_sftanchor_from277_60/test_predicted_text_limit512.jsonl
```

Notes:

- The SFT anchor fixes the JSON-format collapse: parse warnings drop from 478/512 to 0/512.
- The DPO+SFT probe still underperforms the current best on CER because worsened samples increase.
- Decision: keep the SFT anchor in the DPO trainer, but do not promote this checkpoint. Next DPO attempt should reduce edit aggressiveness, likely by increasing no-change preference weight and/or using a smaller DPO update.

### Adaptive Evidence Selection Probe

Goal: test the 2026-style adaptive evidence idea: simple utterances should receive less evidence to avoid over-correction, while harder utterances can receive richer N-best/confusable evidence.

Setup:

```text
base adapter: outputs/qwen35_text_rewrite_hardneg_dropout_lora_2epoch
dev subset: first 1024 records
test subset: first 1024 records
E0: ASR-only prompt
E1: consensus prompt
E2: hard-negative prompt
selector: RandomForest risk gate trained on dev to predict E2 over-correction risk
fallback when risky: E0
```

Test subset results:

| Method | CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: |
| E0 ASR-only | 0.03977 | 165 | 17 | 842 |
| E1 consensus | 0.02604 | 314 | 36 | 674 |
| E2 hard-negative | 0.02504 | 321 | 36 | 667 |
| Oracle adaptive E0/E1/E2 | 0.02122 | 334 | 4 | 686 |
| RF risk gate E2 -> E0 | 0.02497 | 320 | 35 | 669 |

Outputs:

```text
outputs/adaptive_evidence_probe/
outputs/adaptive_evidence_probe/adaptive_selector_test1024.json
```

Notes:

- The oracle gap is large: choosing among evidence levels can theoretically reduce CER from 0.02504 to 0.02122 on the first 1024 test samples, and reduce worsened samples from 36 to 4.
- A simple input-feature risk gate gives only a tiny realized gain: CER 0.02504 -> 0.02497, worsened 36 -> 35.
- Decision: keep the adaptive-evidence route as promising, but the selector needs stronger supervision/features before full-scale evaluation. A useful next step is to train a dedicated lightweight selector on dev/train labels rather than relying on coarse N-best statistics.

Additional score-free confidence simulation:

```text
confidence inputs: top-1 exact support, average N-best similarity, unique-hypothesis ratio,
stable-span coverage, uncertain-span support, variant penalty
selection rule: choose E0 if c0 >= lambda0, else E1 if c1 >= lambda1, else E2
best dev thresholds: lambda0=0.58, lambda1=0.72
```

| Method | Test CER | Improved | Worsened | Counts |
| --- | ---: | ---: | ---: | --- |
| Fixed E2 hard-negative | 0.02504 | 321 | 36 | E2=1024 |
| Score-free confidence E0/E1/E2 | 0.02611 | 314 | 36 | E1=1006, E2=18 |
| Risk-style alpha=0.008 | 0.02625 | 312 | 37 | E0=35, E1=989 |

Conclusion: the direct score-free analogue of the 2026 risk-control method does not help on the 1024-sample probe. Without real ASR/CB-Whisper scores, hand-built pseudo confidence mostly collapses to the consensus prompt and loses the stronger hard-negative gains. For CB-Whisper integration, prefer using actual decoder/KWS scores or train a selector from oracle evidence-level labels rather than relying only on N-best agreement heuristics.

## AISHELL-1 Qwen3.5-9B Hard-Negative + Evidence-Dropout LoRA (2026-08-17)

Training reused the AISHELL-1 4B route with the same 120,082-row hard-negative/evidence-dropout data and LoRA shape (`r=16`, `alpha=32`, dropout `0.05`), but used Qwen3.5-9B, effective batch 8, and two epochs (30,022 steps). Final train loss was 0.1691.

Checkpoint selection first screened all nine saved checkpoints on a uniformly sampled 1,024-row dev subset. The top three were then evaluated on the complete 14,326-row dev set.

| Checkpoint | Full dev CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-30022 | **0.022894** | 0.053375 | 4610 | 516 | 9200 |
| checkpoint-30000 | 0.023030 | 0.053375 | 4598 | 514 | 9214 |
| checkpoint-15000 | 0.023897 | 0.053375 | 4497 | **432** | 9397 |

The lowest-CER checkpoint, `checkpoint-30022`, was selected for the complete 7,174-row test set.

| Method | Test CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B hard-negative + dropout, 2 epochs | 0.027770 | 0.058354 | 2384 | 296 | 4494 |
| Qwen3.5-9B hard-negative + dropout, checkpoint-30022 | **0.026231** | 0.058354 | **2425** | **282** | 4467 |

The 9B model reduces test CER by 0.001539 absolute (0.1539 percentage points) and about 5.54% relative versus the 4B result. It also improves 41 additional utterances and reduces worsened utterances by 14. This confirms a modest but consistent gain from scaling the correction LLM on the same AISHELL-1 route.

Outputs:

```text
outputs/qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022/
outputs/qwen35_9b_aishell_checkpoint_selection_20260817/full_dev_ranking.tsv
outputs/qwen35_9b_aishell_checkpoint_selection_20260817/final_summary.json
outputs/qwen35_9b_aishell_checkpoint_selection_20260817/test/checkpoint-30022.predictions.jsonl
```

### AISHELL-1 CB-SenseVoice + Qwen3.5-9B COVO (2026-08-17)

The selected 9B adapter (`checkpoint-30022`) was evaluated on the existing 7,176-row AISHELL error-targeted CB-SenseVoice COVO message file. This reuses exactly the same messages and evaluator as the historical 4B combination run; no acoustic/KWS stage was rerun.

| Method | CER | COVO baseline CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical Qwen3.5-4B + CB-SenseVoice | **0.018336** | 0.026345 | 890 | **198** | 6088 |
| Qwen3.5-9B checkpoint-30022 + CB-SenseVoice | 0.018460 | 0.026345 | **1150** | 504 | 5522 |
| 9B combination + same-length fallback | 0.018422 | 0.026345 | 1056 | 394 | 5726 |
| 9B combination + anchor-digit fallback | 0.019205 | 0.026345 | 1081 | 502 | 5593 |
| 9B combination + both fallbacks | 0.019129 | 0.026345 | 991 | 394 | 5791 |

The direct CB-SenseVoice test metric from its original evaluator is CER 0.034067. Under the COVO normalization used for the paired comparison, its message top-1 baseline is 0.026345. The raw 9B combination reduces that paired baseline by about 29.93%, and is also substantially better than naked 9B COVO CER 0.026231. However, it is 0.000124 absolute (0.0124 percentage points, about 0.68% relative) worse than the historical 4B combination. The 9B adapter makes more useful corrections but is much more willing to edit CB-SenseVoice output, producing 306 additional worsened utterances. Same-length fallback recovers a small amount but does not beat 4B; digit protection is harmful here.

Decision: retain Qwen3.5-4B as the best CB-SenseVoice combination model on this route, while retaining Qwen3.5-9B checkpoint-30022 as the best naked AISHELL COVO model. Scaling helped the original ChineseHP-style input but did not improve the distribution-shifted CB-SenseVoice prompt without additional conservatism or route-specific training.

Outputs:

```text
/root/autodl-tmp/src/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_predictions_20260817.jsonl
/root/autodl-tmp/src/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_summary_20260817.json
/root/autodl-tmp/src/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_postfilter_20260817_same_length.summary.json
/root/autodl-tmp/src/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_postfilter_20260817_anchor_digits.summary.json
/root/autodl-tmp/src/logs/cb_sensevoice_aishell_error_targeted_covo_9b_ckpt30022_postfilter_20260817_same_length_anchor_digits.summary.json
```

### ST-CMDS Qwen3.5-9B COVO (2026-08-20)

Qwen3.5-9B was trained for two epochs on the 138,180-row ST-CMDS CB hard-negative set. Nine checkpoints were screened on a uniformly sampled 513-row dev subset; the top three were then evaluated on the complete 2,052-row dev set.

| Checkpoint | Full dev CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint-34400 | **0.047553** | 0.059073 | **317** | **125** | 1610 |
| checkpoint-34546 | 0.048044 | 0.059073 | 312 | 126 | **1614** |
| checkpoint-30100 | 0.048223 | 0.059073 | 314 | 135 | 1603 |

The best checkpoint, `checkpoint-34400`, was evaluated on the complete 5,130-row test set.

| Method | Test CER | Baseline CER | Improved | Worsened | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical ST-CMDS Qwen3.5-4B route | 0.052650 | - | - | - | - |
| Qwen3.5-9B checkpoint-34400 | **0.044994** | 0.056407 | 828 | 355 | 3947 |

The selected 9B adapter reduces CER by 0.007656 absolute (0.7656 percentage points), about 14.54% relative, versus the historical 4B result. Because the historical result was recorded from an earlier route whose baseline was about 5.7921%, while this test input has a 5.6407% baseline, treat the comparison as strongly favorable but not a perfectly controlled model-size ablation.

Outputs:

```text
outputs/qwen35_9b_stcmds_cb_hardnegative_2epoch_20260817/checkpoint-34400/
outputs/qwen35_9b_stcmds_checkpoint_selection_20260820/full_dev_ranking.tsv
outputs/qwen35_9b_stcmds_checkpoint_selection_20260820/final_summary.json
outputs/qwen35_9b_stcmds_checkpoint_selection_20260820/test/checkpoint-34400.predictions.jsonl
```

### ST-CMDS CB-SenseVoice + Qwen3.5-9B COVO ablation (2026-08-20)

The selected ST-CMDS 9B adapter (`checkpoint-34400`) was evaluated on two existing 5,130-row CB-SenseVoice message routes. The acoustic/KWS stage was not rerun, so the 9B and historical 4B comparisons reuse identical messages and references.

| Route | Model | CER | Paired baseline CER | Improved | Worsened | Unchanged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original ST-CMDS input | naked Qwen3.5-9B | **0.044994** | 0.056407 | 828 | 355 | 3947 |
| Error-targeted CB-SenseVoice | direct CB evaluator | 0.045447 | - | - | - | - |
| Error-targeted ChineseHP messages | historical Qwen3.5-4B | 0.045421 | 0.044834 | 414 | 464 | 4252 |
| Error-targeted ChineseHP messages | Qwen3.5-9B checkpoint-34400 | **0.044994** | 0.044834 | **447** | **462** | 4221 |
| Standard-3139 ChineseHP messages | historical Qwen3.5-4B | 0.052757 | 0.057921 | 541 | **316** | **4273** |
| Standard-3139 ChineseHP messages | Qwen3.5-9B checkpoint-34400 | **0.049054** | 0.057921 | **753** | 388 | 3989 |

On the error-targeted route, 9B improves slightly over the historical 4B combination (0.0433 percentage points) but still worsens the paired CB-SenseVoice baseline by 0.0160 percentage points. Its CER happens to equal the naked 9B result, although these are different input routes and baselines. On standard-3139, 9B improves over 4B by 0.3704 percentage points (about 7.02% relative) and reduces the paired baseline by 0.8867 percentage points (about 15.31% relative), but it remains worse than naked 9B by 0.4060 percentage points.

Conclusion: scaling the correction model improves both CB-SenseVoice combination routes, especially standard-3139, but CB-SenseVoice does not improve the best ST-CMDS 9B result. The strongest final ST-CMDS configuration remains naked Qwen3.5-9B checkpoint-34400 at CER 4.4994%.

Outputs:

```text
outputs/qwen35_9b_stcmds_cbsense_ablation_20260820/final_summary.json
outputs/qwen35_9b_stcmds_cbsense_ablation_20260820/error_targeted_chinesehp.predictions.jsonl
outputs/qwen35_9b_stcmds_cbsense_ablation_20260820/standard3139_chinesehp.predictions.jsonl
```
