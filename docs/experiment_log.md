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
