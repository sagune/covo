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
