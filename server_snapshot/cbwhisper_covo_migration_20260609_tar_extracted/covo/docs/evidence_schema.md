# Evidence Schema

This document defines the JSONL record shape shared by Whisper-v3, later
CB-Whisper exports, rule baselines, teacher generation, and LoRA inference.

## Internal SFT Record

```json
{
  "id": "utt_id",
  "source": "chinesehp/aishell-1",
  "split": "train",
  "task": "asr_constrained_correction",
  "instruction": "根据 ASR 输出、N-best 候选和拼音证据进行中文 ASR 后纠错。",
  "input": {
    "asr_top1": "ASR one-best text",
    "nbest": ["hypothesis 1", "hypothesis 2"],
    "nbest_scores": [-0.1, -0.3],
    "nbest_pinyin": ["pin yin 1", "pin yin 2"],
    "asr_top1_pinyin": "pin yin 1",
    "hotwords": [
      {"text": "热词", "score": 0.92, "source": "kws"}
    ]
  },
  "output": {
    "edits": [
      {"from": "错词", "to": "正确词", "reason": "same_pinyin"}
    ]
  },
  "reference": "ground truth text"
}
```

## Required Fields

- `id`: stable utterance id.
- `input.asr_top1`: current ASR top hypothesis.
- `input.nbest`: ordered ASR hypotheses; first item should match `asr_top1`.
- `reference`: ground truth text when available.

## Optional Fields

- `input.nbest_scores`: ASR sequence scores or log probabilities.
- `input.nbest_pinyin`: pinyin strings aligned with `nbest`.
- `input.hotwords`: KWS/CB-Whisper hotword evidence.
- `metadata`: dataset-specific fields such as speaker, audio path, or domain.

## Prediction Record

Model inference adds:

```json
{
  "model_output": "raw model text",
  "predicted_edits": {"edits": []},
  "parse_warnings": []
}
```

After verifier/application:

```json
{
  "prediction": "final text after safe edits or fallback",
  "correction_result": {
    "accepted": true,
    "text": "final text",
    "reasons": [],
    "applied_edits": []
  }
}
```

## Safety Contract

The verifier is intentionally conservative:

- `from` must be present in `asr_top1` or N-best evidence.
- `to` must be supported by N-best, hotwords, reference-side evidence, or same-pinyin evidence.
- total changed characters must remain below the configured threshold.
- invalid JSON or unsafe edits must fall back to `asr_top1`.
