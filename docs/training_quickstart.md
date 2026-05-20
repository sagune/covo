# LoRA Training Quickstart

This repository does not vendor CoVoGER. It uses the same high-level idea
(`N-best -> generative correction`) but keeps a project-specific pipeline:

```text
ChineseHP / Whisper-v3 / CB-Whisper evidence
-> internal SFT JSONL
-> Qwen-style messages JSONL
-> LoRA / QLoRA training
-> model output parser
-> verifier / fallback
```

## 1. Build Internal SFT Data

```bash
PYTHONPATH=src python scripts/prepare_chinesehp_sft.py \
  --input data/raw/chinesehp_aishell1.jsonl \
  --output data/processed/chinesehp_aishell1_sft.jsonl \
  --nbest-size 5 \
  --output-mode edits
```

## 2. Filter Training Records

```bash
PYTHONPATH=src python scripts/filter_sft_jsonl.py \
  --input data/processed/chinesehp_aishell1_sft.jsonl \
  --output data/processed/chinesehp_aishell1_sft.clean.jsonl \
  --max-baseline-cer 0.8 \
  --max-total-changed-chars 24 \
  --max-edits 4
```

## 3. Export Qwen Messages

```bash
PYTHONPATH=src python scripts/export_sft_format.py \
  --input data/processed/chinesehp_aishell1_sft.clean.jsonl \
  --output data/processed/chinesehp_aishell1_qwen_messages.jsonl \
  --format qwen-messages
```

## 4. Train LoRA / QLoRA

Install training dependencies on the GPU machine:

```bash
pip install -r requirements-train.txt
```

Dry-run formatting first:

```bash
PYTHONPATH=src python scripts/train_lora_sft.py \
  --train-file data/processed/chinesehp_aishell1_qwen_messages.jsonl \
  --output-dir outputs/qwen_asr_corrector_dryrun \
  --model-name-or-path Qwen/Qwen3-4B \
  --input-format qwen-messages \
  --dry-run
```

Example QLoRA command:

```bash
PYTHONPATH=src python scripts/train_lora_sft.py \
  --train-file data/processed/chinesehp_aishell1_qwen_messages.jsonl \
  --output-dir outputs/qwen_asr_corrector_lora \
  --model-name-or-path Qwen/Qwen3-4B \
  --input-format qwen-messages \
  --qlora \
  --bf16 \
  --gradient-checkpointing \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 1
```

The model name is intentionally configurable. Use whichever local Qwen/DeepSeek
small model is available on the 5090 machine.

## 5. Inference and Verification

```bash
PYTHONPATH=src python scripts/infer_lora_edits.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl \
  --output outputs/sample_model_edits.jsonl \
  --model-name-or-path Qwen/Qwen3-4B \
  --adapter-path outputs/qwen_asr_corrector_lora \
  --input-format internal

PYTHONPATH=src python scripts/apply_edits_jsonl.py \
  --input outputs/sample_model_edits.jsonl \
  --output outputs/sample_model_applied.jsonl

PYTHONPATH=src python scripts/evaluate_correction_jsonl.py \
  --input outputs/sample_model_applied.jsonl
```
