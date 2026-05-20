# covo

CB-Whisper 后纠错实验计划与流程文档。

阶段策略：前期先用 Whisper large-v3 作为 ASR 参考系统，跑通 N-best 纠错、训练和 verifier；后期再接入 CB-Whisper 的 KWS、热词和重排序证据。

相关仓库：

- CB-Whisper 当前进度：<https://github.com/sagune/cbwhisper>

当前入口：

- [CB-Whisper 受约束后纠错计划](docs/cbwhisper_correction_plan.md)
- [LoRA 训练快速开始](docs/training_quickstart.md)

## 当前可用工具

```bash
python scripts/prepare_chinesehp_sft.py \
  --input ../PostASR-Correction-SLT2024-main/chinesehp_aishell1.jsonl \
  --output data/processed/chinesehp_aishell1_sft.jsonl \
  --nbest-size 5 \
  --output-mode edits
```

这会把 ChineseHP AISHELL-1 的 `reference / nbest / nbest_pinyin` 转成后续 LoRA/SFT 可用的数据格式。样例见：

- [examples/chinesehp_aishell1_sft_sample.jsonl](examples/chinesehp_aishell1_sft_sample.jsonl)

本地检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src scripts tests
```

应用 edits 并评测：

```bash
PYTHONPATH=src python scripts/apply_edits_jsonl.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl \
  --output outputs/sample_applied.jsonl

PYTHONPATH=src python scripts/evaluate_correction_jsonl.py \
  --input outputs/sample_applied.jsonl
```

Whisper large-v3 N-best 生成脚本已经预留，需在有 GPU 和音频文件的环境中安装额外依赖：

```bash
pip install -r requirements-whisper.txt
PYTHONPATH=src python scripts/transcribe_whisper_v3.py \
  --manifest data/manifests/aishell_sample.jsonl \
  --output data/processed/whisper_v3_aishell_sample.jsonl
```

导出 Qwen chat SFT 格式、过滤训练数据、解析模型输出：

```bash
PYTHONPATH=src python scripts/filter_sft_jsonl.py \
  --input data/processed/chinesehp_aishell1_sft.jsonl \
  --output data/processed/chinesehp_aishell1_sft.clean.jsonl \
  --max-baseline-cer 0.8 \
  --max-total-changed-chars 24

PYTHONPATH=src python scripts/export_sft_format.py \
  --input data/processed/chinesehp_aishell1_sft.clean.jsonl \
  --output data/processed/chinesehp_aishell1_qwen_messages.jsonl \
  --format qwen-messages

PYTHONPATH=src python scripts/parse_model_edits_jsonl.py \
  --input outputs/model_raw_outputs.jsonl \
  --output outputs/model_parsed_edits.jsonl
```

LoRA/QLoRA 训练脚本已预留，需在 5090/GPU 环境安装训练依赖：

```bash
pip install -r requirements-train.txt
PYTHONPATH=src python scripts/train_lora_sft.py --help
PYTHONPATH=src python scripts/infer_lora_edits.py --help
```
