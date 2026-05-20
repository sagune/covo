# covo

CB-Whisper 后纠错实验计划与流程文档。

阶段策略：前期先用 Whisper large-v3 作为 ASR 参考系统，跑通 N-best 纠错、训练和 verifier；后期再接入 CB-Whisper 的 KWS、热词和重排序证据。

相关仓库：

- CB-Whisper 当前进度：<https://github.com/sagune/cbwhisper>

当前入口：

- [CB-Whisper 受约束后纠错计划](docs/cbwhisper_correction_plan.md)
- [LoRA 训练快速开始](docs/training_quickstart.md)
- [服务器运行手册](docs/server_runbook.md)
- [Evidence JSONL 字段规范](docs/evidence_schema.md)
- [实验日志](docs/experiment_log.md)

## 配置化运行

常用流程已经放在 `configs/` 里，统一用 `scripts/run_pipeline.py` 调起。每个配置可以写入 `log_file`，默认落到 `outputs/logs/`，记录每一步命令、开始/结束时间、耗时和返回码。

先检查命令：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/sample_prepare_chinesehp.json \
  --dry-run
```

实际跑一个 20 条样本的准备流程：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/sample_prepare_chinesehp.json
```

当前配置：

- `configs/sample_prepare_chinesehp.json`：ChineseHP 样本准备、过滤、导出 Qwen messages、统计。
- `configs/prepare_chinesehp_full.json`：ChineseHP 全量准备、过滤、按原始 split 切分、导出 train/dev/test。
- `configs/baselines_chinesehp_test.json`：在 test 集上评测 no-correction、oracle N-best 和 rule baseline。
- `configs/whisper_v3_decode.json`：Whisper large-v3 生成 N-best，需音频和 Whisper 依赖。
- `configs/deepseek_teacher_sample.json`：DeepSeek teacher 生成 edits、应用、评测，需环境变量 `DEEPSEEK_API_KEY`。
- `configs/qwen_lora_5090.json`：5090/GPU 环境上的 Qwen LoRA/QLoRA 训练入口，使用 train/dev。
- `configs/evaluate_qwen_lora_test.json`：加载 LoRA adapter，在 test 集推理、应用 edits、评测和分析。

服务器上的推荐顺序：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/prepare_chinesehp_full.json

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/baselines_chinesehp_test.json

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/qwen_lora_5090.json

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/evaluate_qwen_lora_test.json
```

训练完成后会保存 LoRA adapter、tokenizer 文件和训练元信息：

```text
outputs/qwen_asr_corrector_lora/
outputs/qwen_asr_corrector_lora/training_metadata.json
```

评测配置默认从 `outputs/qwen_asr_corrector_lora` 加载 adapter，base model 仍使用配置里的 `Qwen/Qwen3-4B`。

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

数据切分、统计和纠错分析：

```bash
PYTHONPATH=src python scripts/split_jsonl.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl \
  --output-dir outputs/splits \
  --mode field

PYTHONPATH=src python scripts/dataset_stats.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl

PYTHONPATH=src python scripts/analyze_corrections.py \
  --input outputs/sample_applied.jsonl \
  --output-dir outputs/analysis
```

Manifest、规则 baseline 和 DeepSeek teacher：

```bash
PYTHONPATH=src python scripts/build_manifest.py \
  --text data/local/text \
  --audio-dir data/local/wav \
  --output data/manifests/local.jsonl

PYTHONPATH=src python scripts/rule_correct_jsonl.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl \
  --output outputs/rule_corrected.jsonl

PYTHONPATH=src python scripts/evaluate_baselines_jsonl.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl

DEEPSEEK_API_KEY=... PYTHONPATH=src python scripts/generate_teacher_edits_deepseek.py \
  --input examples/chinesehp_aishell1_sft_sample.jsonl \
  --output outputs/deepseek_teacher.jsonl \
  --limit 5
```
