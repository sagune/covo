# Server Runbook

这份文档写给服务器上的 Codex/操作者。目标是把 ChineseHP AISHELL-1 文本侧后纠错实验完整跑起来：准备数据、跑基线、训练 Qwen LoRA、测试集推理评测，并确认模型与日志已经保存。

## 0. 当前阶段

当前阶段**不需要音频**。先用 `chinesehp_aishell1.jsonl` 里的：

- `reference`
- `nbest`
- `nbest_pinyin`

完成文本侧 ASR 后纠错训练与验证。CB-Whisper 的输出和热词/KWS 证据后期再接入。

## 1. 拉代码

```bash
git clone https://github.com/sagune/covo.git
cd covo
```

如果仓库已经存在：

```bash
cd covo
git pull origin main
```

确认最新代码里至少有这些配置：

```bash
ls configs
```

应该看到：

```text
prepare_chinesehp_full.json
baselines_chinesehp_test.json
qwen_lora_5090.json
evaluate_qwen_lora_test.json
```

## 2. 准备环境

建议新建一个干净环境：

```bash
conda create -n covo python=3.10 -y
conda activate covo
```

先安装轻量依赖：

```bash
pip install -r requirements.txt
```

训练服务器上再装训练依赖：

```bash
pip install -r requirements-train.txt
```

检查 GPU：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

5090 上建议确认 `torch`/CUDA/bitsandbytes 版本能正常加载 4bit。若 bitsandbytes 报 CUDA 兼容问题，先解决环境，不要改实验脚本。

## 3. 数据位置

仓库已经带了压缩后的 ChineseHP AISHELL-1 文本数据：

```text
data/raw/chinesehp_aishell1.jsonl.zip
```

`configs/prepare_chinesehp_full.json` 的第一步会自动解压到：

```text
data/raw/chinesehp_aishell1.jsonl
```

解压后的 JSONL 不提交进 git，只在服务器本地使用。

## 4. 先做 dry-run

所有正式命令前先 dry-run，看路径和参数是否正确：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/prepare_chinesehp_full.json \
  --dry-run

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/baselines_chinesehp_test.json \
  --dry-run

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/qwen_lora_5090.json \
  --dry-run

PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/evaluate_qwen_lora_test.json \
  --dry-run
```

dry-run 不会读模型，也不会训练，只会打印将要执行的命令。

## 5. 准备全量数据

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/prepare_chinesehp_full.json
```

成功后应该生成：

```text
data/raw/chinesehp_aishell1.jsonl
data/processed/chinesehp_aishell1_sft.jsonl
data/processed/chinesehp_aishell1_sft.clean.jsonl
data/processed/chinesehp_aishell1/train.jsonl
data/processed/chinesehp_aishell1/dev.jsonl
data/processed/chinesehp_aishell1/test.jsonl
data/processed/chinesehp_aishell1/train.qwen_messages.jsonl
data/processed/chinesehp_aishell1/dev.qwen_messages.jsonl
data/processed/chinesehp_aishell1/test.qwen_messages.jsonl
outputs/logs/prepare_chinesehp_full.json
```

预期 split 数量大致来自 ChineseHP：

```text
train: 120098
dev: 14326
test: 7176
```

过滤后数量可能略有变化，以脚本输出为准。

## 6. 跑测试集基线

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/baselines_chinesehp_test.json
```

这个会评测：

- no correction：直接使用 ASR top1。
- oracle N-best：在 N-best 中用 reference 选 CER 最低的候选，只作为上界。
- rule baseline：保守规则纠错。

输出位置：

```text
outputs/baselines/chinesehp_test_oracle_predictions.jsonl
outputs/baselines/chinesehp_test_rule_corrected.jsonl
outputs/logs/baselines_chinesehp_test.json
```

注意：oracle N-best 不是可部署模型，只用来判断 N-best 内部还有多少可挖空间。

## 7. 训练 Qwen LoRA

默认配置：

```text
configs/qwen_lora_5090.json
```

默认模型：

```text
Qwen/Qwen3-4B
```

正式训练：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/qwen_lora_5090.json
```

如果有多张 GPU，只想用某一张：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/qwen_lora_5090.json
```

训练完成后必须检查这些文件：

```bash
ls outputs/qwen_asr_corrector_lora
cat outputs/qwen_asr_corrector_lora/training_metadata.json
```

这里保存的是 LoRA adapter 和 tokenizer 文件，不是完整 base model。推理时仍会加载 base model `Qwen/Qwen3-4B`，再加载：

```text
outputs/qwen_asr_corrector_lora
```

关键保存文件包括：

```text
outputs/qwen_asr_corrector_lora/adapter_config.json
outputs/qwen_asr_corrector_lora/adapter_model.safetensors
outputs/qwen_asr_corrector_lora/training_metadata.json
outputs/logs/qwen_lora_5090.json
```

## 8. 测试集推理与评测

训练结束后跑：

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --config configs/evaluate_qwen_lora_test.json
```

它会做：

```text
test internal records
  -> infer_lora_edits.py
  -> apply_edits_jsonl.py
  -> evaluate_correction_jsonl.py
  -> analyze_corrections.py
```

输出位置：

```text
outputs/qwen_asr_corrector_lora/test_predicted_edits.jsonl
outputs/qwen_asr_corrector_lora/test_applied.jsonl
outputs/qwen_asr_corrector_lora/analysis_test/
outputs/logs/evaluate_qwen_lora_test.json
```

重点看评测里的：

- `baseline_cer`
- `cer`
- `improved_samples`
- `worsened_samples`
- `unchanged_samples`

如果 `worsened_samples` 明显偏高，优先检查模型是否在自由改写整句、JSON 是否解析失败、verifier 是否拒绝太多。

## 9. 快速 sanity check

代码检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src scripts tests
```

训练数据格式检查：

```bash
PYTHONPATH=src python scripts/train_lora_sft.py \
  --train-file data/processed/chinesehp_aishell1/train.qwen_messages.jsonl \
  --eval-file data/processed/chinesehp_aishell1/dev.qwen_messages.jsonl \
  --output-dir outputs/dry_train \
  --model-name-or-path dummy \
  --input-format qwen-messages \
  --dry-run
```

## 10. 常见问题

### 找不到 ChineseHP 文件

检查 `configs/prepare_chinesehp_full.json` 的 `input` 路径。服务器路径经常和本地不同。

### bitsandbytes/CUDA 报错

先不要改训练脚本。检查：

```bash
python -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

如果 4bit 加载失败，可以临时去掉配置里的：

```json
"qlora": true
```

但这会显著增加显存占用。

### 显存不够

优先调小：

```json
"max_length": 1536
```

或保持：

```json
"per_device_train_batch_size": 1
```

并增大/维持：

```json
"gradient_accumulation_steps": 16
```

### 训练中断

`Trainer` 会在 `outputs/qwen_asr_corrector_lora/checkpoint-*` 保存阶段 checkpoint。当前 pipeline 没有自动 resume 参数；如果要恢复，需要手动扩展 `scripts/train_lora_sft.py` 或直接用 Trainer 的 checkpoint 继续。

### 推理找不到 adapter

确认训练输出目录存在：

```bash
ls outputs/qwen_asr_corrector_lora
```

并确认 `configs/evaluate_qwen_lora_test.json` 里的：

```json
"adapter_path": "outputs/qwen_asr_corrector_lora"
```

和实际路径一致。

## 11. 跑完后要回传的信息

把这些结果告诉本地/论文整理阶段：

```text
git commit hash:
GPU:
base model:
train/dev/test record counts:
no-correction CER:
oracle N-best CER:
rule baseline CER:
LoRA test CER:
improved/worsened/unchanged:
training output dir:
main logs:
```

主要日志文件：

```text
outputs/logs/prepare_chinesehp_full.json
outputs/logs/baselines_chinesehp_test.json
outputs/logs/qwen_lora_5090.json
outputs/logs/evaluate_qwen_lora_test.json
```
