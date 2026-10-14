# COVO

本目录包含 CB-SenseVoice 后纠错所需的 COVO 数据处理、LoRA/DPO 训练、
推理和评测代码。COVO 接收完整 ASR top-1、N-best、拼音与上下文热词证据，
输出纠错后的完整句子。

## 安装

```bash
pip install -e covo
pip install -r covo/requirements-train.txt
```

基础模型默认放在仓库根目录的 `models/Qwen3.5-4B`，训练数据和输出分别放在
`covo/data/processed/` 与 `covo/outputs/`；这些大型产物不会提交到 Git。

## 常用入口

```bash
PYTHONPATH=covo/src python covo/scripts/train_lora_sft.py --help
PYTHONPATH=covo/src python covo/scripts/train_lora_dpo_text.py --help
PYTHONPATH=covo/src python covo/scripts/infer_lora_text.py --help
PYTHONPATH=covo/src python covo/scripts/evaluate_correction_jsonl.py --help
```

CB-SenseVoice evidence 先由主仓库脚本转成 Qwen messages：

```bash
python src/analysis/cbsensevoice_covo_bridge.py prepare \
  --input src/logs/cb_sensevoice_evidence.jsonl \
  --output covo/data/processed/cb_sensevoice_messages.jsonl \
  --include-pinyin \
  --protect-supported-hotwords
```

训练目标必须是完整纠错句子：

```json
{"text":"纠错后的完整句子"}
```

## 检查

```bash
PYTHONPATH=covo/src python -m unittest discover -s covo/tests
PYTHONPATH=covo/src python -m compileall covo/src covo/scripts covo/tests
```
