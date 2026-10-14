<div align="center">

# CB-SenseVoice + COVO

**轻量、可解释的中文上下文偏置 ASR 与大模型后纠错系统**

SenseVoice 声学建模 · TCResNet 热词检索 · Phrase Cross-Attention · COVO 保守纠错

[实验结果](RESULTS.md) · [实验上下文](EXPERIMENT_CONTEXT.md) · [AISHELL 账本](AISHELL_EXPERIMENTS.md) · [COVO 文档](covo/README.md)

</div>

---

## 项目简介

本项目面向中文热词与命名实体识别，将上下文偏置能力完整迁移到
SenseVoice，并使用 COVO 对 ASR 候选、拼音和热词证据进行保守后纠错。

当前仓库只维护 **CB-SenseVoice + COVO** 路线。早期 CB-Whisper 路线已经
停止维护，不再提供入口和配置。

### 核心结果

| 测试集 | 系统 | CER | 热词/实体召回 |
|---|---|---:|---:|
| AISHELL-NE 808 | CB-SenseVoice w14 | 3.7951% | 91.50% |
| AISHELL-NE 808 | **w14 + DPO-30 COVO** | **3.1354%** | **90.50%** |
| AISHELL-1 全量 7,176 条 | **Routed + DPO-30 COVO** | **3.2062%** | 91.453% 实体召回 |
| THCHS-30 全量 test | SenseVoice 10-best + COVO | **4.217%** | - |
| MAGICDATA-READ 全量 test | **SenseVoice 10-best + domain COVO** | **3.4673%** | Oracle 1.3029% |
| WeNetSpeech TEST_NET 全量 | SenseVoice 10-best + COVO | **7.473%** | - |
| WeNetSpeech TEST_MEETING 全量 | SenseVoice 10-best top1 | **7.461%** | - |

完整口径、消融、跨数据集结果和论文对比见 [RESULTS.md](RESULTS.md)。

## 系统结构

```text
                                 Context vocabulary
                                         │
                                         ▼
Audio ──► SenseVoice encoder ──► TCResNet KWS retrieval
  │                                      │
  │                                      ▼
  └──────────────────────► Phrase cross-attention adapter
                                         │
                                         ▼
                            Context-biased CTC prefix beam
                                         │
                                         ▼
                              CB-SenseVoice N-best
                                         │
                              pinyin / KWS / scores
                                         │
                                         ▼
                              COVO conservative correction
                                         │
                                         ▼
                                  Final transcript
```

### 设计原则

- **轻量适配**：冻结 SenseVoice、KWS 主体和 Qwen 基座，只训练约 729k 参数
  的上下文适配器或 LoRA。
- **短语级声学证据**：热词以完整短语编码，通过 frame-to-phrase
  cross-attention 与声学帧对齐。
- **候选证据驱动**：COVO 同时读取 top1、N-best、拼音、候选分数和 KWS
  热词，不依赖单一文本提示。
- **论文协议优先**：主方法不使用置信度 gate、参考答案感知选择、测试词典
  回退或测试后规则补丁。

## 环境

项目实验环境位于 `/root/autodl-tmp/great`，不要使用 base 环境。

```bash
conda activate /root/autodl-tmp/great
cd /root/autodl-tmp
```

安装仓库依赖与 COVO：

```bash
pip install -r requirements.txt
pip install -e covo
pip install -r covo/requirements-train.txt
```

大型模型、数据集和 checkpoint 不包含在 Git 仓库中。

## 数据准备

### 1. 构建 SenseVoice KWS 数据

```bash
python src/analysis/prepare_sensevoice_kws_dataset.py \
  --source datasets/aishell/data_aishell \
  --target datasets/aishell/data_aishell_sensevoice
```

### 2. 提取 SenseVoice 隐藏层

```bash
bash src/scripts/run_sensevoice_kws_extraction_20260714.sh
```

### 3. 训练 KWS

```bash
cd src
/root/autodl-tmp/great/bin/python run_CLI.py fit \
  --config configs/train-sensevoice-kws.yaml
```

## 运行主流程

### CB-SenseVoice 评测

原始 AISHELL-NE 808 条基准：

```bash
cd /root/autodl-tmp/src
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

全量测试入口：

```bash
cd /root/autodl-tmp/src

# AISHELL-1 全量 7,176 条
./scripts/run_cb_sensevoice_full_test.sh aishell

# ST-CMDS 固定 held-out 5,130 条
./scripts/run_cb_sensevoice_full_test.sh stcmds
```

| 全量入口 | 语句 | 热词 | 正样本 | 热词来源 |
|---|---:|---:|---:|---|
| AISHELL-1 | 7,176 | 1,291 | 2,239 | 原 400 词 + AISHELL-NER test 实体 |
| ST-CMDS held-out | 5,130 | 3,139 | 1,279 | HanLP 从 ST-CMDS test 参考文本抽取 |

这两个入口用于测试 CB-SenseVoice 覆盖全量语句的能力，属于由测试参考文本构造
词表的 `oracle-context` 协议。它们不能与开放词表 ASR 结果混写。运行脚本启用
每组 top-32 声学预筛选，以限制大词表 KWS 的相似度矩阵开销。大词表评测不再
为每个词表分组强制补入一个低于阈值的热词，避免误检数量随分组数线性增长。

错误定向的高浓度诊断入口：

```bash
cd /root/autodl-tmp/src
./scripts/run_cb_sensevoice_full_test.sh aishell-targeted
./scripts/run_cb_sensevoice_full_test.sh stcmds-targeted
```

| 数据 | 词表 | 热词正样本 | 字符覆盖 | 裸 ASR 错误覆盖 |
|---|---:|---:|---:|---:|
| AISHELL-1 targeted | 3,499 | 52.26% | 16.65% | 85.50% |
| ST-CMDS targeted | 5,241 | 51.73% | 15.39% | 92.49% |

该协议根据测试参考文本和裸 SenseVoice 错误构造目标热词，存在明确的数据泄漏，
只能作为热词利用能力的 oracle 上界诊断，不能作为论文主结果。AISHELL 前 71 条
pilot 中，裸 SenseVoice CER 为 3.30%，大词表 KWS 使用原阈值时为 3.20%，关闭
分组强制补词并将阈值提高到 0.95 后为 2.68%，gold 热词直接注入为 0.21%。
高置信 KWS 相对裸模型降低约 18.8%，说明正确热词能显著影响 CER；gold 与 KWS
之间的差距表明当前主要瓶颈仍是大词表检索质量。ST-CMDS 前 51 条 pilot 也从
6.31% 降到 5.63%，绝对下降 0.68 个百分点。

全量泄漏诊断中，AISHELL KWS/Gold 句均 CER 分别为 `3.4067%/1.4628%`，
ST-CMDS 分别为 `4.5447%/1.8306%`。按 COVO 的 corpus-CER 口径，AISHELL
从 `2.6345%` 降到 `1.8336%`，ST-CMDS 则从 `4.4834%` 变差到 `4.6917%`。
两个 CER 统计口径不可直接混算，完整说明见 [RESULTS.md](RESULTS.md)。

### 导出 COVO evidence

```bash
cd /root/autodl-tmp/src
CBW_EVIDENCE_ONLY=1 \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_evidence.jsonl \
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

### 转换为 COVO 输入

```bash
cd /root/autodl-tmp/src
/root/autodl-tmp/great/bin/python analysis/cbsensevoice_covo_bridge.py prepare \
  --input logs/cb_sensevoice_evidence.jsonl \
  --output ../covo/data/processed/cb_sensevoice_messages.jsonl \
  --max-nbest 6 \
  --max-pinyin 3 \
  --max-hotwords 8 \
  --include-pinyin \
  --protect-supported-hotwords
```

### COVO 推理

```bash
cd /root/autodl-tmp
PYTHONPATH=covo/src /root/autodl-tmp/great/bin/python \
  covo/scripts/infer_lora_text.py \
  --input covo/data/processed/cb_sensevoice_messages.jsonl \
  --output covo/outputs/cb_sensevoice_predictions.jsonl \
  --model-name-or-path models/Qwen3.5-4B \
  --adapter-path covo/outputs/qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16
```

Adapter 名称中的 `cbwhisper` 是历史 checkpoint 标识，不代表当前 ASR
后端。当前 evidence 和候选均由 CB-SenseVoice 生成。

## 主要模型

| 组件 | 模型或路径 |
|---|---|
| ASR | `iic/SenseVoiceSmall` |
| KWS | `src/outputs/aishell_sensevoice_kws/checkpoints/f1G/f1G-epoch=16-step=102085.ckpt` |
| Context adapter | `src/outputs/sensevoice_context_adapter/phrase_crossattn_aishell_full_20260722.pt` |
| COVO base | `models/Qwen3.5-4B` |
| CER adapter | `covo/outputs/qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7` |
| Joint adapter | `covo/outputs/qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16` |

默认 phrase-confidence weight 为 `14.0`。

## 仓库结构

```text
.
├── README.md
├── RESULTS.md                     # 统一实验结果与论文比较
├── EXPERIMENT_CONTEXT.md          # 当前路线、约束与复现上下文
├── AISHELL_EXPERIMENTS.md         # AISHELL 逐轮实验账本
├── covo/
│   ├── src/covo/                  # COVO 核心库
│   ├── scripts/                   # SFT/DPO、推理与评测
│   ├── configs/
│   └── tests/
└── src/
    ├── model/cb_sensevoice.py     # CB-SenseVoice 主模型
    ├── model/sensevoice_ctc.py    # 上下文 CTC beam
    ├── model/sensevoice_context_adapter.py
    ├── analysis/                  # 数据、evidence 与评测工具
    ├── configs/                   # KWS/CB-SenseVoice 配置
    └── tests/
```

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_sensevoice*.py'
PYTHONPATH=src python -m unittest discover -s src/tests -p 'test_cbsensevoice_covo_bridge.py'
PYTHONPATH=covo/src python -m unittest discover -s covo/tests
python -m compileall src covo/src covo/scripts covo/tests
```

## 实验纪律

1. 每次代码修改后提交 Git。
2. 每轮实验记录数据范围、评价口径、checkpoint 和相对变化。
3. 明确区分 full、pilot、smoke、oracle 和数据泄漏诊断。
4. 不覆盖当前最好 checkpoint，不把数据集特例包装成论文贡献。
5. 数据、模型权重、evidence、预测 JSONL 和运行日志不提交 GitHub。

## 文档

| 文档 | 内容 |
|---|---|
| [RESULTS.md](RESULTS.md) | 当前所有可比较实验结果、消融与论文对比 |
| [EXPERIMENT_CONTEXT.md](EXPERIMENT_CONTEXT.md) | 当前研究目标、模型路径与复现流程 |
| [AISHELL_EXPERIMENTS.md](AISHELL_EXPERIMENTS.md) | AISHELL 历史实验账本 |
| [covo/README.md](covo/README.md) | COVO 训练、推理和数据格式 |

## 致谢与许可

方法思想参考
[CB-Whisper](https://aclanthology.org/2024.lrec-main.262/)，ASR 使用
[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)，后纠错模块基于
COVO 与 Qwen 系列模型开发。

使用数据和模型时请遵守各上游项目与数据集许可。本仓库许可见
[LICENSE.md](LICENSE.md)。
