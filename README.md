<h1 align="center">CBSense · CEGER</h1>

<p align="center">
  <strong>Contextual ASR + Candidate-Evidence-guided Generative Error Correction</strong><br />
  <sub>面向热词与专名的上下文语音识别及候选证据引导纠错</sub><br />
  <sub>Context-aware speech recognition and evidence-guided transcript correction</sub>
</p>

<p align="center">
  <a href="https://github.com/sagune/covo/actions/workflows/test.yml"><img src="https://github.com/sagune/covo/actions/workflows/test.yml/badge.svg" alt="Source checks and lightweight tests" /></a>
  <img src="https://img.shields.io/badge/Status-Research_Prototype-0891B2" alt="Research prototype" />
  <img src="https://img.shields.io/badge/ASR-SenseVoice-2563EB" alt="SenseVoice ASR" />
  <img src="https://img.shields.io/badge/LLM-Qwen3.5--9B_%2B_LoRA-7C3AED" alt="Qwen3.5-9B with LoRA" />
</p>

<p align="center">
  <a href="#中文">中文</a> &nbsp;|&nbsp;
  <a href="#english">English</a> &nbsp;|&nbsp;
  <a href="#快速开始">快速开始</a> &nbsp;|&nbsp;
  <a href="#实验结果">实验结果</a> &nbsp;|&nbsp;
  <a href="#quick-start">Quick Start</a>
</p>

> [!IMPORTANT]
> 本仓库保留的最终主线是 **CBSense / SenseVoice 识别候选 → 文本证据整理 → 9B CEGER 纠错**，对应 AISHELL、ST-CMDS、MAGICDATA-READ、THCHS-30 与 LibriSpeech 的已有实验。各数据集使用匹配的 adapter，并非同一 checkpoint 通用于五个数据集。旧连接器和后续声学复核尝试已归档。
>
> The retained mainline is the multi-dataset ASR + 9B correction family, with dataset-specific adapters. Superseded neural connectors and acoustic-review trials are archived, not presented as the final pipeline.

<p align="center">
  <img src="docs/figures/cbsense-ceger-full-flow.png" alt="Offline hotword feature bank and online CBSense to CEGER pipeline" width="100%" />
</p>

---

# 中文

## 项目简介

CBSense · CEGER 面向中文语音中的人名、地名、专业术语及其他上下文热词，将热词识别与大模型转写纠错组织成一条可追踪的处理流程。

- **CBSense**：基于 SenseVoice 的上下文偏置识别模块，通过热词声学特征匹配和 TCResNet 检出相关热词，将上下文信息用于语音识别。
- **CEGER**：候选证据引导的生成式纠错模块，使用 Qwen3.5-9B + LoRA，联合读取识别候选、拼音与候选分歧等文本证据，生成最终转写。

CEGER 是原 CoVo 大模型模块的新名称；仓库名和 Python 包名仍为 `covo`。当前发布包含源码、锁定实验协议、历史报告及论文流程图。

### 我们研究什么问题

| 问题 | 系统中的设计 |
|---|---|
| 低频专名和热词容易识别成常见词 | 利用 TTS 构建热词声学特征库，并在输入语音中检索相关热词 |
| 大量候选热词会带来上下文干扰 | 先进行 KWS 检出与筛选，再将选出的热词用于上下文识别 |
| 单一 Top-1 隐藏了识别过程中的其他可能 | 将 N-best、拼音和候选分歧整理为结构化文本证据 |
| 纠错模型需要适应 ASR 的错误分布 | 使用候选证据增强与 LoRA 监督训练进行任务适配 |
| 总体 CER 可能掩盖专名丢失和过度改写 | 分别记录热词召回、正确输入误改及不同错误类型 |

这些是设计目标和评估问题，不代表每个实验都已经实现无代价改善。

## 核心特色

<table>
<tr>
<td width="50%" valign="top">
<h3>热词声学检索</h3>
<p>通过合成语音与冻结的 SenseVoice 编码器构建帧级热词表示。在线语音与特征库匹配，TCResNet 判断候选热词是否出现。</p>
</td>
<td width="50%" valign="top">
<h3>上下文偏置识别</h3>
<p>将筛选后的热词编码与音频特征一起用于上下文识别，产生 Top-1 和 N-best，供后续纠错使用。</p>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<h3>多候选证据引导</h3>
<p>保留候选之间的一致与分歧，结合拼音线索及适用的热词信息，组织成可读取的纠错输入。</p>
</td>
<td width="50%" valign="top">
<h3>LoRA 任务适配</h3>
<p>冻结大模型基座，训练低秩适配参数。基础阶段引入困难候选增强和证据随机丢弃，以适应识别错误与不完整候选。</p>
</td>
</tr>
</table>

## 系统架构

| 阶段 | 主要处理 | 产物 |
|---|---|---|
| 离线准备 | 热词 → TTS → SenseVoice → 有效帧提取与归一化 | 热词声学特征库 |
| 在线检索 | 输入音频特征与热词库匹配 → TCResNet 检出与筛选 | 检出热词及其分数 |
| 上下文识别 | 音频特征 + 热词编码 | Top-1 / N-best |
| 证据整理 | 候选、拼音、稳定/分歧片段、适用的热词信息 | 结构化文本输入 |
| 生成式纠错 | Qwen3.5-9B + 已训练 LoRA | 最终转写 JSON |

CEGER 主线读取识别候选等文本证据，不直接接收波形或神经连接器张量。旧 MLP、Residual、局部匹配及后来基于 SenseVoice 分数的 A/B 复核均属于[归档尝试](docs/archive/README.md)，不再作为默认方案展示。

<details>
<summary>展开 CBSense 模块图</summary>

![CBSense 热词检索与上下文识别](docs/figures/cbsense.png)

</details>

### 一个真实候选示例

| 位置 | 内容 |
|---|---|
| CBSense Top-1 | 金华时报记者王海欣摄 |
| 候选 2 | 京华时报记者王海欣摄 |
| 候选 3 | 金华时报记者王海欣说 |
| 检出热词 | 王海欣 |
| 历史 9B 输出 | 京华时报记者王海欣摄 |

这是已有验证结果中的说明样本，不表示用于训练，也不代表整体性能。[样本来源与图示边界](docs/figures/SAMPLE_PROVENANCE.md)。

## 训练模式

第一轮基础 LoRA 沿用早期 4B 的数据构造思路，为 9B 独立加载基座并训练，不转移 4B 模型权重。

| 项目 | 第一轮 9B 基础训练 |
|---|---|
| 输入 | ASR 候选、拼音、稳定片段与候选分歧 |
| 数据增强 | 困难候选增强；随机丢弃部分候选与辅助证据 |
| 监督 | 参考转写进入训练目标，不作为推理输入 |
| 历史损失实现 | 整段训练对话的因果语言建模监督 |
| 参数更新 | 冻结 Qwen3.5-9B 基座，只更新 LoRA |

多数据集实验使用各自匹配的模型与 adapter，THCHS-30 保留无训练迁移的全部 9B 对照。旧声学研究分支已归档，不作为正式主线的训练阶段；不能把探索模型的初始化、损失或开发集指标混入基础模型说明。

<details>
<summary>展开第一轮 LoRA 与候选证据图</summary>

![CEGER 第一轮 LoRA 训练](docs/figures/ceger-lora-training.png)

![候选证据与训练增强](docs/figures/ceger-evidence.png)

</details>

## 实验结果

当前结果页只展示已做过多数据集实验的基础 9B 纠错系列，不再混入后来的 AISHELL 开发集连接器或复核尝试。

| 数据集 / 范围 | Adapter / checkpoint | 指标 | 输入 → 输出 |
|---|---|---|---|
| AISHELL-1 test，7,174 条有效计分 | AISHELL checkpoint-30022 | CER | 5.8354% → 2.6231% |
| ST-CMDS test，5,130 条 | ST-CMDS checkpoint-34400 | CER | 5.6407% → 4.4994% |
| MAGICDATA-READ test，24,279 条 | MAGICDATA checkpoint-39185 | CER | 3.7245% → 2.4334% |
| THCHS-30 test，2,495 条，无训练迁移 | AISHELL checkpoint-30022 | CER | 5.1664% → 4.3111% |
| THCHS-30 test，2,495 条，无训练迁移 | ST-CMDS checkpoint-34400 | CER | 5.1664% → 4.3358% |
| THCHS-30 test，2,495 条，无训练迁移 | MAGICDATA checkpoint-39185 | CER | 5.1664% → 4.4023% |
| LibriSpeech test-clean，2,620 条 | final 2-epoch adapter | WER | 3.2277% → 2.6419% |
| LibriSpeech test-other，2,939 条 | final 2-epoch adapter | WER | 7.2789% → 5.9206% |

来源：[主线结果与模型索引](docs/results/mainline-multidataset.md)、[完整历史总表](server_snapshot/RESULTS_ALL_DATASETS_20260826.md)。这里只转录原始结果，不进行新评估或重新选 checkpoint；THCHS-30 的三个跨域 adapter 全部保留，不按其 test 成绩挑一个作为预先选定模型。

上述行的前端是历史 SenseVoice 10-best（MAGICDATA 为 WOITN）；不能全部改称 CBSense 热词增强结果。CBSense/CB-SenseVoice 联合实验在完整总表中另列；测试参考构造热词的 Error-targeted、OracleLex 以及候选 Oracle 仍明确标为诊断，不包装成正式 held-out 成绩。AISHELL 旧评估器跳过了两条空参考，因此有效计分数为 7,174，不隐瞒此边界。

旧连接器、热词续训、自由 JSON 复核、声学 A/B LoRA 与候选上限等尝试从当前展示中移出，原文与失败记录保存在[历史档案](docs/archive/README.md)及不可变源码快照中。历史 1,334 contextual dev 的结果不再充当这套多数据集模型的主成绩。

## 快速开始

### 环境与资源

- 轻量数据工具的包元数据要求 Python `3.9+`；下面的完整源码轻量检查推荐 Python `3.11`，与 CI 一致。GPU 实验需匹配各协议的 Python、PyTorch、CUDA 与模型后端版本。
- 浏览文档、语法检查和后端轻量测试不需要 GPU。
- 端到端识别与 9B 推理需自行准备 SenseVoice、TCResNet、上下文适配器、9B 基座、匹配的 LoRA 和热词特征库。
- 历史配置保留原机器的路径与数据设置，使用前请阅读[快照说明](docs/current_snapshot.md)。

### 1. 获取源码与运行轻量检查

```bash
git clone https://github.com/sagune/covo.git
cd covo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/check_source_snapshot.py
PYTHONPATH=src python -m unittest discover -s tests
```

### 2. 准备完整系统环境

在适配 CUDA 的独立环境中，从 `server_snapshot` 安装：

```bash
cd server_snapshot
python -m pip install -r requirements.txt
python -m pip install -e covo
python -m pip install -r covo/requirements-train.txt

PYTHONPATH=covo/src python covo/scripts/train_lora_sft.py --help
PYTHONPATH=covo/src python covo/scripts/infer_lora_text.py --help
```

前端模型位于 [`src/model/cb_sensevoice.py`](server_snapshot/src/model/cb_sensevoice.py)，CLI 与配置位于 [`src/run_CLI.py`](server_snapshot/src/run_CLI.py) 和 [`src/configs/`](server_snapshot/src/configs/)。文本证据转换入口为 [`cbsensevoice_covo_bridge.py`](server_snapshot/src/analysis/cbsensevoice_covo_bridge.py)。

> [!CAUTION]
> 部分历史配置是参考词表或错误定向的 Oracle 诊断，不能作为无泄漏主结果直接运行。先选择实验协议，再准备路径、热词来源与匹配检查点。依赖列表给出范围约束，不能替代特定实验的版本锁定。

### 3. 使用已准备好的文本证据推理

下面仅展示后端调用方式，不会从音频自动生成输入；路径需替换为自己的资源，提示格式需匹配 LoRA。

```bash
# 在 server_snapshot 中执行
python covo/scripts/infer_lora_text.py \
  --input /path/to/messages.jsonl \
  --output outputs/demo/predictions.jsonl \
  --model-name-or-path /path/to/Qwen3.5-9B \
  --adapter-path /path/to/matching-lora \
  --batch-size 1 \
  --disable-thinking
```

## 验证

```bash
# 从仓库根目录执行
python scripts/check_source_snapshot.py
PYTHONPATH=src python -m unittest discover -s tests

cd server_snapshot
PYTHONPATH=covo/src python -m unittest discover -s covo/tests

cd cbwhisper_covo_migration_20260609_tar_extracted/covo
PYTHONPATH=src python -m unittest discover -s tests
```

2026-10-14 发布检查：773 个服务器文件哈希一致；旧后端、快照后端、迁移运行时各 **19 项测试通过**；GitHub Actions 通过。三套测试存在重叠，不应视为 57 项独立功能。[发布检查记录](docs/PUBLICATION_AUDIT_20261014.md)。

轻量测试不代表已完成 GPU 数值检查、完整 CER 复算或端到端测速。ASR 正式验证仍需固定数据、归一化、检查点和计分口径，并保留失败样本。

## 目录与文档

```text
covo/
├── server_snapshot/                 # 当前完整系统源码留痕
│   ├── src/model/                   # CBSense、上下文适配与 CTC
│   ├── src/efficient_kws/           # TCResNet/KWS
│   ├── src/analysis/                # 证据桥接、数据工具与分析
│   ├── src/experiments/             # 锁定研究实验
│   ├── covo/                       # 主工程 CoVo/CEGER 后端
│   ├── cbwhisper_covo_migration_20260609_tar_extracted/covo/
│   │                               # 迁移实验对应后端源码
│   └── SOURCE_MANIFEST.json         # 原始文件大小与 SHA256
├── docs/figures/                    # 全流程、模块与训练图
├── docs/results/                    # 多数据集主线结果与模型索引
├── docs/archive/                    # 已停止展示的探索尝试，原报告保留
├── src/ · scripts/ · tests/         # 保留的早期 CoVo 工具
└── .github/workflows/               # 轻量自动检查
```

- [源码快照与重现说明](docs/current_snapshot.md)
- [主线结果与模型](docs/results/README.md)
- [旧尝试档案](docs/archive/README.md)
- [论文图与真实样本来源](docs/figures/README.md)
- [源码哈希清单](server_snapshot/SOURCE_MANIFEST.json)
- [原仓库 README](docs/legacy/README-before-20261014.md)

## 当前边界

本项目是科研原型与源码归档，不是零配置部署产品。模型权重、语音数据、训练输入、逐条预测、依赖 overlay 和服务器环境未随仓库公开。部分历史实验保留绝对路径，重现需要另行准备对应资源。

CER 改善、热词召回和正确文本保护需要同时评价。重复使用的开发集、参考构造的词表及 Oracle 结果均应明确标注；不能将其包装成独立测试集上的泛化能力。当前也没有规范的单条端到端 GPU 延迟结论。

---

# English

## Overview

CBSense · CEGER is a research pipeline for contextual speech recognition and transcript correction, with a focus on Mandarin names, specialized terminology and user-provided hotwords.

**CBSense** retrieves relevant hotwords using SenseVoice acoustic representations and TCResNet keyword spotting, then incorporates the selected context into recognition. **CEGER** uses a LoRA-adapted Qwen3.5-9B to correct transcripts from textual candidate evidence: Top-1, N-best, pinyin, candidate agreement/disagreement and applicable hotword information.

CEGER is the renamed CoVo module; the repository and package retain `covo`. The retained mainline is the previously evaluated multi-dataset 9B correction family. Neural-connector and later acoustic-review experiments are archived and are not the default pipeline.

## Design Highlights

- **Acoustic hotword retrieval:** an offline feature bank built from synthesized hotwords and frozen SenseVoice representations.
- **Context-aware recognition:** retrieved hotwords are encoded and incorporated into the ASR stage.
- **Candidate-evidence correction:** multiple hypotheses and pronunciation cues are organized into structured textual input.
- **Parameter-efficient adaptation:** a frozen LLM backbone with trainable LoRA parameters; foundational training uses candidate augmentation and evidence dropout.
- **Traceable experiments:** source hashes, fixed protocols and result reports distinguish measured predictions from reference-based Oracle bounds.

## Quick Start

For documentation and lightweight checks, no GPU is needed:

```bash
git clone https://github.com/sagune/covo.git
cd covo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_source_snapshot.py
PYTHONPATH=src python -m unittest discover -s tests
```

For the current full-system sources, work under `server_snapshot/`, install its requirements and `covo` package in an isolated GPU-compatible environment, and provide the matching models, adapters, data and hotword feature bank. See the [snapshot guide](docs/current_snapshot.md) and the [Chinese quick start](#快速开始) for entry points. Do not install multiple same-named `covo` packages into one environment without controlling import paths.

## Evaluation and Scope

The retained model family has recorded evaluations on AISHELL-1, ST-CMDS, MAGICDATA-READ, THCHS-30 and LibriSpeech, using matching adapters rather than one shared checkpoint. See the [mainline result and model index](docs/results/mainline-multidataset.md) for source records and protocol boundaries.

Chinese results use CER and LibriSpeech results use WER. THCHS-30 has no in-domain training; all three historical 9B transfer adapters remain reported. The main table uses SenseVoice 10-best inputs, not uniformly CBSense contextual inputs. Contextual, test-reference-derived vocabulary and Oracle rows remain separately classified in the full historical ledger.

Superseded neural connectors, continuation-training and acoustic-review trials are available in the [archive](docs/archive/README.md). Their repeated 1,334-utterance contextual dev results are not the multi-dataset model's test results. Failed trials, baselines, controls and locked source files are retained unchanged.

This release does not bundle weights, datasets, per-utterance predictions or runtime environments. No new evaluation, checkpoint selection or server cleanup was performed during this repository-only organization.

## Acknowledgements and License

The front-end snapshot retains the original [BSD-3-Clause notice](server_snapshot/LICENSE.md) for the Priberam contextual-biasing code. Models, dependencies and datasets remain governed by their respective licenses.

README layout takes inspiration from [智塔 · FAROS](https://github.com/OpenNSWM-Lab/FAROS). The system description, experiment records and figures belong to this project; no affiliation is implied.
