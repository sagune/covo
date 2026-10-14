# CBSense + CEGER

上下文热词语音识别与候选证据引导的生成式纠错。CEGER 是原 CoVo 大模型模块的新名称，代码包名仍保留 `covo`。

本次更新保存截至 **2026-10-14** 的服务器源码、实验协议、已有结果报告与论文流程图。原仓库早期文本纠错工具保留；**当前完整系统以 [`server_snapshot/`](server_snapshot/) 为准**，不要把根目录历史配置当成当前 9B 实验配置。

![CBSense + CEGER 全流程](docs/figures/cbsense-ceger-full-flow.png)

## 当前系统

1. **离线热词库**：候选热词经 TTS 合成、冻结 SenseVoice 编码、有效帧提取与归一化，缓存帧级声学表示。
2. **CBSense**：输入语音经 SenseVoice 提取特征，与热词特征库匹配，TCResNet 检出相关热词；热词编码与音频特征共同参与上下文识别，输出 Top-1 与 N-best。
3. **CEGER**：整理候选、拼音、稳定/分歧片段及适用的热词信息，以结构化文本输入 Qwen3.5-9B + LoRA，生成最终转写。

这条主链路只有一个文本 9B，不直接接收原始音频或连接器张量。连接器、声学分数 A/B 判决和复核实验是独立研究分支，不是上图默认流程。

## 代码位置

| 内容 | 入口 |
|---|---|
| CBSense 模型与上下文适配 | [`server_snapshot/src/model/`](server_snapshot/src/model/) |
| TCResNet/KWS | [`server_snapshot/src/efficient_kws/`](server_snapshot/src/efficient_kws/) |
| 前端入口 | [`server_snapshot/src/run_CLI.py`](server_snapshot/src/run_CLI.py) |
| 文本证据桥接 | [`server_snapshot/src/analysis/cbsensevoice_covo_bridge.py`](server_snapshot/src/analysis/cbsensevoice_covo_bridge.py) |
| CoVo/CEGER 包、LoRA 训练与推理 | [`server_snapshot/covo/`](server_snapshot/covo/) |
| 迁移实验实际使用的运行时代码 | [`server_snapshot/cbwhisper_covo_migration_20260609_tar_extracted/covo/`](server_snapshot/cbwhisper_covo_migration_20260609_tar_extracted/covo/) |
| 连接器、发音匹配、声学复核研究 | [`server_snapshot/src/experiments/`](server_snapshot/src/experiments/) |
| 源文件哈希清单 | [`server_snapshot/SOURCE_MANIFEST.json`](server_snapshot/SOURCE_MANIFEST.json) |

## 训练与评估

第一轮 9B 基础 LoRA 沿用早期 4B 的候选证据构造思路：N-best、拼音、候选分歧、困难候选增强和证据随机丢弃。9B 独立加载自己的基座，不继承 4B 权重。历史第一轮采用整段对话因果语言建模监督，只更新 LoRA；不能把后续 Assistant-only loss 倒写成第一轮实现。

CBSense 输入适配、联合连接器训练、声学证据判决的初始化和目标不同，详见各实验锁定协议与报告。历史 **2.777% 是 4B 结果，不是当前 9B 的统一基线**。

- [源码快照使用说明](docs/current_snapshot.md)
- [历次实验结果索引](docs/results/README.md)
- [流程图与样例来源](docs/figures/README.md)
- [原仓库说明（保留）](docs/legacy/README-before-20261014.md)

中文主要报告 CER，英文报告 WER。历史 AISHELL contextual dev 的 **1,334 句 / 21,102 字 / 599 热词 mentions** 不是官方完整 dev，也不是新测试集。Oracle 是参考已知的候选上限，不是模型实测能力。

## 安装与运行边界

```bash
git clone https://github.com/sagune/covo.git
cd covo/server_snapshot
# 在独立、适配 GPU/CUDA 的环境安装。
pip install -r requirements.txt
pip install -e covo
pip install -r covo/requirements-train.txt
PYTHONPATH=covo/src python covo/scripts/train_lora_sft.py --help
PYTHONPATH=covo/src python covo/scripts/infer_lora_text.py --help
```

这是源码留痕快照，不是零配置部署产品。锁定实验保留原服务器路径及哈希，没有为发布改写实验。重现需另备数据、基座、adapter、热词库、匹配环境及路径；宽松的 `requirements.txt` 不能代替特定实验的后端版本锁定。

**不包含**：服务器密码与 SSH 配置、私钥/API token、模型权重、语音数据集、逐条预测、训练样本、运行环境、长日志和第三方论文 PDF。推送代码不会启动或改变服务器实验。

`server_snapshot/LICENSE.md` 保留原前端 BSD-3-Clause 许可和版权声明。第三方模型、库与数据仍受各自许可约束，本仓库不为其授予新许可。
