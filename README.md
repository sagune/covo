# CB-SenseVoice + COVO

本仓库研究轻量、可解释的中文上下文偏置语音识别。项目从
[CB-Whisper](https://aclanthology.org/2024.lrec-main.262/) 出发，将其
KWS、上下文候选生成和重排思想迁移到 SenseVoice，并使用 COVO 对
N-best、拼音和热词证据进行最终纠错。

当前主流程：

```text
音频
  -> SenseVoice 编码器隐藏层
  -> TCResNet KWS 检索候选热词
  -> 轻量 phrase cross-attention 上下文适配器
  -> 带声学短语证据的 SenseVoice CTC beam
  -> CB-SenseVoice 候选重排
  -> COVO 保守纠错
  -> 最终文本
```

当前方法不使用按样本置信度 gate、参考答案感知选择或测试后规则回退。
SenseVoice 主体、KWS 主体和 COVO 基座均保持冻结，只训练轻量上下文适配器
或 LoRA。

详细的逐轮实验记录保存在：

- [`EXPERIMENT_CONTEXT.md`](EXPERIMENT_CONTEXT.md)：完整实验流水、失败原因和模型路径。
- [`AISHELL_EXPERIMENTS.md`](AISHELL_EXPERIMENTS.md)：AISHELL 专项实验和论文比较。

## 当前主结果

### AISHELL-NE 808 条热词测试集

本地数据对应论文中的 `Test-Aishell1-NE`、`Test-Aishell1-Middle` 或
AISHELL hotword test：共 808 条语音、400 个指定热词，其中 226 个为
R1 难例热词。

下表统一采用 `OpenCC t2s + NFKC + 去除空格和标点` 后的 corpus CER。
Recall@400 和 R1 Recall@226 均按数据集指定热词及其对齐语句计算。

| 系统 | CER | Edits | Exact | Recall@400 | R1 Recall@226 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 裸 SenseVoiceSmall | 10.3609% | 1335 | 268 | 159/400 | 23/226 | 无上下文基线 |
| 裸 SenseVoice + preserve2 COVO | 7.7299% | - | - | 173/400 | 29/226 | COVO 单模块消融 |
| 原始 CB-SenseVoice | 7.8231% | 1008 | 432 | 313/400 | 141/226 | 无适配器 |
| + frame position adapter | 7.6911% | 991 | 436 | 318/400 | 146/226 | 第一代适配器 |
| + phrase cross-attention | 4.8273% | 622 | 498 | 323/400 | 150/226 | 结构改进 |
| **CB-SenseVoice w14** | **3.7951%** | **489** | 536 | **366/400** | **192/226** | 当前单独 CB 主结果 |
| w14 + preserve2 COVO | **3.0268%** | **390** | **587** | 357/400 | 184/226 | CER 最优消融 |
| **w14 + DPO-30 COVO** | **3.1354%** | **404** | 575 | **362/400** | **189/226** | **当前联合主结果** |

选择 DPO-30 作为联合主结果的原因是其 Recall@400 为 `90.50%`，达到项目
设定的 90% 热词召回目标；preserve2 的 CER 更低，但召回为 `89.25%`。

历史对照：

| 系统 | CER | 热词召回 | 说明 |
|---|---:|---:|---|
| 原论文 CB-Whisper | MER 8.6% | 82.4% | 论文报告口径 |
| 原始代码 + large-v3 + ResNet1 | 9.07% | 78.01% | 本地兼容性复现 |
| 本项目 true-v3 CB-Whisper | 6.61% | 92.38% | Whisper 路线历史最好结果 |
| 本项目 CB-Whisper + COVO | 4.284% | 91.083% | 迁移 SenseVoice 前的历史结果 |

这些历史行使用过不同的本地 evaluator，不能与统一 corpus CER 做逐字符的
严格等价比较，主要用于说明方法演进。

### 完整 AISHELL-1 test

完整测试集含 7,176 条语音。808 条具有外部上下文词表的语句使用
CB-SenseVoice w14，其余 6,368 条使用裸 SenseVoice。路由仅由是否存在
上下文词表决定，不查看模型置信度和参考答案。

| 系统 | Corpus CER | Edits | Exact | 说明 |
|---|---:|---:|---:|---|
| 裸 SenseVoiceSmall | 6.2922% | 6591 | 4454 | 全量、开放词表 |
| SenseVoice + 原始 COVO | 5.7053% | 5977 | 4769 | 全量、开放词表 |
| **SenseVoice + preserve2 COVO** | **3.7647%** | **3944** | **4982** | 全量、开放词表 |
| Routed SenseVoice / CB-SenseVoice w14 | 5.4894% | 5751 | 4722 | 808 条有上下文 |
| **Routed + DPO-30 COVO** | **3.2062%** | **3359** | **5235** | **联合主结果** |
| Routed + preserve2 COVO | **3.1929%** | **3345** | **5247** | CER 消融，808 召回低于 90% |

联合主结果在 AISHELL-NER 标注上的重新统计为：

| CER | NNE-CER | NE-CER | 实体召回 |
|---:|---:|---:|---:|
| **3.2129%** | **3.1343%** | **3.9546%** | **3103/3393 = 91.453%** |

另一个不使用 test 实体词表的检索消融，从 AISHELL-NER train/dev 建立实体库，
仅接受整段拼音完全匹配的同长度实体候选：

| 系统 | CER | NNE-CER | NE-CER | 实体召回 | 状态 |
|---|---:|---:|---:|---:|---|
| SenseVoice + COVO | 3.7732% | 3.2958% | 8.2777% | 82.85% | 无实体检索 |
| + exact-pinyin entity retrieval | **3.5384%** | **3.2927%** | **5.8572%** | **88.15%** | train/dev 实体库 |

`Routed` 结果属于上下文 ASR 协议，不应描述为完全开放词表的裸 AISHELL-1
结果。

## 模块实验

### SenseVoice KWS

KWS 使用 SenseVoice 编码器隐藏层构造热词/语音相似度矩阵，再由单通道
TCResNet 分类。SenseVoice 隐藏层维度为 512。

| 数据/模型 | 阈值 | Precision | Recall | F1 | 状态 |
|---|---:|---:|---:|---:|---|
| AISHELL large-v2 KWS validation | learned | 91.33% | 82.18% | 86.51% | Whisper 历史模型 |
| AISHELL true large-v3 KWS validation | 0.974 | 93.34% | 82.51% | 87.59% | Whisper-v3 KWS |
| AISHELL true large-v3 KWS test | 0.974 | 91.78% | 80.57% | 85.81% | 独立测试 |
| 原始 ResNet1 natural KWS test | 0.500 | 80.23% | 87.47% | 83.70% | 原论文代码对照 |
| **SenseVoice KWS test, TTS** | **0.787** | **90.948%** | **90.658%** | **90.803%** | 当前 SenseVoice KWS |
| SenseVoice KWS test, validation 阈值 | 0.954 | 95.721% | 83.121% | - | 高精度工作点 |

Whisper KWS 的 top-k 诊断表明真正热词进入候选池的概率已经很高：
Recall@1 `80.36%`、Recall@3 `95.22%`、Recall@6 `96.92%`、
Recall@12 `98.62%`。因此当前瓶颈不是 KWS 检索，而是 ASR 是否能把热词
实现为完整且正确的候选。

### 上下文适配器

当前接受的适配器使用完整短语表示，而不是把热词拆成独立字符：

- 冻结 SenseVoice 和 KWS。
- 以 CTC 字符向量、短语内位置和带声调拼音构造短语表示。
- 使用两层轻量 Transformer 编码热词。
- 通过 frame-to-phrase cross-attention 将声学帧与热词短语对齐。
- 训练目标包括 transcript CTC、短语位置和正负短语检测。
- 可训练参数约 729k。

训练使用完整 17,301 条 AISHELL train 热词语句，共 2,162 optimizer steps。

最终候选生成将适配器的 phrase-presence probability 注入 CTC prefix beam：

```text
effective_hotword_score = KWS_score * (1 + weight * phrase_probability)
```

权重扫描：

| Weight | Mention recall | N-best Recall@400 | Top1 Recall@400 | Unified CER |
|---:|---:|---:|---:|---:|
| 1 | 85.52% | 330 | 327 | 4.6566% |
| 3 | 87.73% | 345 | 342 | 4.4005% |
| 5 | 88.73% | 349 | 346 | 4.2685% |
| 7 | 89.94% | 354 | 350 | 4.1055% |
| 8 | 90.39% | 356 | 353 | 4.0667% |
| 10 | 91.27% | 361 | 358 | 3.9426% |
| 11 | 91.60% | 364 | 361 | 3.9115% |
| 12 | 92.04% | 367 | 364 | 3.8339% |
| **14** | **92.49%** | **369** | **366** | **3.7951%** |
| 16 | 92.49% | 369 | 365 | 3.8184% |

权重 14 同时得到最低 CER 和最高 top1 召回，因此是当前默认配置。

### COVO

COVO 输入包含：

- CB-SenseVoice top1；
- 六条带来源和分数的 N-best；
- 三条拼音候选；
- KWS/prompt hotwords；
- 已被可靠候选支持的 protected hotwords。

COVO 的目标是做最小必要纠错、保护已经正确出现的热词，并拒绝没有候选
支持的误报热词。当前主模型从 preserve2/no-op LoRA 出发，再做 30-step
hotword-preservation DPO。

最近的真实 CB-SenseVoice evidence pilot 使用 AISHELL train 的 301 条语句，
不使用测试参考：

| COVO adapter | 训练数据 | CER | Exact | Recall@400 | 结论 |
|---|---:|---:|---:|---:|---|
| **原 DPO-30** | 历史 hotword pairs | **3.1354%** | 575 | **362/400** | 保留 |
| Targeted DPO-20 | 117 correction + 60 hotword | 3.1354% | **584** | 355/400 | 召回下降 |
| Hotword-only DPO-10 | 120 hotword preferences | 3.1354% | 573 | **362/400** | 无收益 |

结论：当前 COVO 偏好学习基本饱和。继续微调只能重新分配错误，不能降低总
edit；下一步必须先让候选池产生更多正确形式。

## 跨数据集结果

所有结果均明确区分全量、固定划分和 pilot。不同数据集的标点、数字和语气词
归一化方式不同，不应把绝对 CER 当作完全相同的评价协议。

### THCHS-30

| 测试范围 | Samples | SenseVoice CER | + COVO CER | Improved | Worsened |
|---|---:|---:|---:|---:|---:|
| 国内镜像 pilot | 1,339 | 6.698% | **4.733%** | 331 | 14 |
| **完整 test** | **2,495** | **7.959%** | **5.601%** | **676** | **17** |

COVO 使用 AISHELL 训练的 preserve2 adapter，没有使用 THCHS-30 热词或测试
标注进行训练。

### ST-CMDS

ST-CMDS 原始包没有官方 train/dev/test 目录。本项目以 seed `20260719`
固定划分为 95,418 train、2,052 dev 和 5,130 held-out test。

| 实验 | Samples | 输入 CER | COVO CER | Improved | Worsened | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 随机设备分层 pilot | 2,000 | 6.101% | **5.788%** | 51 | 11 | 小实验 |
| ChineseHP-style held-out test | 5,130 | 5.641% | **5.167%** | 545 | 330 | 完整固定划分 |
| 未做 ST-CMDS 训练的 DPO60 | 5,130 | 5.641% | 6.969% | 498 | 1065 | 分布不匹配 |

ST-CMDS ChineseHP-style 模型在完整 95,418 条 train 上训练一轮，共
5,964 optimizer steps，dev loss `0.2392`。相对输入 CER 的绝对下降为
`0.474` 个百分点。

另有一个使用测试参考构造 3,139 条实体字典的 CB-SenseVoice 上下文实验：

| Samples | Entity recall | CER | Hotword CER | Candidate oracle CER | 协议 |
|---:|---:|---:|---:|---:|---|
| 10,260 | 87.9351% | 5.3331% | 7.1991% | 1.9355% | 测试字典诊断，不作为开放词表主结果 |

### 水利课程录音

原始水利课程数据集共 1,152 条语音，以下结果没有在该测试集上训练 COVO，
保留自然口语和语气词：

| 系统 | Number-normalized CER | Raw CER | Hotword recall | Edits | Exact |
|---|---:|---:|---:|---:|---:|
| 裸 SenseVoice | 4.636% | 4.654% | 82.884% | 731 | 722 |
| CB-SenseVoice | 4.059% | 4.076% | 93.050% | 640 | 763 |
| **AISHELL DPO60 listwise COVO** | **3.887%** | **3.905%** | **93.136%** | **613** | **774** |
| Candidate oracle | 1.820% | - | - | - | - |

### 新水利视频集

新水利视频集共 990 条。原始 180 词结果可作为独立实验；259 词和后续
confusion-term 结果使用了测试错误或测试标签，只能作为诊断上限。

| 系统 | CER | Filler+number CER | Hotword recall | 状态 |
|---|---:|---:|---:|---|
| 裸 SenseVoice | 4.873% | 4.902% | 82.567% | 独立基线 |
| CB-SenseVoice, 180 terms | 4.798% | - | 97.849% | 独立词表 |
| CB-SenseVoice, 259 terms | 4.072% | 4.074% | 95.157% | 测试错误扩词诊断 |
| AISHELL DPO60 listwise COVO | 3.736% | 3.722% | - | 使用泄露候选池 |
| Confusion-term DPO70 | 3.540% | 3.522% | - | 显式测试泄露 |
| Confusion-term DPO70, tuned w=0.3 | **3.503%** | **3.484%** | **95.642%** | 显式测试泄露上限 |
| Candidate oracle | 2.227% | 2.208% | - | 上限 |

最后三行不得作为论文独立测试结果。

### Speechio-Formal

该任务要求把口语转成正式书面语，与普通逐字 ASR 不同。现有 COVO 仅在
AISHELL 上训练，因此跨域提升很小。

| Domain | Samples | SenseVoice formal CER | COVO formal CER |
|---|---:|---:|---:|
| ZH00000 | 879 | 25.8634% | **25.8014%** |
| ZH00006 | 1,561 | 19.8511% | **19.7852%** |

## 失败路线与结论

| 路线 | 现象 | 决策 |
|---|---|---|
| 反复手调 exact/phonetic/consensus rerank | 候选池不变时收益接近零 | 停止规则堆叠 |
| 简单扩大 N-best 到 24 | 热词召回提高，但 CER/WER 恶化 | 需要提高候选质量而非数量 |
| Prompt recency/排列/单独注入 | 无稳定提升 | 不作为主方法 |
| 窄窗口 hotword CTC continuation | CER 和召回同时下降 | 回滚 |
| Wide-window hotword CTC | 仍弱于 phrase adapter | 回滚 |
| Lexicon synthetic confusable ranking | 错误负样本导致召回下降 | 只用真实声学混淆 |
| Monotonic hard activation | 召回略升但 CER 变差 | 仅保留消融 |
| Neutral/context dual branch | 召回保持但 rerank 选择弱候选 | 回滚 |
| Mixed COVO DPO | 模型过度保守，抑制正确修改 | 回滚 |
| Actual-evidence targeted COVO DPO | CER 不变、召回下降 | 不做全量扩展 |
| 水利 test-term / test-label training | 指标改善但发生数据泄露 | 只作为诊断上限 |

核心结论：

1. KWS top-k 已经不是主要上限。
2. phrase-level 声学证据显著改善 SenseVoice 对热词的候选生成。
3. COVO 可以降低一般字符错误，但会损失少量已正确热词。
4. 当前进一步提升取决于增加“正确热词且整句 ASR 质量良好”的候选，而不是
   加重 rerank 规则或继续短程 DPO。

## 与论文结果的关系

协议不同的系统只能做系统级参考，不能声称严格组件优越。

### AISHELL-1 / AISHELL-NER

| 系统 | 年份/会议 | CER | NE-CER | NE recall | 本项目对比 |
|---|---|---:|---:|---:|---|
| SeACo-Paraformer | 2023 | 2.48% | - | 90% hotword | 同 808 热词集 |
| SeACo-Paraformer + ASF | 2023 | 2.27% | - | 94% hotword | 同 808 热词集 |
| Efficient Text Augmentation | Interspeech 2024 | 4.50% | - | - | 同类上下文集 |
| CB-Whisper | LREC-COLING 2024 | MER 8.6% | - | 82.4% entity | 原始方法 |
| Confidence Homophone Detector | Interspeech 2024 | 6.46% | - | 85.5% hotword | Test-Aishell1-Middle |
| PARCO | ASRU 2025 | 4.22% | - | - | 含 1000 distractors |
| Generative Annotation for ASR NEC | EMNLP 2025 | 9.85% | 7.41% | 87.31% | 上游和实体协议不同 |
| DBA-wav2vec 2.0 | Sensors 2026 | 6.97% | - | - | 普通 AISHELL-1 ASR |
| Streaming Decoder-Only LLM ASR | 2026 | 5.10% | - | - | 普通 AISHELL-1 ASR |
| RASTAR-8B | 2026 | 4.21% | 6.21% | 89.33% | 完整 AISHELL-NER |
| **本项目 routed CB-SenseVoice + COVO** | 2026 | **3.2129%** | **3.9546%** | **91.45%** | 808 条使用上下文 |

主要参考：

- [CB-Whisper, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.262/)
- [SeACo-Paraformer](https://arxiv.org/abs/2308.03266)
- [Efficient Text Augmentation, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/zheng24_interspeech.pdf)
- [Confidence-based Homophone Detector, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/yang24j_interspeech.pdf)
- [PARCO, ASRU 2025](https://arxiv.org/abs/2509.04357)
- [Generative Annotation, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1052/)
- [RASTAR](https://arxiv.org/abs/2601.17264)

### THCHS-30

完整 THCHS-30 test 上，本项目 SenseVoice+COVO 为 `5.601%` CER。公开直接
参考包括 Interspeech 2018 的 LSTM `11.93%`、TDNN-LSTM `10.97%`、
mGRUIP-B `10.38%`，以及 Interspeech 2024 的 Whisper-medium `8.60%`、
Whisper-large-v2 `6.80%`、HuBERT-CTC `6.00%`。

- [Interspeech 2018](https://www.isca-archive.org/interspeech_2018/li18k_interspeech.pdf)
- [Interspeech 2024](https://www.isca-archive.org/interspeech_2024/li24s_interspeech.pdf)

## 环境与主要模型

必须使用准备好的环境，不要在 base 环境运行：

```bash
conda activate /root/autodl-tmp/great
cd /root/autodl-tmp/src
```

主要模型：

```text
SenseVoice:
  iic/SenseVoiceSmall

SenseVoice KWS:
  src/outputs/aishell_sensevoice_kws/checkpoints/f1G/
  f1G-epoch=16-step=102085.ckpt

Phrase context adapter:
  src/outputs/sensevoice_context_adapter/
  phrase_crossattn_aishell_full_20260722.pt

COVO base:
  cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-4B

COVO CER adapter:
  cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/
  qwen35_cbwhisper_preserve2_aishell_train_noop_1epoch_bf16_bs7

COVO joint adapter:
  cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/
  qwen35_cbwhisper_hotword_preserve_dpo_30steps_bf16
```

当前 CB-SenseVoice 配置：

```text
src/configs/cb-sensevoice-aishell.yaml
```

其中接受的 phrase confidence weight 为 `14.0`。

## 数据准备与运行

### SenseVoice KWS 数据

准备 AISHELL 目录结构：

```bash
python src/analysis/prepare_sensevoice_kws_dataset.py \
  --source datasets/aishell/data_aishell \
  --target datasets/aishell/data_aishell_sensevoice
```

提取 SenseVoice 隐藏层：

```bash
src/scripts/run_sensevoice_kws_extraction_20260714.sh
```

训练 KWS：

```bash
cd src
/root/autodl-tmp/great/bin/python run_CLI.py fit \
  --config configs/train-sensevoice-kws.yaml
```

### CB-SenseVoice 验证

```bash
cd src
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

导出 COVO evidence：

```bash
CBW_EVIDENCE_ONLY=1 \
CBW_EVIDENCE_OUT=logs/cb_sensevoice_evidence.jsonl \
/root/autodl-tmp/great/bin/python run_CLI.py test \
  --config configs/cb-sensevoice-aishell.yaml
```

完整 AISHELL train evidence 可按 shard 断点生成：

```bash
src/analysis/run_aishell_sensevoice_w14_train_evidence_shards.sh
```

### COVO 输入

```bash
cd src
/root/autodl-tmp/great/bin/python analysis/cbwhisper_covo_bridge.py prepare \
  --input logs/cb_sensevoice_evidence.jsonl \
  --output logs/cb_sensevoice_covo_messages.jsonl \
  --max-nbest 6 \
  --max-pinyin 3 \
  --max-hotwords 8 \
  --max-prompt-hotwords 6 \
  --max-candidates-with-scores 6 \
  --hotword-source all \
  --include-pinyin \
  --protect-supported-hotwords
```

COVO inference 使用
`cbwhisper_covo_migration_20260609_tar_extracted/covo/scripts/infer_lora_text.py`。
训练数据中的 assistant target 必须是完整纠错句子：

```json
{"text":"纠错后的完整句子"}
```

## 重要文件

```text
src/model/cb_whisper.py
src/analysis/train_sensevoice_context_adapter.py
src/analysis/extract_sensevoice_hidden_states.py
src/analysis/prepare_sensevoice_kws_dataset.py
src/analysis/cbwhisper_covo_bridge.py
src/analysis/build_covo_mixed_dpo_pairs.py
src/analysis/build_routed_aishell_predictions.py
src/configs/cb-sensevoice-aishell.yaml
src/configs/train-sensevoice-kws.yaml
EXPERIMENT_CONTEXT.md
AISHELL_EXPERIMENTS.md
```

## 实验纪律

- 每次代码修改后提交 Git，方便恢复。
- 每轮实验记录数据范围、评价口径、checkpoint 和相对变化。
- 小实验、全量实验、oracle 和 test-leak diagnostic 必须明确标注。
- KWS checkpoint 不随意更换；只有明确的 KWS 实验才重新训练。
- 主方法保持轻量和可解释，不把数据集特例补丁包装成论文贡献。
- 实验失败时记录原因并回滚运行配置，不覆盖当前最好模型。
- 大型数据、checkpoint、JSONL 和运行日志默认不提交 GitHub。

## 上游项目与许可

本仓库基于 CB-Whisper / Enhance-CB-Whisper 和 COVO 实验代码继续开发。
使用数据和模型时请同时遵守各上游项目、SenseVoice、Qwen 和数据集的许可。
本仓库自身许可见 [`LICENSE.md`](LICENSE.md)。
