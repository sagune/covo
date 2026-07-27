<div align="center">

# Experimental Results

**CB-SenseVoice + COVO 实验结果汇总**

主结果 · 模块消融 · 跨数据集验证 · 论文对比

[返回 README](README.md) · [实验上下文](EXPERIMENT_CONTEXT.md) · [AISHELL 账本](AISHELL_EXPERIMENTS.md)

</div>

---

## 阅读说明

本文档只汇总具有明确数据范围和评价口径的结果。所有实验按以下标签区分：

| 标签 | 含义 |
|---|---|
| **Full** | 对目标测试范围中的全部样本进行评测 |
| **Pilot** | 固定子集上的方向验证，不作为最终结果 |
| **Oracle** | 使用参考答案选择候选，仅表示候选池上限 |
| **Diagnostic** | 诊断实验，可能包含额外词典或测试信息 |
| **Leaked** | 使用测试参考、测试错误或测试标签，不可作为论文主结果 |

### AISHELL 统一口径

- 文本归一化：`OpenCC t2s + NFKC + 去除空格和标点`
- CER：corpus character error rate
- `Recall@400`：AISHELL-NE 指定 400 个热词的召回率
- `R1 Recall@226`：其中 226 个困难热词的召回率
- `AISHELL-NE 808`：808 条具有上下文热词标注的语音
- `AISHELL-1 Full`：官方 test 全量 7,176 条语音

不同论文、数据集和历史实验可能采用不同 normalization 或上游 ASR。表中会
明确标注可比性，不把协议不同的数字描述为严格同条件超越。

## 一览

| 数据集 | 范围 | 系统 | CER | 上下文指标 |
|---|---:|---|---:|---:|
| AISHELL-NE | 808 Full | **CB-SenseVoice w14 + DPO-30 COVO** | **3.1354%** | Recall@400 **90.50%** |
| AISHELL-NE | 808 Full | w14 + preserve2 COVO | **3.0268%** | Recall@400 89.25% |
| AISHELL-1 | 7,176 Full | **Routed + DPO-30 COVO** | **3.2062%** | NE recall 91.453% |
| THCHS-30 | 2,495 Full | SenseVoice + COVO | **5.601%** | - |
| ST-CMDS | 5,130 Held-out | ChineseHP-style COVO | **5.167%** | - |
| 水利课程 | 1,152 Full | CB-SenseVoice + COVO | **3.887%** | Hotword recall 93.136% |

## AISHELL-NE 808

### 主结果

| 系统 | CER | Edits | Exact | Recall@400 | R1 Recall@226 | 角色 |
|---|---:|---:|---:|---:|---:|---|
| SenseVoiceSmall | 10.3609% | 1,335 | 268 | 159/400 | 23/226 | 无上下文基线 |
| SenseVoice + preserve2 COVO | 7.7299% | - | - | 173/400 | 29/226 | COVO-only 消融 |
| 原始 CB-SenseVoice | 7.8231% | 1,008 | 432 | 313/400 | 141/226 | 无适配器 |
| + frame position adapter | 7.6911% | 991 | 436 | 318/400 | 146/226 | 第一代适配器 |
| + phrase cross-attention | 4.8273% | 622 | 498 | 323/400 | 150/226 | 短语级声学建模 |
| **CB-SenseVoice w14** | **3.7951%** | **489** | 536 | **366/400** | **192/226** | CB 单模块主结果 |
| w14 + preserve2 COVO | **3.0268%** | **390** | **587** | 357/400 | 184/226 | 最低 CER 消融 |
| **w14 + DPO-30 COVO** | **3.1354%** | **404** | 575 | **362/400** | **189/226** | **联合主结果** |

联合主结果选择 DPO-30，是因为它在保持 `3.1354%` CER 的同时达到项目设定的
`90%` 热词召回目标。Preserve2 的 CER 更低，但 Recall@400 为 `89.25%`。

### 相对改进

| 对比 | CER 绝对下降 | CER 相对下降 | Recall@400 变化 |
|---|---:|---:|---:|
| SenseVoice → CB-SenseVoice w14 | 6.5658 点 | 63.37% | +51.75 点 |
| SenseVoice → w14 + DPO-30 | 7.2255 点 | 69.74% | +50.75 点 |
| 原始 CB-SenseVoice → w14 | 4.0280 点 | 51.49% | +13.25 点 |
| w14 → w14 + DPO-30 | 0.6597 点 | 17.38% | -1.00 点 |

### 热词区域与运行效率

为补齐热词模块论文常用指标，2026-07-27 在当前代码上对 808 条测试集进行了
受控重跑。所有文本统一采用 OpenCC `t2s`、NFKC 和去标点；热词区由
`hotword/test/aligned.txt` 中的指定热词在参考文本里的字符范围确定。插入错误
只有位于热词内部时才计入热词区，否则计入非热词区。

FIR 是负样本级 false insertion rate：448 条不含指定热词的语句中，输出新增
词表热词的语句比例。运行时间包含模型加载和 808 条解码，不包含 CER 统计；
音频总长 4,550.55 秒，硬件为单张 RTX 5090。

| 系统 | 重跑 CER | 热词区 CER | 非热词区 CER | FIR | 时间 | RTF | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SenseVoiceSmall | 10.3764% | 24.4380% | 8.6911% | 0/448 = 0% | 51 s | **0.0112** | **1,603 MiB** |
| 原始 CB-SenseVoice | 7.7765% | 11.3125% | 7.3527% | 1/448 = 0.2232% | 592 s | 0.1301 | 3,783 MiB |
| + phrase cross-attention | 4.8273% | 10.2248% | 4.1804% | 1/448 = 0.2232% | 602 s | 0.1323 | 3,787 MiB |
| **CB-SenseVoice w14** | **3.7951%** | **4.6410%** | **3.6937%** | 1/448 = 0.2232% | 593 s | 0.1303 | 3,787 MiB |

裸 SenseVoice 和无适配器版本的受控重跑分别比历史主表波动 `+0.0155` 和
`-0.0466` 个百分点；phrase-cross 与 w14 精确复现历史 edits。区域分析统一
使用本次重跑文本，不混合后来被覆盖的历史预测文件。

## AISHELL-1 全量

808 条有外部上下文词表的语句使用 CB-SenseVoice w14，其余 6,368 条使用
裸 SenseVoice。路由只依据是否存在上下文词表，不查看模型置信度或参考答案。

| 系统 | Corpus CER | Edits | Exact | 协议 |
|---|---:|---:|---:|---|
| SenseVoiceSmall | 6.2922% | 6,591 | 4,454 | Full，开放词表 |
| SenseVoice + 原始 COVO | 5.7053% | 5,977 | 4,769 | Full，开放词表 |
| **SenseVoice + preserve2 COVO** | **3.7647%** | **3,944** | **4,982** | Full，开放词表 |
| Routed SenseVoice / CB-SenseVoice w14 | 5.4894% | 5,751 | 4,722 | 808 条使用上下文 |
| **Routed + DPO-30 COVO** | **3.2062%** | **3,359** | **5,235** | **联合主结果** |
| Routed + preserve2 COVO | **3.1929%** | **3,345** | **5,247** | 最低 CER 消融 |

### AISHELL-NER 重统计

| 系统 | CER | NNE-CER | NE-CER | 实体召回 |
|---|---:|---:|---:|---:|
| **Routed + DPO-30 COVO** | **3.2129%** | **3.1343%** | **3.9546%** | **3103/3393 = 91.453%** |

不使用 test 实体词表的检索消融，仅从 AISHELL-NER train/dev 构建实体库：

| 系统 | CER | NNE-CER | NE-CER | 实体召回 |
|---|---:|---:|---:|---:|
| SenseVoice + COVO | 3.7732% | 3.2958% | 8.2777% | 82.85% |
| + exact-pinyin entity retrieval | **3.5384%** | **3.2927%** | **5.8572%** | **88.15%** |

## 模块消融

### SenseVoice KWS

SenseVoice 隐藏层维度为 512。KWS 使用单通道 TCResNet 对语音与热词隐藏层
相似度矩阵进行分类。

| 模型/范围 | 阈值 | Precision | Recall | F1 | 状态 |
|---|---:|---:|---:|---:|---|
| 原始 ResNet1 natural KWS test | 0.500 | 80.23% | 87.47% | 83.70% | 原论文结构对照 |
| **SenseVoice KWS test, TTS** | **0.787** | **90.948%** | **90.658%** | **90.803%** | 当前模型 |
| SenseVoice KWS, validation 阈值 | 0.954 | 95.721% | 83.121% | - | 高精度工作点 |

历史 top-k 诊断得到 Recall@1 `80.36%`、Recall@3 `95.22%`、
Recall@6 `96.92%`、Recall@12 `98.62%`。这说明召回瓶颈主要位于 ASR
候选实现，而不是 KWS 候选检索。

### 高浓度热词诊断

为确认热词是否能明显影响整体 CER，另构造了错误定向的 oracle-context
诊断集。目标词由测试参考文本和裸 SenseVoice 错误共同产生，因此属于
`Pilot + Diagnostic + Leaked`，只表示模块能力上界，不能用于论文主结果。

| 数据 | 词表 | 热词正样本率 | 字符覆盖 | 裸 ASR 错误覆盖 |
|---|---:|---:|---:|---:|
| AISHELL-1 targeted | 3,499 | 52.26% | 16.65% | 85.50% |
| ST-CMDS targeted | 5,241 | 51.73% | 15.39% | 92.49% |

扩词时发现旧 KWS 会为每个 100 词分组强制补入一个低于阈值的词，使误检随
词表规模增长。关闭分组补词，并将大词表工作阈值从 `0.787` 提高到 `0.95`：

| 数据/范围 | 裸 SenseVoice CER | Targeted KWS CER | 绝对下降 | Gold 热词 CER |
|---|---:|---:|---:|---:|
| AISHELL 前 71 条 | 3.299% | **2.680%** | **0.619 点** | **0.206%** |
| ST-CMDS 前 51 条 | 6.314% | **5.631%** | **0.683 点** | - |

AISHELL 中原阈值 targeted KWS 为 `3.196%`；提高精度后相对裸模型降低
`18.8%`。Gold 与实际 KWS 的差距说明上下文解码能够利用正确热词，下一瓶颈是
大词表中高召回、低误检的候选检索。

### Phrase Cross-Attention

当前上下文适配器约有 729k 个可训练参数，训练使用 AISHELL train 中完整的
17,301 条热词语句，共 2,162 optimizer steps。

适配器输出的 phrase-presence probability 以如下方式进入 CTC prefix beam：

```text
effective_hotword_score = KWS_score * (1 + weight * phrase_probability)
```

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

`weight=14` 同时取得最低 CER 和最高 top1 召回，因此作为默认配置。

### COVO

COVO 输入包含 CB-SenseVoice top1、六条带来源和分数的 N-best、三条拼音
候选、KWS/prompt hotwords 以及候选支持的 protected hotwords。

| Adapter | 额外训练数据 | CER | Exact | Recall@400 | 决策 |
|---|---:|---:|---:|---:|---|
| **DPO-30** | 历史 hotword pairs | **3.1354%** | 575 | **362/400** | 保留 |
| Targeted DPO-20 | 117 correction + 60 hotword | 3.1354% | **584** | 355/400 | 召回下降 |
| Hotword-only DPO-10 | 120 hotword preferences | 3.1354% | 573 | **362/400** | 无收益 |

短程偏好学习已接近饱和：后续微调主要重新分配错误，无法继续降低总 edits。
下一步收益更依赖候选生成质量。

## 跨数据集结果

### THCHS-30

| 范围 | Samples | SenseVoice CER | + COVO CER | Improved | Worsened |
|---|---:|---:|---:|---:|---:|
| 国内镜像 Pilot | 1,339 | 6.698% | **4.733%** | 331 | 14 |
| **完整 test** | **2,495** | **7.959%** | **5.601%** | **676** | **17** |

COVO 只使用 AISHELL 训练的 preserve2 adapter，没有使用 THCHS-30 测试标注。

### ST-CMDS

固定划分 seed 为 `20260719`：95,418 train、2,052 dev、5,130 held-out test。

| 实验 | Samples | 输入 CER | COVO CER | Improved | Worsened | 标签 |
|---|---:|---:|---:|---:|---:|---|
| 随机设备分层 | 2,000 | 6.101% | **5.788%** | 51 | 11 | Pilot |
| ChineseHP-style held-out | 5,130 | 5.641% | **5.167%** | 545 | 330 | Full split |
| 未做 ST-CMDS 训练的 DPO60 | 5,130 | 5.641% | 6.969% | 498 | 1,065 | 分布不匹配 |

ChineseHP-style 模型在完整 95,418 条 train 上训练一轮，共 5,964 optimizer
steps，dev loss 为 `0.2392`。

### 水利课程录音

共 1,152 条自然口语语音。COVO 未使用该测试集训练。

| 系统 | Number-normalized CER | Raw CER | Hotword recall | Edits | Exact |
|---|---:|---:|---:|---:|---:|
| SenseVoice | 4.636% | 4.654% | 82.884% | 731 | 722 |
| CB-SenseVoice | 4.059% | 4.076% | 93.050% | 640 | 763 |
| **AISHELL DPO60 listwise COVO** | **3.887%** | **3.905%** | **93.136%** | **613** | **774** |
| Candidate oracle | 1.820% | - | - | - | - |

### 新水利视频集

共 990 条。只有 180 词版本可作为独立实验；其余行使用测试错误、测试标签或
测试衍生词表，只表示诊断上限。

| 系统 | CER | Filler+number CER | Hotword recall | 标签 |
|---|---:|---:|---:|---|
| SenseVoice | 4.873% | 4.902% | 82.567% | Full baseline |
| CB-SenseVoice, 180 terms | 4.798% | - | 97.849% | Full independent |
| CB-SenseVoice, 259 terms | 4.072% | 4.074% | 95.157% | Leaked |
| AISHELL DPO60 listwise COVO | 3.736% | 3.722% | - | Leaked |
| Confusion-term DPO70 | 3.540% | 3.522% | - | Leaked |
| Confusion-term DPO70, w=0.3 | **3.503%** | **3.484%** | **95.642%** | Leaked upper bound |
| Candidate oracle | 2.227% | 2.208% | - | Oracle |

### Speechio-Formal

该任务要求将口语改写为正式书面语，与普通逐字 ASR 不同。现有 COVO 未针对
该任务训练，因此跨域收益有限。

| Domain | Samples | SenseVoice formal CER | COVO formal CER |
|---|---:|---:|---:|
| ZH00000 | 879 | 25.8634% | **25.8014%** |
| ZH00006 | 1,561 | 19.8511% | **19.7852%** |

## 与公开论文比较

只有相同或相近协议可以用于论文主张；`Informative` 表示数据、上游 ASR 或
归一化存在差异。

| 年份 | 系统 | 数据集/协议 | 论文结果 | 本项目最近结果 | 可比性 |
|---:|---|---|---|---|---|
| 2023 | SeACo-Paraformer | AISHELL-NE 808 | CER 2.48%，Recall 90% | 3.135%，90.50% | 相同名义子集 |
| 2023 | SeACo + ASF | AISHELL-NE 808 | CER 2.27%，Recall 94% | 3.135%，90.50% | 本项目未超过 |
| 2024 | CB-Whisper | AISHELL contextual | MER 8.6%，Recall 82.4% | 3.135%，90.50% | 原始思想基线 |
| 2024 | Efficient Text Augmentation | AISHELL-NE | CER 4.50% | 3.135% | 需确认 normalization |
| 2024 | Confidence Homophone Detector | AISHELL1-Middle | CER 6.46%，Recall 85.5% | 3.135%，90.50% | 名义子集相近 |
| 2025 | Adaptive Context Biasing | AISHELL contextual | CER 5.56%/6.01% | 3.135% | Informative |
| 2025 | Generative Annotation NEC | AISHELL NEC | CER 9.85%，NE-CER 7.41% | 3.213%，NE-CER 3.955% | 上游与协议不同 |
| 2025 | PARCO | AISHELL-1 + 1000 distractors | CER 4.22% | Full 3.765% | Informative |
| 2026 | DBA-wav2vec 2.0 | AISHELL-1 ASR | CER 6.97% | Full 3.765% | 普通 ASR 比较 |
| 2026 | Streaming decoder-only LLM ASR | AISHELL-1 ASR | CER 5.10% | Full 3.765% | 模型设置不同 |
| 2026 | RASTAR-8B | AISHELL-1 NEC | CER 4.21%，NE-CER 6.21%，Recall 89.33% | 3.213%，3.955%，91.45% | 协议相近，仍需统一复核 |

### 参考链接

- [SeACo-Paraformer](https://arxiv.org/abs/2308.03266)
- [CB-Whisper, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.262/)
- [Efficient Text Augmentation, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/zheng24_interspeech.pdf)
- [Confidence-based Homophone Detector, Interspeech 2024](https://www.isca-archive.org/interspeech_2024/yang24j_interspeech.pdf)
- [PARCO, ASRU 2025](https://arxiv.org/abs/2509.04357)
- [Generative Annotation for ASR Named Entity Correction, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1052/)
- [RASTAR](https://arxiv.org/abs/2602.12287)

## 失败路线

| 路线 | 观察 | 决策 |
|---|---|---|
| 反复调整 exact/phonetic/consensus 权重 | 候选池不变时收益接近零 | 停止规则堆叠 |
| 简单扩大 N-best | 热词召回提高，整体 CER 恶化 | 优先提高候选质量 |
| Prompt 排列、recency、逐词注入 | 没有稳定收益 | 不作为主方法 |
| 窄/宽窗口 hotword CTC | 弱于 phrase adapter | 回滚 |
| Monotonic hard activation | 召回略升，CER 变差 | 仅保留消融 |
| Neutral/context dual branch | Rerank 易选中弱候选 | 回滚 |
| Mixed COVO DPO | 模型过度保守 | 回滚 |
| Actual-evidence targeted DPO | CER 不变，召回下降 | 不扩大全量 |
| 测试词典与测试标签训练 | 指标改善但发生数据泄漏 | 只作上限诊断 |

## 当前结论

1. KWS top-k 已不是主要性能上限。
2. Phrase-level 声学证据显著改善 SenseVoice 的热词候选生成。
3. COVO 能降低普通字符错误，但可能损失少量已正确热词。
4. 下一阶段应增加“热词正确且整句质量良好”的候选，而不是继续堆叠 rerank
   规则或做极短程 DPO。
5. 论文主结果排除 test-derived lexicon、test-label DPO、candidate oracle 和
   smoke subset。

## 结果来源

主结果对应的机器可读摘要保存在 `src/logs/`，包括：

- `aishellne808_cb_sensevoice_acoustic_phrase_w14_covo_dpo30_unified_eval_20260724.json`
- `aishellne808_cb_sensevoice_acoustic_phrase_w14_covo_preserve2_unified_eval_20260724.json`
- `aishell_full_routed_sensevoice_cb_sensevoice_w14_covo_20260724_summary.json`
- `aishell_full_routed_sensevoice_cb_sensevoice_w14_covo_20260724_ner_eval.json`
- `thchs30_full_sensevoice_summary.json`
- `stcmds_chinesehp_covo_from_dpo60_metrics_20260719.json`

历史逐轮记录见 [AISHELL_EXPERIMENTS.md](AISHELL_EXPERIMENTS.md)，当前复现
上下文见 [EXPERIMENT_CONTEXT.md](EXPERIMENT_CONTEXT.md)。
