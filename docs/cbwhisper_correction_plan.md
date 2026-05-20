# CB-Whisper 受约束后纠错计划

## 目标

在现有 `cbwhisper` 工作基础上，做一个可以作为论文第二个创新点的后端模块：

```text
CB-Whisper 输出
+ N-best 候选
+ KWS 热词与置信度
+ 拼音证据
+ 可选的声学/相似度证据
-> 受约束后纠错
-> verifier / fallback
-> 最终转写
```

核心目标不是简单调用大模型 API，也不是再调一个 rerank 分数，而是提出一个 **CB-Whisper-aware 的热词保护型生成式后纠错方法**。它应当能在不明显增加误改率的前提下，提升中文热词识别质量和整体转写质量。

## 方法定位

现有 CB-Whisper 已经包括：

- KWS 检索热词。
- 热词 prompt 注入 Whisper。
- N-best 生成与重排序。
- 拼音相似度打分。
- consensus rerank。
- 保守的 phonetic surface repair。

因此第二创新点不建议继续写成“新的 rerank score”。更合适的叙事是：

```text
从 rerank-only 扩展到 constrained generative correction。
```

也就是：

- 当正确答案已经在 N-best 中时，优先选择/保留候选。
- 当 N-best 里没有完全正确句子，但存在拼音、热词、局部片段证据时，允许做最小编辑。
- 当纠错结果不可信时，回退到 CB-Whisper 原输出。

## 可参考工作

### CoVoGER

CoVoGER 是一个 N-best generative error correction 框架：

```text
ASR/ST 5-best hypotheses -> LLM -> corrected transcription / translation
```

可以借鉴：

- N-best 到 instruction tuning 数据的构造方式。
- LoRA 微调流程。
- 推理与评测脚本结构。

但本项目不能直接照搬 CoVoGER。我们要加入：

- CB-Whisper 的 KWS 热词。
- 热词置信度。
- 中文拼音证据。
- verifier / fallback。
- 可选的热词位置或相似度矩阵证据。

### ASR-EC Benchmark

ASR-EC 的结论对本项目很重要：

- 单纯 prompting 容易 over-correction。
- LoRA 微调比 prompting 更稳定。
- 音频 + 文本的多模态纠错效果最好。

本项目可以借鉴其监督微调思想，但先做轻量文本侧和 CB-Whisper 中间证据侧，不直接复现大多模态模型。

### ChineseHP

ChineseHP 提供中文 ASR 的 N-best 与拼音数据，适合训练：

```text
ASR hypothesis + N-best + pinyin -> reference
```

本地已经整理出：

```text
chinesehp_aishell1.jsonl
```

可作为第一批训练数据。

## 总体流程

### 阶段 1：规则版原型

目的：快速验证“热词 + 拼音 + N-best 约束纠错”是否有空间。

输入：

```text
CB-Whisper top1
CB-Whisper N-best
KWS hotwords
KWS scores
pinyin evidence
ASR scores
```

操作：

1. 判断是否需要纠错。
2. 找到高置信热词缺失或疑似错写的片段。
3. 用拼音一致性和 N-best 支持做局部替换。
4. 用 verifier 检查安全性。
5. 不安全则回退原输出。

输出：

```text
final_transcript
correction_trace
fallback_reason
```

这一版不需要训练，可以先接在 `cbwhisper` 的 test output 后面跑。

### 阶段 2：构造训练数据

统一构造 SFT 数据，建议格式：

```json
{
  "instruction": "根据 CB-Whisper 输出、N-best、热词和拼音证据，只做必要的 ASR 后纠错。",
  "input": {
    "asr": "...",
    "nbest": ["...", "...", "..."],
    "hotwords": [
      {"text": "...", "score": 0.92},
      {"text": "...", "score": 0.76}
    ],
    "pinyin": {
      "asr": "...",
      "nbest": ["...", "..."]
    }
  },
  "output": {
    "edits": [
      {"from": "...", "to": "...", "reason": "pinyin_match_and_hotword_support"}
    ]
  }
}
```

优先输出 `edits`，不是完整句子。原因：

- 更容易控制过纠错。
- 更容易做 verifier。
- 更容易解释模型行为。

数据来源：

1. ChineseHP AISHELL-1。
2. ASR-EC Benchmark。
3. CB-Whisper 在 AISHELL/Shuili 上跑出的预测、N-best、KWS 日志。
4. 可选：DeepSeek API 作为 teacher 生成初始 edits，但不能作为最终主方法。

### 阶段 3：小模型 LoRA 训练

推荐模型按优先级：

```text
Qwen3.5-4B LoRA
Qwen3.5-9B QLoRA
Qwen3.5-2B 快速实验
DeepSeek-R1-Distill-Qwen-7B 作为对比 baseline
```

不建议把 DeepSeek-R1-Distill 作为主模型，因为它偏 reasoning，容易输出解释或思考链；ASR 后纠错需要短输出、保守改动和严格格式。

训练目标：

```text
input evidence -> JSON edits
```

必要训练约束：

- 输出必须是合法 JSON。
- 只输出 edits，不输出解释。
- 没有必要纠错时输出空 edits。
- 不引入 N-best、热词、拼音证据之外的新实体。

### 阶段 4：Verifier 与回退

模型输出 edits 后，代码层做检查。

建议 verifier 规则：

1. JSON 合法性检查。
2. `from` 片段必须出现在原 ASR 或 N-best 中。
3. `to` 片段必须被热词、N-best 或拼音支持。
4. 单句编辑距离不能超过阈值。
5. 高置信热词不能被删除。
6. 数字、人名、地名变化必须有候选支持。
7. 输出长度异常增长则回退。

伪代码：

```python
edits = correction_model.predict(evidence)
if not verifier.is_safe(edits, evidence):
    final = cbwhisper_top1
else:
    final = apply_edits(cbwhisper_top1, edits)
```

### 阶段 5：接入 CB-Whisper

接入点建议在 CB-Whisper 完成 N-best rerank 之后：

```text
CB-Whisper forward
-> selected candidate
-> latest_forward_candidates
-> latest_keywords / latest_keyword_scores
-> constrained corrector
-> verifier
-> final text
```

需要从 `cbwhisper` 导出的信息：

- top1 transcription。
- N-best candidates。
- ASR score。
- rerank score。
- KWS keyword list。
- KWS score。
- pinyin representation。
- optional: KWS similarity matrix 或热词位置证据。

## 实验设计

### Baselines

```text
1. Whisper baseline
2. 原始 CB-Whisper
3. 当前改进 CB-Whisper
4. 当前改进 CB-Whisper + DeepSeek prompt correction
5. 当前改进 CB-Whisper + CoVoGER-style N-best correction
6. 当前改进 CB-Whisper + ours
```

### Ablation

```text
ours w/o hotwords
ours w/o pinyin
ours w/o N-best
ours w/o verifier
ours with full-sentence output instead of edits
ours with correction applied to all samples
ours with trigger-only correction
```

### Metrics

必须报告：

```text
CER
WER
Entity Recall
Hotword Only CER
Hotword Sentence CER
```

建议新增：

```text
Over-correction Rate
Correction Precision
Correction Recall
Fallback Rate
Safe Edit Acceptance Rate
```

其中 over-correction 可以定义为：

```text
CB-Whisper 原本正确或更接近 reference，但后纠错使结果变差的比例。
```

## 里程碑

### M1：数据与日志整理

- 整理 ChineseHP AISHELL-1 训练格式。
- 从 CB-Whisper 导出 top1、N-best、KWS、pinyin、reference。
- 建立统一 JSONL schema。

产物：

```text
data/processed/*.jsonl
```

### M2：规则版 constrained corrector

- 实现 trigger。
- 实现 pinyin-local edit。
- 实现 verifier/fallback。
- 跑 AISHELL 和 Shuili。

目标：

```text
不明显伤害 CER/WER；
证明热词局部修复有正收益或明确诊断失败原因。
```

### M3：LoRA 训练

- 基于 CoVoGER 思路构造 SFT 数据。
- 用 Qwen3.5-4B 做第一版 LoRA。
- 输出 JSON edits。
- 加推理脚本。

目标：

```text
模型能稳定输出合法 edits；
相较规则版有更高 correction recall；
通过 verifier 抑制 over-correction。
```

### M4：CB-Whisper 集成

- 将 corrector 接到 CB-Whisper 推理流程后。
- 支持开关配置。
- 输出纠错 trace。

目标：

```text
形成完整 pipeline：
audio -> cbwhisper -> constrained correction -> final output
```

### M5：论文实验

- 完整 baseline。
- 消融实验。
- 错误类型分析。
- 典型案例分析。
- 运行时间与成本分析。

## 风险与应对

### 风险 1：小模型过纠错

应对：

- 输出 edits 而不是完整句子。
- verifier/fallback。
- trigger-only correction。

### 风险 2：训练数据和 CB-Whisper 分布不一致

应对：

- 先用 ChineseHP/ASR-EC 预训练式 SFT。
- 再用 CB-Whisper 自己的预测日志做二阶段微调。

### 风险 3：热词上下文有噪声

应对：

- 不盲信 KWS。
- 引入 score threshold、rank、pinyin support、N-best support。
- 对 noisy hotwords 做 dropout 训练。

### 风险 4：CoVoGER 框架不支持最新 Qwen3.5

应对：

- 借 CoVoGER 的数据格式和训练思想。
- 训练代码优先使用 HuggingFace Transformers + PEFT + TRL。

## 推荐技术栈

```text
Python
PyTorch
Transformers
PEFT
TRL / SFTTrainer
bitsandbytes
pypinyin
jiwer
```

5090 单卡建议：

```text
Qwen3.5-2B: LoRA 快速试验
Qwen3.5-4B: 主力 LoRA
Qwen3.5-9B: QLoRA 强模型
```

## 最小可行版本

最小可行版本不训练模型，只做规则版：

```text
CB-Whisper N-best + hotword + pinyin
-> trigger
-> local edit
-> verifier/fallback
```

如果规则版有效，再进入 LoRA。

如果规则版无效，也能得到重要诊断：

- 正确答案是否经常不在 N-best。
- KWS 热词是否能到达后处理模块。
- 拼音证据是否足够区分同音/近音错误。
- 主要失败是否来自候选池、KWS、还是纠错策略。

## 论文表述草案

第二创新点可以表述为：

```text
本文提出一种面向 CB-Whisper 的热词保护型受约束生成式后纠错方法。
该方法融合 N-best 候选、KWS 热词置信度与拼音一致性证据，
通过局部编辑预测和安全验证机制抑制大模型后纠错中的过纠错问题，
在提升热词识别质量的同时保持整体 CER/WER 稳定。
```
