# 文本条件化声学残差连接器 v1

独立于 `staged_coupling_20260917`；不修改旧实现、缓存、检查点或结果。

## 结构与边界

ASR top1 的原始 9B token embedding → LN/投影至 256 维 → 正弦位置编码＋一层无 dropout 的上下文 Transformer → 查询 SenseVoice 特征。声学分支仍对连续有效 512 维特征每四帧均值池化（末尾全部保留），投影到 256 维并加入位置编码。单层 4-head cross-attention → LN → 零初始化输出投影 → 残差加到原 top1 token embedding。

- 不插入音频前缀；不生成第二份转写；不做热词保护、输出恢复或候选限制。
- 完整文本 prompt 一次性 tokenize，以 offset mapping 标定 JSON 的 asr_top1 内容。重复出现在 nbest 的同一句不被误选。边界 token 若同时包含 JSON 引号等字符，将作为一个整体接收残差；不会重新切词。
- Query 只来自输入 top1，不读取 reference、oracle_hotwords、keyword_mentions 或目标 token。其余提示词、N-best、热词保持旧文本基线格式。
- 零初始化严格保持文本 embedding。第一步只有末层得到非零梯度，末层更新后梯度进入上游注意力；测试专门检查此行为。
- `off/text` 完全跳过连接器，不加载音频，token 数、位置和 prompt 相同。`zero` 保留连接器但清零编码器特征，不等于 off；位置编码、参数偏置仍存在。
- `mismatch` 沿用旧相近长度错配映射：无同 ID/同参考配对，但帧数可能不同，不能声称完全消除了长度混杂。
- 声学残差关闭只能回到原文本模型，不保证该模型不误改；不能保证训练后 CER 或召回非劣。

## 训练与公平对照

提供 `residual` 和 `mlp` 两个独立配置，均冻结 SenseVoice（只读特征）、9B 和 checkpoint-625 LoRA，只训练连接器，以输入音频＋已有识别证据预测最终 JSON。保留原 4096 个固定训练样本，shuffle seed 917，累积 8、512 optimizer steps、lr 5e-5、warmup 32。不更改损失或按当前 dev 错误选训练样本。

训练集是原有错误富集子集，包含正确输入但不是自然分布或 AISHELL 全 train。prepare 会记录正确/错误样本数，核对 train/dev 的 ID 和归一化参考互斥、固定数据哈希，并审核全部 train/dev 声学缓存。没有自动启动训练的 supervisor。

MLP 默认从 Afull_3243 开始，残差结构从零输出开始，故相同的是本轮纠错训练预算和数据，不是累计预训练历史或参数量。它是实用对照，不能单凭这个对照声称纯结构因果收益。原先“连接器＋LoRA 一起更新”的 B 流程仍在旧目录，完全未修改。

## 命令（服务器）

工作目录：`/root/autodl-tmp/src/experiments/text_residual_coupling_20260920`。
解释器必须用 `/root/autodl-tmp/great/bin/python`，不是 base 环境。

```bash
/root/autodl-tmp/great/bin/python -m unittest -v test_connector
/root/autodl-tmp/great/bin/python preflight.py
/root/autodl-tmp/great/bin/python workflow.py prepare --architecture residual --output runs/residual_v1
/root/autodl-tmp/great/bin/python workflow.py prepare --architecture mlp --output runs/mlp_B_control
/root/autodl-tmp/great/bin/python workflow.py verify runs/residual_v1
```

以上均不加载 9B、不使用 GPU。CPU 单元测试不等于真 9B 集成测试或实验有效性验证。

有卡且无其他 GPU 作业时，再显式依次运行（两个配置顺序执行，不并发）：

```bash
/root/autodl-tmp/great/bin/python workflow.py smoke runs/residual_v1
/root/autodl-tmp/great/bin/python workflow.py train runs/residual_v1
/root/autodl-tmp/great/bin/python workflow.py eval runs/residual_v1
/root/autodl-tmp/great/bin/python workflow.py smoke runs/mlp_B_control
/root/autodl-tmp/great/bin/python workflow.py train runs/mlp_B_control
/root/autodl-tmp/great/bin/python workflow.py eval runs/mlp_B_control
```

真实 GPU smoke 使用一个 train 样本：检查原文本路径、零初始化 logits/greedy 输出一致、两步反向传播、上游梯度及骨干无梯度。不保存 smoke 更新后的权重；正式训练重置 seed、重新初始化。没有匹配协议的 smoke PASS 就拒绝训练。

## 评估与留痕

固定最终 step512 后才全量评估，5 个条件各覆盖当前协议全部 1334 条 dev：纯文本、正确音频、关闭分支、零音频、错配音频。NFKC 去空格标点、不做繁简转换；21102 参考字、599 指定热词 mentions。使用旧原始输出 evaluator，不调用保护恢复。解析失败计为空串，任何异常保留 partial 并停止，不漏样本，不自动减量，不在64条上做结论。

逐组保存预测、metrics、文件哈希和正确音频相对 text/zero/mismatch 的配对 bootstrap，额外统计替换/删除/插入、原本正确句被改错的句数和字符数。分别报告相对文本的 CER/召回联合点估计、前端热词召回是否保留。已有 dev 被多次查看，结果只能标诊断，不是独立 test；点估计不是统计非劣性证明。

协议和缓存哈希在 prepare 时冻结；之后实现变化须准备新目录，不能改锁后继续原 run。训练/评估拒绝覆盖已有尝试。失败时保留原始证据，需要人工诊断，没有自动恢复、换模型或重训。旧目录和原 checkpoints 始终只读。
