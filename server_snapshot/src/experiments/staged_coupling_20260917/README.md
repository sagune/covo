# 分阶段连续声学耦合 pilot

目标：验证 COVO 能否利用 CBSense/SenseVoice 的连续音频表示，在不降低已有热词召回的前提下降低 CER。此目录不修改旧模型、旧训练代码或旧实验结果。

## 实验边界

- AISHELL：4096 条哈希固定训练样本；64 条固定 dev screen；1334 条 full dev。
- 排除与 dev 相同的训练转写；仅继承原有上下文协议。本轮是诊断 pilot，不是未接触测试集上的论文最终结果。
- 训练源 17301 条约 75% 为错误输入，非自然分布；不能把本轮外推成全数据结论。
- 固定已有 AISHELL `high_lr3e-5/adapter/checkpoint-625`，不用 ST-CMDS adapter 替代。
- 连续有效 encoder 帧 float16 缓存，去掉 4 个查询帧；每 4 帧均值池化，末尾不足 4 帧也保留。参考文本不参与声学导出。
- 连接器：LayerNorm(512) → Linear(512,1024) → GELU → Linear(1024,4096)，约 472 万参数。

## 自动流程

1. 导出 train 和完整 dev 的连续声学缓存。
2. A：冻结 9B 和已有 adapter，仅训练连接器，纯音频转写，512 optimizer steps，累积 8 条/step。
3. 在固定 64 条 dev 上比较正确音频、全零音频 embedding、无自配对的相近长度错配音频。正确音频须输出可解析非空 JSON，且 CER 低于两个音频消融，否则结束并记录负结果。
4. A 通过后，生成相同提示词的纯文本基线；B 训练连接器与已有 LoRA，共 512 steps。
5. B 原始输出须同时满足：CER 优于匹配提示词文本保护基线及旧提示词保护强基线，热词 hits 均不降低，并优于同一 checkpoint 的零/错配音频。失败停止，不更换门槛。
6. 固定最终 checkpoint 通过 screen 后，写入选择记录，再跑 full dev 确认。没有自动 test 调参或自动扩大训练。

中间 256-step checkpoint 只用于留存，不参与选择；唯一候选是预先固定的 512-step checkpoint。所有输出另报原始和保护后指标，不把后处理收益当声学耦合收益。

## 命令（服务器）

工作目录 `/root/autodl-tmp`，脚本 `src/experiments/staged_coupling_20260917/run.py`，解释器 `/root/autodl-tmp/great/bin/python`。

按顺序执行：`prepare` → `selftest` → `export smoke` → `smoke` → `smoke-b` → `lock` → `pipeline`。

`smoke` 检查标签掩码、连接器梯度、冻结模型无梯度、保存重载和 inputs_embeds 生成。`smoke-b` 检查仅 LoRA 更新并执行真实 optimizer step。

正式 `pipeline` 应通过 nohup 或持久终端启动，日志与预测分开。不要重复启动；文件锁拒绝并行副本，已有阶段日志拒绝覆盖。失败后先审计，不要手动修改旧 run 的模型或数据。

## 留痕

`run/protocol.json` 保存首次数据/预算约定；`implementation_lock.json` 保存冒烟后的执行代码与前端权重哈希、运行库版本，并明确记录启动前实现修订。

`run/status.json` 为当前阶段或终态；`train_A.log`、`train_B.log` 记录步数/loss/梯度/显存/时间；`gate_*.json` 给出原始指标和通过原因。`.partial` 预测不是完成结果。`.predictions.jsonl` 经过行数和 ID 校验后才发布。

评估统一调用旧 `restore_eval.py` 的归一化与恢复函数：NFKC、去空格标点，不做繁简转换。报告 corpus CER、hits/mentions、丢失/新增热词、改善/恶化/不变句、空输出和解析失败。B 对保护强基线提供配对 bootstrap；点估计召回不下降不等于统计非劣性证明。

本流程不自动关机，不触发旧实验队列，不包含任何 SSH 密码。
