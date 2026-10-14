# 全类型复核v2：OOM后的授权重跑

用户于2026-10-07授权“重新跑一下，保证不会中途oom，然后总结之前改进让我汇报”。不能承诺数学意义上永不OOM，采取下列固定执行防护且先压力测试再正式训练。保留v1失败目录与日志，不将v1的smoke权重冒充928步正式权重。

## 未改变的学术实验

17294来源train、12396句产生18956候选块，49失败草稿保留。完整同音/异音/增删/未知候选，不用参考筛选路由；train参考只监督A/B选择。全上下文FULL-only，AUDIO真实CTC、TEXT未提供，所有输入/提示/分词/候选順序/标签/训练顺序/超参/数据SHA与v1相同。两臂重新从旧最佳B/text2162步初始化，一遍2370更新，29,097,984 LoRA参数；SenseVoice和9B基座冻结。无保护阈值、第三候选或新seed。

固定最终2370步，完整1334 contextual dev，21102参考字、599mentions，每组604决定；UNADAPTED/AUDIO/TEXT×none/true/zero/mismatch=12组，true_reversed仅顺序诊断。无dev挑检查点/阈值，不能称官方14326全dev或新heldout。

## 唯一执行修订

- 非重入激活重算（gradient_checkpointing=True，use_reentrant=False），以重计算换显存。两臂完全一致，重新做连续/跨进程精确重载检查。不声称与v1原无重算后端逐位等价。
- 不保留逐样本CUDA输入/mask/position缓存；预分词CPU bank保留，逐次传输，结束图释放后不积累GPU token缓存。
- PyTorch expandable_segments:True，在torch导入前设置，降低动态长度的分配碎片风险。
- 每128更新保存完整LoRA、optimizer/scheduler及Python/NumPy/CPU/CUDA RNG；写入.partial目录并校验哈希后原子重命名。只选择固定最终检查点，不按中间dev结果挑选。不删除任何原失败或checkpoint。
- 阶段训练若出现明确CUDA OOM且存在完整可核验断点，仅允许一次相同协议的新进程续训，重新加载权重与优化器/RNG。第二次失败停队。不得降低序列、删样本、改损失或换精度暗中补救。
- 各臂先16更新压力测试：8份最长train输入、原失败929批、覆盖最短至最长的变长批交替；优化器实际初始化/更新，压力测试权重丢弃。要求GPU缓存始终空、基座不变、梯度有限、最大reserved加外部占用后至少4GiB余量。压力测试只用train，不是训练/验证结果。
- 保存磁盘预留2GiB，当前约19GiB余量；预计38个周期点约12.7GiB，另smoke及final约2GiB，不删除旧实验。空间不足停队而不是损坏或覆盖。

## 流程

复用hash-verified v1已完成数据→独立CPU准备审计→两臂smoke/精确重载→两臂压力测试→AUDIO全量候选训练→TEXT匹配训练→三臂完整15套决定封存→12套全1334预测及指标→独立拼接/指标/CI/完整2370步训练记录核验。

远端新目录/root/autodl-tmp/src/experiments/all_edit_review_20261007_v2，独立后台串行pipeline，Python/root/autodl-tmp/great/bin/python，不恢复原v1后台队列。密钥严格核验，密码仅交互输入，不落盘。

## 信息边界与报告

遵循asr-result-validation进行结果完整性/评分口径审计，当前协议NFKC去空白及Unicode标点，无OpenCC。任何新CER/召回须完整验证后才能汇报。原实际异音复核三seed均值2.8038%/79.8553%；604块oracle1.8624%/92.8214%只是参考辅助上限。新全类型训练没有正式效果结果。

报告将区分连接器失败、接口/声学评分问题、显式候选监督已验证小幅收益、扩大同音覆盖的理论空间，以及待验证的新训练。压力测试降低已知OOM风险，不排除外部GPU抢占、硬件故障或未知分配峰值；若失败保留现场且禁止宣称验证完成。
