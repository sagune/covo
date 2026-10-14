# 保留的多数据集 9B 纠错主线

## 来源与保留规则

按用户于 2026-10-14 的确认，当前首页恢复到已开展五数据集实验的基础方法系列。这里转录 server_snapshot/RESULTS_ALL_DATASETS_20260826.csv 中系统名称包含 Qwen3.5-9B、分类为 full_held_out / full_cross_domain / full_test 的全部八行。筛选依据方法和协议类别，不按错误率高低筛选；未重算指标或重新选择检查点。

原 CSV SHA256：90ac026098e8d30700924fafff215e1917e8a90657d4b21b2be9c415ccf161bb。

| 数据集 / 范围 | 有效计分数 | Adapter / checkpoint | 指标 | 输入错误率 | 输出错误率 | 原分类 |
|---|---:|---|---|---:|---:|---|
| AISHELL-1 test | 7174 | AISHELL checkpoint-30022 | CER | 5.8354% | 2.6231% | full_held_out |
| ST-CMDS test | 5130 | ST-CMDS checkpoint-34400 | CER | 5.6407% | 4.4994% | full_held_out |
| MAGICDATA-READ test | 24279 | MAGICDATA checkpoint-39185 | CER | 3.7245% | 2.4334% | full_held_out |
| THCHS-30 test | 2495 | MAGICDATA checkpoint-39185 | CER | 5.1664% | 4.4023% | full_cross_domain |
| THCHS-30 test | 2495 | ST-CMDS checkpoint-34400 | CER | 5.1664% | 4.3358% | full_cross_domain |
| THCHS-30 test | 2495 | AISHELL checkpoint-30022 | CER | 5.1664% | 4.3111% | full_cross_domain |
| LibriSpeech test-clean | 2620 | final 2-epoch adapter | WER | 3.2277% | 2.6419% | full_test |
| LibriSpeech test-other | 2939 | final 2-epoch adapter | WER | 7.2789% | 5.9206% | full_test |

五数据集的不同 adapter 构成方法系列，不是同一套权重。THCHS-30 没有训练，三个 adapter 都保留；不凭其 test 数值挑选一个伪称预先指定的最终模型。LibriSpeech 这里只沿用源表的 final 2-epoch adapter 标识，不猜测未提供的 checkpoint 编号。

## 不混用的条件

- 上述中文/英文正式行的前端是 SenseVoice 10-best，MAGICDATA 为 WOITN。不能全部更名为 CBSense 热词增强结果。
- ST-CMDS 的 Standard-3139 CBSense + 9B、AISHELL contextual 4B 等联合实验保留在完整原表里，是不同前端/模型条件，不能替代上面行内的输入基线。
- THCHS-30 的 Error-targeted CBSense、MAGICDATA OracleLex 等使用测试参考构造上下文的行，仍保留为泄漏诊断，不混成正式 test 成绩。
- AISHELL 原评估器跳过两条空参考，因此有效计分数是 7174；此次未悄悄补回或改写原分母。
- 后续 1334 contextual dev 的 MLP、Residual、局部匹配、热词续训或声学复核都是另一次探索，不是这八行模型的正式验证。

## 模型/代码留痕

生成式后端源码在[主工程 CoVo 包](../../server_snapshot/covo/)和[历史迁移运行时](../../server_snapshot/cbwhisper_covo_migration_20260609_tar_extracted/covo/)，按原实验对应的实现与 adapter 运行。两版本存在差异，不能随意互换。CBSense 前端、KWS 与文本证据桥接仍在 server_snapshot/src/ 中。

模型权重、语音数据及原始预测未公开上传；源码清单不等于可无资源重现。SOURCE_MANIFEST 以及多数据集原表保持不变。

[完整原表](../../server_snapshot/RESULTS_ALL_DATASETS_20260826.md) · [旧尝试档案](../archive/README.md)。
