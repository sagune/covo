# 保留主线：多数据集 CBSense / SenseVoice + 9B CEGER

最终展示主线按用户确认：保留已经在多个数据集做过实验的基础识别候选 + 9B 纠错系列，不把后续连接器或声学复核尝试当作最终路线。

- [主线结果与模型索引](mainline-multidataset.md)：AISHELL-1、ST-CMDS、MAGICDATA-READ、THCHS-30、LibriSpeech。
- [完整历史总表](../../server_snapshot/RESULTS_ALL_DATASETS_20260826.md)：保留普通识别、上下文联合、4B 基线、Oracle 与泄漏诊断的原始分类和全部行。
- [原始总表 CSV](../../server_snapshot/RESULTS_ALL_DATASETS_20260826.csv)。
- [历史 Excel 总表](../../server_snapshot/RESULTS_ALL_DATASETS_20260826.xlsx)。

## 归档而不再作为当前成绩展示

- [神经连接器尝试](../archive/connectors/README.md)：Residual、MLP、联合训练、局部音文匹配。
- [CEGER 扩展尝试](../archive/ceger/README.md)：旧训练日志、自由 JSON 复核、三种子声学判决、全类型复核过程和候选上限。
- [全部档案与迁移哈希](../archive/README.md)。

这些档案不代表最终方法；其失败、完整控制与原指标仍公开保留。原始源码/协议快照不移动、不修改，避免破坏历史哈希与复现依赖。

## 模型与协议边界

五数据集结果对应同一基础方法系列，不是同一个 adapter。多数正式 9B 行使用 SenseVoice 10-best；不能把它们全部标成热词增强 CBSense。上下文联合行按原热词来源单列。Error-targeted、OracleLex 或使用参考选择候选的行只属于诊断，不能混入 held-out 主结果。

较晚的 1,334 句 AISHELL contextual dev 尝试不是五数据集模型的 test 成绩。此次整理未训练、推理、重算指标、选新 checkpoint、删失败样本或清理服务器/本地原始文件。
