# 旧 CEGER 扩展与复核尝试

本目录不是当前方法的成绩页。原文保留，不修改指标、格式失败计分或失败解释。

| 记录 | 性质 |
|---|---|
| [早期 CoVo 实验日志](experiment-log-early.md) | 早期训练/改写方案的原始日志，非当前 SenseVoice 证据复核结果 |
| [自由 JSON 声学复核](acoustic-rereview-20261005.md) | 旧复核接口及格式/候选范围失败；不是后来的原生 A/B 判决接口 |
| [全类型复核训练启动汇报](all-edit-training-20261007.md) | 当时的设计与内存安全重跑留痕，不能冒充最终验证报告 |
| [三种子声学证据判决](evidence-decision-20261006.md) | 后来的 190 块开发集复核，不是五数据集基础模型 |
| [三种子结果 CSV](evidence-decision-all-seeds.csv) | 上述尝试的全部种子与控制条件 |
| [全类型候选 Oracle](all-edit-oracle-20261007.md) | 参考已知的理论上限，不是模型成绩 |

旧最佳 B/text 初稿属于后续实验的重要基线，保留在[联合训练完整报告](../connectors/joint-lora-20260923.md)中。对应初始化、预测及源码未删除；B/text 并不是独立纯文本训练组。

早期四中文数据集和 LibriSpeech 汇总仍在[原始快照](../../../server_snapshot/RESULTS_ALL_DATASETS_20260826.md)中；其 split、热词来源、模型和口径需按历史说明解读。旧 4B 的 2.777% 不能当成当前 9B 的统一基线。

[保留主线](../../results/README.md)展示五数据集方法系列，不再展示此处尝试为最终模型。较晚全类型复核的[原始摘要](../../../server_snapshot/src/experiments/all_edit_review_20261007_v2/summary.json)及[独立审计](../../../server_snapshot/src/experiments/all_edit_review_20261007_v2/independent_audit.json)仍在原始快照中，用于历史追溯，不修改其完整条件或指标。

原报告中的 `C:/.../outputs/...` 链接是当时本地留痕路径，不表示这些数据已经上传 GitHub。迁移没有修改这些原文链接；GitHub 导航请使用本页与当前结果索引。
