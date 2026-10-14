# 历史实验结果索引

保存已有报告，发布不进行新评估或改写指标。按各实验原口径阅读，不混用检查点、数据集、提示词。

| 报告 | 范围与性质 |
|---|---|
| [分片连接器](shard-connectors-20260922.md) | 17,294 训练句；Residual/MLP 的 1,334 contextual dev 对照 |
| [连接器问题定位](connector-diagnosis-20260922.md) | 初始化、训练与接口迁移限制 |
| [联合训练](joint-lora-20260923.md) | A/B/C/D；旧最佳 B/text CER 2.9476%、召回 78.30% |
| [局部音文匹配](local-match-20261004.md) | CE/MATCH 与音频控制；不是默认主方法 |
| [冻结 9B 声学复核](acoustic-rereview-20261005.md) | 格式/候选范围问题；不能把失败 CER 当纯声学能力 |
| [三种子证据判决](evidence-decision-20261006.md) | AUDIO/TEXT 与声学控制；不挑最好种子 |
| [全部种子 CSV](evidence-decision-all-seeds.csv) | 配对复验汇总 |
| [全类型候选 Oracle](all-edit-oracle-20261007.md) | 参考已知理论上限，不能用于 dev 调阈值 |
| [全类型复核训练汇报](all-edit-training-20261007.md) | 当时设计和启动记录，不是最终验证报告 |

这些诊断多次使用同一历史 1,334 contextual dev（21,102 字、599 mentions），不是官方 14,326 全 dev 或新的 held-out test；三种子不是三份独立样本。

早期四中文数据集和 LibriSpeech 表在 `server_snapshot/RESULTS_*`，不能与这里的 dev 混合排名。部分旧文档涉及已停用规则、保护或实验模型，需结合日期/协议解读。

逐条预测与 checkpoint 未上传；这里是报告归档，不声称可无资源复算所有指标。
