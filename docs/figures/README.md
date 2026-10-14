# 论文流程图

- [保留的基础流程](cbsense-ceger-full-flow.png)：离线热词库 + CBSense → CEGER，属于已开展多数据集实验的基础方法结构，不含归档的声学复核或神经连接器。
- [CBSense](cbsense.png)：热词特征构建、检索与上下文识别。
- [CEGER 第一轮 LoRA](ceger-lora-training.png)：候选增强和基础训练，不展开 CBSense 二次适配。
- [CEGER 双子图](ceger-evidence.png)：证据整理、困难候选增强和证据丢弃。
- [真实例句来源](SAMPLE_PROVENANCE.md)。

内置图像生成，经目视核查。候选/转写示例来自历史结果；热图、随机丢弃、网络图标是机制示意，不是实测中间结果。验证样本作说明，不表示用于训练。

这些图展示保留的基础结构；归档的 MLP、Residual、局部匹配或后续声学证据复核不混入默认流程。LoRA 表示已训练参数，不是在线更新。某条候选实例不代表所有五数据集用了同一前端/adapter，主线指标需按结果索引的条件阅读。
