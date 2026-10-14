# 论文流程图

- [全流程](cbsense-ceger-full-flow.png)：离线热词库 + CBSense → CEGER。
- [CBSense](cbsense.png)：热词特征构建、检索与上下文识别。
- [CEGER 第一轮 LoRA](ceger-lora-training.png)：候选增强和基础训练，不展开 CBSense 二次适配。
- [CEGER 双子图](ceger-evidence.png)：证据整理、困难候选增强和证据丢弃。
- [真实例句来源](SAMPLE_PROVENANCE.md)。

内置图像生成，经目视核查。候选/转写示例来自历史结果；热图、随机丢弃、网络图标是机制示意，不是实测中间结果。验证样本作说明，不表示用于训练。

图展示主论文链路，不包括连接器或后续声学复核。9B 仅接收文本证据；LoRA 表示已训练参数，不是在线更新。
