# covo

CB-Whisper 后纠错实验计划与流程文档。

阶段策略：前期先用 Whisper large-v3 作为 ASR 参考系统，跑通 N-best 纠错、训练和 verifier；后期再接入 CB-Whisper 的 KWS、热词和重排序证据。

相关仓库：

- CB-Whisper 当前进度：<https://github.com/sagune/cbwhisper>

当前入口：

- [CB-Whisper 受约束后纠错计划](docs/cbwhisper_correction_plan.md)
