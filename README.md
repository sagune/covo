# covo

CB-Whisper 后纠错实验计划与流程文档。

阶段策略：前期先用 Whisper large-v3 作为 ASR 参考系统，跑通 N-best 纠错、训练和 verifier；后期再接入 CB-Whisper 的 KWS、热词和重排序证据。

相关仓库：

- CB-Whisper 当前进度：<https://github.com/sagune/cbwhisper>

当前入口：

- [CB-Whisper 受约束后纠错计划](docs/cbwhisper_correction_plan.md)

## 当前可用工具

```bash
python scripts/prepare_chinesehp_sft.py \
  --input ../PostASR-Correction-SLT2024-main/chinesehp_aishell1.jsonl \
  --output data/processed/chinesehp_aishell1_sft.jsonl \
  --nbest-size 5 \
  --output-mode edits
```

这会把 ChineseHP AISHELL-1 的 `reference / nbest / nbest_pinyin` 转成后续 LoRA/SFT 可用的数据格式。样例见：

- [examples/chinesehp_aishell1_sft_sample.jsonl](examples/chinesehp_aishell1_sft_sample.jsonl)

本地检查：

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src scripts tests
```
