# 五个数据集实验结果总表

更新时间：2026-08-26

中文数据集使用 corpus CER；LibriSpeech 使用标准词级 WER。`变化` 为“输出错误率 − 输入错误率”，负值表示改善。不同输入解码、候选池和 normalization 的行只应在行内比较，不能仅按输出数字横向判定模块优劣。

| 语言 | 数据集 / 范围 | 样本数 | 系统 | Adapter / checkpoint | 输入错误率 | 输出错误率 | 变化 (pp) | 相对变化 | Improved / Worsened / Unchanged | 性质 | 补充指标 / 说明 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 中文 | AISHELL-NE 808 Full | 808 | CB-SenseVoice w14 + DPO-30 COVO | DPO-30 4B | CER 10.3609% | CER 3.1354% | -7.2255 | -69.74% | — | Full contextual，联合主结果 | Recall@400 90.50%；R1 Recall@226 83.63% |
| 中文 | AISHELL-1 Full | 7,176 | Routed SenseVoice / CB-SenseVoice w14 + DPO-30 COVO | DPO-30 4B | CER 6.2922% | CER 3.2062% | -3.0860 | -49.04% | — | Full contextual | 808 条有外部上下文，其余 6,368 条走裸 SenseVoice |
| 中文 | AISHELL-1 test | 7,174 | SenseVoice 10-best + Qwen3.5-9B | AISHELL checkpoint-30022 | CER 5.8354% | CER **2.6231%** | -3.2123 | -55.05% | 2,425 / 282 / 4,467 | Full held-out | 2 条空参考被 evaluator 跳过 |
| 中文 | AISHELL-1 Full | 7,176 | Error-targeted CB-SenseVoice + preserve2 COVO | preserve2 4B | CER 2.6345% | CER **1.8336%** | -0.8008 | -30.40% | 890 / 198 / 6,088 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | AISHELL-1 Full | 7,176 | Error-targeted CB-SenseVoice + Qwen3.5-9B | AISHELL checkpoint-30022 | CER 2.6345% | CER 1.8460% | -0.7884 | -29.93% | 1,150 / 504 / 5,522 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | ST-CMDS test | 5,130 | SenseVoice 10-best + Qwen3.5-9B | ST-CMDS checkpoint-34400 | CER 5.6407% | CER **4.4994%** | -1.1413 | -20.23% | 828 / 355 / 3,947 | Full held-out | checkpoint 仅由 dev 选择 |
| 中文 | ST-CMDS Full | 5,130 | Standard-3139 CB-SenseVoice + ChineseHP COVO | historical 4B | CER 5.7921% | CER 5.2757% | -0.5164 | -8.91% | 541 / 316 / 4,273 | Full contextual | 标准上下文词表；与裸 9B 的输入不同 |
| 中文 | ST-CMDS Full | 5,130 | Standard-3139 CB-SenseVoice + Qwen3.5-9B | ST-CMDS checkpoint-34400 | CER 5.7921% | CER **4.9054%** | -0.8867 | -15.31% | 753 / 388 / 3,989 | Full contextual | 标准上下文词表；与裸 9B 的输入不同 |
| 中文 | ST-CMDS Full | 5,130 | Error-targeted CB-SenseVoice + ChineseHP COVO | historical 4B | CER 4.4834% | CER 4.6917% | +0.2083 | +4.65% | 403 / 495 / 4,232 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | ST-CMDS Full | 5,130 | Error-targeted CB-SenseVoice + Qwen3.5-9B | ST-CMDS checkpoint-34400 | CER 4.4834% | CER 4.4994% | +0.0160 | +0.36% | 447 / 462 / 4,221 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | MAGICDATA-READ test | 24,279 | WOITN SenseVoice 10-best + domain COVO | checkpoint-18000 4B | CER 3.7245% | CER 3.4673% | -0.2572 | -6.91% | 1,755 / 1,264 / 21,260 | Full held-out | 旧 domain-COVO 协议 |
| 中文 | MAGICDATA-READ test | 24,279 | WOITN SenseVoice 10-best + original-style COVO | checkpoint-20000 4B | CER 3.7245% | CER 2.4903% | -1.2343 | -33.14% | 3,438 / 1,309 / 19,532 | Full held-out | 2026-08-07 完整测试 |
| 中文 | MAGICDATA-READ test | 24,279 | WOITN SenseVoice 10-best + Qwen3.5-9B | MAGICDATA checkpoint-39185 | CER 3.7245% | CER **2.4334%** | -1.2911 | -34.67% | 3,452 / 1,250 / 19,577 | Full held-out | checkpoint 仅由 dev 选择；正式最佳 |
| 中文 | MAGICDATA-READ test | 24,279 | OracleLex-6000 CB-SenseVoice + Qwen3.5-9B | MAGICDATA checkpoint-39185 | CER 3.7003% | CER 2.7416% | -0.9586 | -25.91% | 3,080 / 1,388 / 19,811 | Leaked diagnostic | 6,000 词上下文词表来自 test 参考 |
| 中文 | THCHS-30 test | 2,495 | SenseVoice 10-best + acoustic-listwise COVO | AISHELL acoustic 4B | CER 5.1664% | CER **4.2175%** | -0.9489 | -18.37% | 792 / 316 / 1,387 | Full cross-domain | 无 THCHS-30 训练 |
| 中文 | THCHS-30 test | 2,495 | SenseVoice 10-best + original-style COVO | MAGICDATA checkpoint-20000 4B | CER 5.1664% | CER 4.7536% | -0.4129 | -7.99% | 683 / 502 / 1,310 | Full cross-domain | 无 THCHS-30 训练 |
| 中文 | THCHS-30 test | 2,495 | CB-SenseVoice-style bridge + COVO | MAGICDATA checkpoint-20000 4B | CER 5.1664% | CER 4.7215% | -0.4449 | -8.61% | 647 / 461 / 1,387 | Full cross-domain | bridge 使用同一 SenseVoice 10-best；无 test 调参 |
| 中文 | THCHS-30 test | 2,495 | SenseVoice 10-best + Qwen3.5-9B | MAGICDATA checkpoint-39185 | CER 5.1664% | CER 4.4023% | -0.7641 | -14.79% | 735 / 413 / 1,347 | Full cross-domain | 无 THCHS-30 训练 |
| 中文 | THCHS-30 test | 2,495 | SenseVoice 10-best + Qwen3.5-9B | ST-CMDS checkpoint-34400 | CER 5.1664% | CER 4.3358% | -0.8307 | -16.08% | 662 / 243 / 1,590 | Full cross-domain | 无 THCHS-30 训练 |
| 中文 | THCHS-30 test | 2,495 | SenseVoice 10-best + Qwen3.5-9B | AISHELL checkpoint-30022 | CER 5.1664% | CER **4.3111%** | -0.8553 | -16.56% | 813 / 415 / 1,267 | Full cross-domain | 9B 跨域最佳；无 THCHS-30 训练 |
| 中文 | THCHS-30 test | 2,495 | Error-targeted CB-SenseVoice + COVO | MAGICDATA checkpoint-20000 4B | CER 3.4681% | CER 3.6986% | +0.2305 | +6.65% | 380 / 507 / 1,608 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | THCHS-30 test | 2,495 | Error-targeted CB-SenseVoice + acoustic-listwise COVO | AISHELL acoustic 4B | CER 3.4681% | CER **3.0898%** | -0.3784 | -10.91% | 405 / 185 / 1,905 | Leaked diagnostic | 当前最低观察值，但不是 held-out 主结果 |
| 中文 | THCHS-30 test | 2,495 | Error-targeted CB-SenseVoice + Qwen3.5-9B | MAGICDATA checkpoint-39185 | CER 3.4681% | CER 3.4176% | -0.0505 | -1.46% | 437 / 433 / 1,625 | Leaked diagnostic | 错误定向词表来自 test 参考 / 错误 |
| 中文 | THCHS-30 test | 2,495 | Error-targeted CB-SenseVoice + Qwen3.5-9B | ST-CMDS checkpoint-34400 | CER 3.4681% | CER **3.2931%** | -0.1750 | -5.05% | 329 / 214 / 1,952 | Leaked diagnostic | CB-Sense + 9B 中最佳 |
| 中文 | THCHS-30 test | 2,495 | Error-targeted CB-SenseVoice + Qwen3.5-9B | AISHELL checkpoint-30022 | CER 3.4681% | CER 3.4804% | +0.0123 | +0.36% | 472 / 464 / 1,559 | Leaked diagnostic | 略微退化 |
| 英文 | LibriSpeech dev-clean | 2,703 | SenseVoice 10-best + Qwen3.5-9B | final 2-epoch adapter | WER 3.3804% | WER 2.6047% | -0.7757 | -22.95% | 506 / 213 / 1,984 | Full dev | 字符 CER 1.2998% → 1.0641% |
| 英文 | LibriSpeech dev-other | 2,864 | SenseVoice 10-best + Qwen3.5-9B | final 2-epoch adapter | WER 7.0111% | WER 5.5881% | -1.4230 | -20.30% | 730 / 240 / 1,894 | Full dev | 字符 CER 3.4088% → 2.9476% |
| 英文 | LibriSpeech test-clean | 2,620 | SenseVoice 10-best oracle | — | WER 3.2277% | WER 1.7517% | -1.4760 | -45.73% | — | Oracle | 仅表示候选池上界 |
| 英文 | LibriSpeech test-clean | 2,620 | SenseVoice 10-best + Qwen3.5-9B | final 2-epoch adapter | WER 3.2277% | WER **2.6419%** | -0.5858 | -18.15% | 445 / 208 / 1,967 | Full test | 字符 CER 1.2237% → 1.0258% |
| 英文 | LibriSpeech test-other | 2,939 | SenseVoice 10-best oracle | — | WER 7.2789% | WER 4.9061% | -2.3728 | -32.60% | — | Oracle | 仅表示候选池上界 |
| 英文 | LibriSpeech test-other | 2,939 | SenseVoice 10-best + Qwen3.5-9B | final 2-epoch adapter | WER 7.2789% | WER **5.9206%** | -1.3583 | -18.66% | 761 / 283 / 1,895 | Full test | 字符 CER 3.4059% → 2.9635% |

## 结论

- 正式 held-out 最佳：AISHELL-1 9B 为 2.6231% CER；ST-CMDS 9B 为 4.4994% CER；MAGICDATA 9B 为 2.4334% CER。
- THCHS-30 没有训练：完整跨域最佳为 AISHELL 9B 的 4.3111% CER。
- LibriSpeech 的 9B adapter 在 test-clean / test-other 上分别把 WER 降至 2.6419% / 5.9206%。
- 所有 Error-targeted、OracleLex 和测试参考构词表结果均保留为 Leaked diagnostic，不能作为 held-out 主结果。

## 主要来源

- `/root/autodl-tmp/RESULTS.md`
- `/root/autodl-tmp/src/RESULTS_MAGICDATA_20260811.md`
- `/root/autodl-tmp/src/RESULTS_THCHS30_20260811.md`
- `/root/autodl-tmp/src/RESULTS_THCHS30_9B_20260825.md`
- `/root/autodl-tmp/src/RESULTS_LIBRISPEECH_20260812.md`
- `/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/*/final_summary.json`

LibriSpeech WER 由 2026-08-13 的完整 aligned prediction JSONL 重新统计；归一化为 NFKC、小写、保留字母数字及词内撇号后按空格切词，采用 corpus edit distance / reference words。
