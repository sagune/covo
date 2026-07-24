# AISHELL Experiment Ledger

Updated: 2026-07-22

This ledger consolidates the AISHELL experiments that produced interpretable
metrics. It deliberately separates evaluation scopes and normalizers:

- **HW808 local**: 808-utterance AISHELL-NE hotword subset, mean per-utterance
  CB CER and mention-level Entity Recall.
- **HW808 COVO**: the same 808 rows under the COVO/OpenCC correction evaluator.
- **Recall@400/R1**: designated 400-hotword and 226 difficult-hotword protocol.
- **Full7176**: complete official AISHELL-1 test split, corpus CER unless marked
  as mean CER.
- **Full** means every row in that scope was evaluated. **Pilot/smoke** results
  are diagnostic and must not be compared as final numbers.

## Standalone KWS

| Scope | Representation/model | Threshold | Precision | Recall | F1 | Status |
|---|---|---:|---:|---:|---:|---|
| Validation, full | Previous large-v2 KWS | learned | 91.33% | 82.18% | 86.51% | Historical accepted checkpoint |
| HW808, full | Stale 1024-d large-v3-style KWS | 0.982 | 92.13% | 78.24% | 84.62% | Invalid v3 representation diagnosis |
| Validation, full | True large-v3 TCResNet | 0.974 | 93.34% | 82.51% | 87.59% | Best v3 validation |
| HW808, full | True large-v3 TCResNet | 0.974 | 91.78% | 80.57% | 85.81% | Independent test |
| HW808, full | Original ResNet1 natural KWS | 0.500 | 80.23% | 87.47% | 83.70% | Original-paper code, minimally adapted |

The separate top-k diagnosis on 942 true mentions gives Recall@1/3/6/12 of
80.36%/95.22%/96.92%/98.62%. The CB rescore pool covers 96.07% and the prompt
set 93.21%, so final hotword recall is mainly candidate-generation limited.

## CB-Whisper On HW808

All rows below are full 808-row runs unless explicitly marked smoke/offline.
Metrics use the HW808 local evaluator.

| Date | Method | Entity Recall | CER | Hotword CER | WER | Decision |
|---|---|---:|---:|---:|---:|---|
| 04-27 | Large-v2 baseline, n-best 8, pinyin/consensus rerank | 84.75% | 8.22% | 10.87% | 51.61% | Baseline |
| 04-30 | Broader prompt/rescore keyword budget | 84.64% | 8.78% | 11.22% | 50.99% | Rejected |
| 04-30 | Static rerank weight shift | 84.75% | 8.22% | 10.87% | 51.61% | No effect |
| 04-30 | Broad decode-time hotword token bias | 84.86% | 8.45% | 10.77% | 52.72% | Recall/CER trade-off |
| 04-30 | Decoder-evidence restart bias | 84.97% | 8.42% | 11.26% | 52.48% | Rejected |
| 04-30 | ASR plausibility guard | 79.45% | 8.82% | 13.70% | 55.45% | Rejected |
| 04-30 | Overlong-candidate penalty | 84.75% | 8.22% | 10.87% | 51.61% | No effect |
| 05-01 | Delete nested short hotwords | 84.20% | 8.24% | 11.08% | 51.61% | Rejected |
| 05-01 | Promote nested long hotwords | 84.86% | 8.20% | 10.81% | 51.49% | Accepted v2 setting |
| 05-01 | Length-weighted exact coverage | 84.86% | 8.20% | 10.81% | 51.49% | No effect |
| 05-01 | Prefix-only bias, weight 0.18 | 84.75% | 8.16% | 10.77% | 51.61% | CER gain, recall loss |
| 05-02 | Prefix-only bias, weight 0.08 | 84.75% | 8.19% | 10.84% | 51.36% | Rejected |
| 05-02 | Candidate-supported exact repair | 84.86% | 8.20% | 10.81% | 51.49% | No effect |
| 05-02 | Broader nested promotion | 84.86% | 8.23% | 10.81% | 51.49% | Rejected |
| 05-02 | Candidate generation cap 24 | 85.52% | 8.59% | 10.28% | 52.23% | Recall gain, CER loss |
| 05-02 | Cap 24 plus beam-rank prior | 85.52% | 8.59% | 10.28% | 52.23% | No effect |
| 05-03 | Confidence-dependent candidate budget | 84.64% | 8.44% | 11.08% | 51.86% | Rejected |
| 05-03 | Rescore keyword filtering | 84.31% | 8.27% | 11.05% | 51.61% | Rejected |
| 05-03 | Softmax-calibrated KWS priors | 84.86% | 8.20% | 10.81% | 51.49% | No effect |
| 05-03 | Gated phonetic-only rescoring | 84.86% | 8.20% | 10.81% | 51.49% | No effect |
| 05-03 | Specificity-aware keyword ordering | 84.75% | 8.51% | 10.63% | 51.11% | Rejected |
| 05-03 | Phonetic n-best consensus rerank | 84.86% | 8.20% | 10.81% | 51.49% | No effect |
| 05-03 | Natural-language prompt | 83.20% | 7.86% | 11.85% | 51.36% | CER gain, recall loss |
| 05-03 | Individual bracket prompts | 84.53% | 8.64% | 12.23% | 51.73% | Rejected |
| 05-03 | Direct large-v3 swap, broken language control | 12.04% | 375.86% | 84.52% | 94.31% | Invalid diagnosis |
| 05-03 | Fixed v3 language tokens, 20-row smoke | 50.00% | 7.67% | 23.61% | 55.00% | Smoke only |
| 05-03 | Fixed-token large-v3 full run | 47.96% | 11.80% | 29.94% | 70.67% | Rejected |
| 05-12 | v3 ASR plus stale 1024-d v3-style KWS | 47.96% | 11.80% | 29.94% | 70.67% | Rejected |
| 05-14 | True v3 ASR/KWS/1280-d database | 54.81% | 11.32% | 27.29% | 66.83% | Candidate bottleneck |
| 05-14 | True v3 plus real-length KWS trimming | 72.15% | 11.96% | 23.56% | 58.42% | KWS bug fixed |
| 05-14 | True v3 plus prompt-aware no-repeat | **92.38%** | **6.61%** | **4.67%** | **46.41%** | Best standalone CB-Whisper |
| 05-28 | No-prompt clean v3 fallback | 47.85% | 11.45% | 29.73% | 70.05% | Rejected |
| 05-28 | Offline character MBR over same n-best | 92.89% | 6.45% | 4.94% | - | Offline diagnostic |
| 05-28 | Dev-trained linear n-best selector | 92.58% | 6.43% | 5.08% | - | Offline diagnostic |
| 06-26 | Integrated clean 10-best generation | 92.71% | 7.10% | 4.53% | 46.54% | Better evidence, worse top1 |

Candidate-generation diagnostics:

| Scope | Candidate construction | Avg unique | Exact ref in pool | Oracle CER | Status |
|---|---|---:|---:|---:|---|
| HW808, full | Original CB evidence | 6.10 | 576/808 | 3.29% mean | Baseline pool |
| HW808, full | Multi-prompt Whisper only | 3.93 | - | 3.84% mean | Complement only |
| HW808, full | CB plus multi-prompt union | 8.01 | 606/808 | 2.71% mean | Strong oracle |
| HW808, full | Integrated CB 10-best | 8.82 | 612/808 | 2.864% corpus | Cleaner integrated pool |
| HW808, full | Two-pass targeted supplement | 10.00 | - | 2.864% corpus | Unclean ceiling |
| HW808, full | Strict-clean supplement | 9.91 | - | 2.864% corpus | Preferred rich pool |

## COVO On HW808

These rows use the COVO correction evaluator unless “local” is stated. They
must not be mixed directly with the CB-Whisper local table.

| Training/inference route | CER | Hotword recall | Improved/worsened/unchanged | Status |
|---|---:|---:|---:|---|
| CB evidence input | 10.043% | 90.28% | - | Input baseline, different normalizer |
| Real-dev600 compact SFT160 | 5.650% | 85.41% | 296/96/416 | Recall damaged |
| Preserve2 SFT120 | 4.400% | 87.85% | 309/62/437 | Improved preservation |
| Full AISHELL train no-op SFT | 4.354% | 90.39% | 300/41/467 | Historical no-gate best |
| Full real-error/no-op epoch | 4.680% | 83.16% | 312/95/401 | Over-aggressive, rejected |
| Hotword-preservation DPO30 | 4.501% | 92.27% | - | Recall-priority trade-off |
| Full-pass preservation DPO | 5.658% | 90.89% | 244/58/506 | Rejected |
| Clean 9.9-best, old adapter | 4.594% | 89.07% | 310/48/450 | Rich evidence underused |
| Reliability labels, no training | 4.494% | 90.34% | 314/43/451 | Better prompt structure |
| Compact reliability prompt | 4.408% | 91.083% | - | Strong zero-training variant |
| Reliability SFT40/80 | 4.362% | 90.87%/90.76% | - | Near old CER best |
| Actual-error mild SFT40 | **4.284%** | 90.764% | - | CER-best adapter |
| Targeted CB supplement + expanded adapter | **4.284%** | **91.083%** | - | Best combined recall/CER |

For the historical no-op model, the equivalent legacy CB mean CER is 4.297%
and corpus CER is 4.149%. Its n-best-only oracle is 6.178%, while COVO-or-nbest
oracle is 3.097%, demonstrating genuine text rewriting rather than selection.

## SenseVoice Contextual Routes On HW808

| System | Evaluation | CER | Recall | Recall@400 | R1@226 | Status |
|---|---|---:|---:|---:|---:|---|
| Naked SenseVoice | HW808 unified corpus | 10.361% | - | 159/400 | 23/226 | No contextual input |
| Naked SenseVoice + standalone COVO | HW808 unified corpus | 7.730% | - | 173/400 | 29/226 | COVO-only ablation |
| SenseVoice KWS + Whisper-v3 | HW808 local | 7.174% | 92.155% | - | - | Hybrid baseline |
| CB-SenseVoice | HW808 local | 6.202% | 83.204% | - | - | Standalone contextual model |
| CB-SenseVoice | HW808 unified corpus | 7.823% | - | 313/400 | 141/226 | Paper protocol recount |
| CB-SenseVoice + preserve2 COVO | HW808 local | 4.379% | 83.978% | - | - | Independent no-gate pipeline |
| CB-SenseVoice candidate oracle | HW808 local | 3.686% | 85.128% | - | - | Upper bound |
| + frame position/pronunciation adapter | HW808 local | 6.086% | 83.757% | - | - | 131k trainable parameters |
| + frame position/pronunciation adapter | HW808 unified corpus | 7.691% | - | 318/400 | 146/226 | Candidate gain +5 |
| + phrase cross-attention adapter | HW808 local | **5.093%** | **83.978%** | - | - | 729k parameters; CER main model |
| + phrase cross-attention adapter | HW808 unified corpus | **4.827%** | - | **323/400** | **150/226** | Candidate/top1 gain +10 |
| + acoustic phrase evidence (w=14) | HW808 unified corpus | **3.795%** | 92.486% local | **366/400** | **192/226** | Standalone CB-SenseVoice main |
| + preserve2 COVO | HW808 unified corpus | **3.027%** | - | 357/400 | 184/226 | Lowest CER; recall below 90% |
| + hotword-preserve DPO-30 COVO | HW808 unified corpus | **3.135%** | - | **362/400** | **189/226** | Joint main; no gate |

## Complete AISHELL-1 Test (7,176 Utterances)

| System | Corpus CER | Mean CER | Edits | Exact | Status |
|---|---:|---:|---:|---:|---|
| Whisper-large-v3 greedy | 9.023% | 9.060% | - | - | Full, independent |
| Whisper-large-v3 beam5 | 8.730% | 8.735% | - | - | Full; unique n-best=1.0 |
| Whisper-large-v3 sampled n-best top1 | 9.012% | 9.050% | - | - | Full; oracle mean 6.471% |
| SenseVoiceSmall | 6.291% | 6.389% | 6591 | 4454 | Full, independent |
| SenseVoiceSmall + original COVO | 5.705% | 5.820% | 5977 | 4769 | Full, independent |
| **SenseVoiceSmall + AISHELL COVO** | **3.765%** | **3.955%** | **3944** | **4982** | **Full, independent** |
| Routed SenseVoice / CB-SenseVoice w14 | 5.489% | - | 5751 | 4722 | 808 contextual rows |
| **Routed + hotword-preserve COVO** | **3.206%** | - | **3359** | **5235** | **Joint main; no confidence gate** |
| Routed + preserve2 COVO | **3.193%** | - | **3345** | **5247** | CER ablation; contextual recall <90% |

## Comparison With Published Work

Only rows with matching or closely related protocols should support paper
claims. “Informative” means the dataset/upstream ASR or normalization differs.

| Year | Paper/system | Dataset/protocol | Reported result | Our nearest result | Comparability |
|---:|---|---|---|---|---|
| 2023 | SeACo-Paraformer | Test-Aishell1-NE, 808/400 | CER 2.48%, recall 90% | CB-Whisper+COVO 4.284%, local recall 91.083% | Same subset; recount/normalizer required |
| 2023 | SeACo-Paraformer + ASF | Test-Aishell1-NE | CER 2.27%, recall 94% | 4.284%, 91.083% | We do not beat it |
| 2024 | Original CB-Whisper | Aishell-test | MER 8.6%, recall 82.4% | Standalone v3: CER 6.61%, recall 92.38% | Closest direct baseline |
| 2024 | Efficient Text Augmentation | Test-Aishell1-NE | Best CER 4.50% | COVO combined 4.284% | Same nominal subset, verify normalization |
| 2024 | Confidence Homophone Detector | Test-Aishell1-Middle | CER 6.46%, recall 85.5%, F1 90.3% | 4.284%, 91.083% local recall | Same nominal subset, metric details differ |
| 2025 | GLCLAP | AISHELL-1 test NT | Retrieval F1 96.96% | KWS ranking Recall@12 98.62% | Retrieval only; no endpoint CER |
| 2025 | Adaptive Context Biasing | AISHELL contextual subset | Dev/test CER 5.56%/6.01% | 4.284% HW808 COVO | Informative, upstream/protocol differ |
| 2025 | Generative Annotation NEC | AISHELL entity correction | CER 9.85%, NE-CER 7.41%, recall 87.31% | Full7176 entity COVO: 3.773%, NE-CER 8.278%, recall 82.85% | Better overall CER, worse entity metrics |
| 2025 | PARCO | AISHELL-1, 1000 distractors | CER 4.22% | Full7176 SenseVoice+COVO 3.765% | Informative, contextual setup differs |
| 2026 | DBA-wav2vec 2.0 | AISHELL-1 ASR | CER 6.97% | Full7176 3.765% | Ordinary-ASR comparison |
| 2026 | Streaming decoder-only LLM ASR | AISHELL-1 ASR | CER 5.10% | Full7176 3.765% | Different streaming/model setting |
| 2026 | RASTAR-8B | AISHELL-1 NEC | CER 4.21%, NE-CER 6.21%, recall 89.33% | 3.773%, 8.278%, 82.85% | Better overall; RASTAR better on entities |
| 2026 | PAC | AISHELL-1 contextual biasing | 53.8% relative CER/WER and 60.5% relative B-WER reduction | Phrase adapter pending | Paper reports relative metrics |
| 2026 | Bias-position Speech LLM | Contextual ASR | Up to 8.4% relative B-WER gain from position task | Frame adapter: +5 Recall@400 hits | Different backbone/dataset |

Primary references:

- SeACo-Paraformer: <https://arxiv.org/abs/2308.03266>
- CB-Whisper: <https://aclanthology.org/2024.lrec-main.262/>
- Generative Annotation NEC: <https://aclanthology.org/2025.emnlp-main.1052/>
- PAC: <https://arxiv.org/abs/2509.12647>
- RASTAR: <https://arxiv.org/abs/2602.12287>
- Bias-word position prediction: <https://arxiv.org/abs/2604.12398>

## Current Defensible Claims

1. On HW808, the true-v3 CB-Whisper implementation substantially improves the
   original CB-Whisper endpoint: 6.61% vs 8.6% CER and 92.38% vs 82.4% recall.
2. The full no-gate CB-SenseVoice w14 + hotword-preserve COVO route reaches
   3.135% corpus CER and 362/400 designated-hotword recall. The unconstrained
   preserve2 COVO variant reaches 3.027% CER but falls to 357/400 recall.
3. On complete AISHELL-1, SenseVoice+COVO reaches 3.765% corpus CER and beats
   the listed 2026 systems on overall CER, while RASTAR remains stronger on
   named-entity CER and recall.
4. Test-derived lexicons, direct test-label DPO, candidate oracles, and smoke
   subsets are diagnostics only and are excluded from paper claims.
