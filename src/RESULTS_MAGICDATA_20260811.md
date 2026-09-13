# MAGICDATA experiment notes, closed on 2026-08-11

This note records the MAGICDATA-READ / OpenSLR68 experiment round and the closest external CER results found for comparison.

## Dataset

- Dataset: MAGICDATA Mandarin Chinese Read Speech Corpus, OpenSLR 68.
- Official description: 755 hours, 1080 native Mandarin speakers, train/dev/test split, transcription accuracy above 98%.
- Link: https://www.openslr.org/68/

## Our Results

| Setting | Split | Samples | Baseline CER | Final CER | Delta | Improved / Worsened / Unchanged | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| SenseVoice top-1, n-best10 extraction | test | 24,279 | - | 6.0680% | - | exact top1 16,714; oracle-in-nbest 20,586 | `logs/magicdata_read_sensevoice_nbest10_summary_20260730.json` |
| SenseVoice official-normalized COVO trial | test | 24,279 | 5.1085% | 4.9989% | -0.1096 pp | exact base 17,168; exact pred 17,128; oracle 21,225 | `logs/magicdata_read_official_normalized_summary_20260731.json` |
| Original-style COVO checkpoint-20000 | dev | 11,793 | 4.4229% | 3.1618% | -1.2611 pp | 1,758 / 744 / 9,291 | `logs/magicdata_original_covo_dev_ckpt20000_summary_20260807.json` |
| Original-style COVO checkpoint-20000 | test | 24,279 | 3.7245% | 2.4903% | -1.2343 pp | 3,438 / 1,309 / 19,532 | `logs/magicdata_original_covo_test_ckpt20000_summary_20260807.json` |
| Bare CB-SenseVoice, inferred from COVO bridge baseline | dev | 11,793 | - | 4.4144% | - | evidence-only run completed; full post-test metrics path was skipped due slow/stuck bootstrap eval | `logs/cb_sensevoice_magicdata_dev_evidence_only_20260810.jsonl`, `logs/cb_sensevoice_magicdata_dev_covo_ckpt20000_summary_20260811.json` |
| CB-SenseVoice + COVO checkpoint-20000 | dev | 11,793 | 4.4144% | 3.5505% | -0.8639 pp | 1,473 / 777 / 9,543 | `logs/cb_sensevoice_magicdata_dev_covo_ckpt20000_summary_20260811.json` |

Additional counts:

- CB-SenseVoice evidence-only rows: 11,793.
- CB-SenseVoice + COVO prepared message rows: 11,793.
- CB-SenseVoice + COVO prediction rows: 11,793.

Main takeaways:

- The best result in this round is original-style COVO checkpoint-20000 on MAGICDATA test: 2.4903% CER.
- On MAGICDATA dev, original-style COVO is stronger than CB-SenseVoice + COVO: 3.1618% vs 3.5505%.
- CB-SenseVoice + COVO still improves over its own CB-SenseVoice baseline on dev: 4.4144% to 3.5505%.
- Raw SenseVoice test top-1 is much weaker in the earlier n-best extraction setting: 6.0680% CER; its n-best oracle is 3.2397%.

## External MAGICDATA CER Comparisons

These are the closest papers/results found that use MAGICDATA-READ/OpenSLR68 and report CER-like metrics.

| Paper / Report | Year | MAGICDATA use | Metric | Reported result | Comparison note |
|---|---:|---|---|---|---|
| Pronunciation guided copy and correction model for ASR error correction | 2024 | MagicData dev/test for Chinese ASR error correction | CER / CERR | test: 27.88% no-correction to 15.38% PGCC, CERR 44.84%; dev: 34.05% to 22.77% | Most similar task type, but baseline ASR is intentionally cross-domain and much worse than ours, so absolute CER is not directly comparable. |
| MOSS-Audio Technical Report | 2026 | MAGICDATA-READ test as ASR benchmark | CER | Paraformer 4.67, Fun-ASR-Nano 3.46, SenseVoice-Small 4.93, Kimi-Audio 2.15, Qwen3-Omni 2.47, MOSS-8B 3.20 | Strong external benchmark. Our original-style COVO test CER 2.4903% is close to Qwen3-Omni 2.47 and better than SenseVoice/FunASR/MOSS, but below Kimi-Audio 2.15. |
| Towards inclusive automatic speech recognition | 2024 | MagicData adult test; gender/accent fairness analysis | CER | TDNNF hybrid 3.3%, Conformer E2E 2.9% | Useful ASR baseline reference, but preprocessing/split details may differ. |
| Effects of Speaker Count, Duration, and Accent Diversity on Zero-Shot Accent Robustness in Low-Resource ASR | 2025 | MAGICDATA speakers selected by accent/region for low-resource robustness experiments | CER | Example low-resource OOD accent setting: 60.3 to 39.9 | Same CER family, but not a standard full-test benchmark and not an error-correction setting. |
| Contextualized Token Discrimination for Speech Search Query Correction | 2025 | MagicData used to construct ASR query correction data | ASR CER plus downstream correction metrics | Conformer evaluation CER 5.58; final task mainly uses F1 | Related to ASR post-processing, but final metric is not CER, so it is secondary for comparison. |

References:

- OpenSLR 68: https://www.openslr.org/68/
- PGCC: https://link.springer.com/article/10.1007/s13042-024-02191-7
- MOSS-Audio: https://arxiv.org/html/2606.01802v1
- Towards inclusive ASR: https://www.researchgate.net/publication/374063854_Towards_inclusive_automatic_speech_recognition
- Accent diversity ASR: https://arxiv.org/html/2506.04364v1
- Contextualized Token Discrimination: https://arxiv.org/pdf/2509.04393

## Closure

This MAGICDATA experiment round is closed after the CB-SenseVoice + COVO dev run completed on 2026-08-11. Do not start additional long-running MAGICDATA runs from this round unless a new question is opened.
