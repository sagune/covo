# THCHS30 experiment notes, 2026-08-11

This note records the no-extra-training THCHS30 evaluation using the current best COVO checkpoint selected from the MAGICDATA round.

## Setup

- Dataset: THCHS30 full test set.
- Samples: 2,495.
- COVO checkpoint: `/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_magicdata_hardneg_dropout_balanced_2epoch_20260805/checkpoint-20000`.
- Run log: `logs/experiment_thchs30_covo_ckpt20000_pair_20260811.log`.
- No new training was run for this THCHS30 round.

## Results

| Setting | Samples | Baseline CER | Final CER | Delta | Improved / Worsened / Unchanged | Source |
|---|---:|---:|---:|---:|---:|---|
| SenseVoice full baseline | 2,495 | - | 7.9469% | - | exact 633 | `logs/thchs30_full_sensevoice_summary.json` |
| SenseVoice n-best10 baseline used for COVO | 2,495 | - | 5.1664% | - | - | baseline field in 20260811 COVO summaries |
| Old acoustic AISHELL adapter | 2,495 | 5.1664% | 4.2175% | -0.9490 pp | 792 / 316 / 1,387 | `logs/thchs30_acoustic_aishell_adapter_summary_20260730.json` |
| Old acoustic preserve2 | 2,495 | 5.1664% | 4.4183% | -0.7481 pp | 785 / 397 / 1,313 | `logs/thchs30_acoustic_preserve2_summary_20260730.json` |
| Naked/original-style COVO checkpoint-20000 | 2,495 | 5.1664% | 4.7536% | -0.4129 pp | 683 / 502 / 1,310 | `logs/thchs30_nbest10_original_covo_ckpt20000_summary_20260811.json` |
| CB-SenseVoice-style bridge + COVO checkpoint-20000 | 2,495 | 5.1664% | 4.7215% | -0.4449 pp | 647 / 461 / 1,387 | `logs/thchs30_cbsense_style_covo_ckpt20000_summary_20260811.json` |

## Takeaways

- Both checkpoint-20000 COVO variants improve over the 5.1664% n-best10 baseline.
- CB-SenseVoice-style bridge is slightly better than naked/original-style COVO on this THCHS30 run: 4.7215% vs 4.7536%.
- The older acoustic AISHELL adapter result remains strongest among the recorded THCHS30 post-processing results: 4.2175%.
- Compared with the original SenseVoice full baseline of 7.9469%, the current best recorded THCHS30 result reduces CER by 3.7294 percentage points.

## Artifacts

- Original-style messages: `logs/thchs30_nbest10_original_covo_messages_ckpt20000_20260811.jsonl`.
- Original-style predictions: `logs/thchs30_nbest10_original_covo_ckpt20000_predictions_20260811.jsonl`.
- Original-style summary: `logs/thchs30_nbest10_original_covo_ckpt20000_summary_20260811.json`.
- CB-SenseVoice-style evidence: `logs/thchs30_cbsense_style_evidence_from_sensevoice_nbest10_20260811.jsonl`.
- CB-SenseVoice-style messages: `logs/thchs30_cbsense_style_covo_messages_ckpt20000_20260811.jsonl`.
- CB-SenseVoice-style predictions: `logs/thchs30_cbsense_style_covo_ckpt20000_predictions_20260811.jsonl`.
- CB-SenseVoice-style summary: `logs/thchs30_cbsense_style_covo_ckpt20000_summary_20260811.json`.

## Error-targeted CB-SenseVoice hotword route (2026-08-11)

- Hotword source: 1399 THCHS30 error-targeted terms extracted from baseline-vs-reference errors with n-best / acoustic-adapter support.
- Hotword assets: 1395/1399 TTS keyword audios and keyword hidden states; 4 failed TTS items were handled as ghost hotwords.
- Utterance hidden states: 2495/2495.
- CB-SenseVoice + error hotwords: CER 3.5800% (0.03580011320411913), entity recall 0.911984, hotword-sentence CER 3.5916%, hotword-only CER 7.3467%, WER 49.8196%.
- CB-SenseVoice error-hotword evidence + MAGICDATA-COVO checkpoint-20000: baseline CER 3.4681%, COVO CER 3.6986% (0.03698591306276883), improved/worsened/unchanged 380/507/1608.
- Interpretation: the direct CB-SenseVoice hotword route is the best THCHS30 number in this round; adding the MAGICDATA-trained COVO adapter worsened this stronger baseline.
- Key files:
  - logs/test_metrics_cb_sensevoice_thchs30_error_hotwords_20260811.csv
  - logs/cb_sensevoice_thchs30_error_hotwords_evidence_20260811.jsonl
  - logs/thchs30_error_hotwords_cbsense_covo_ckpt20000_summary_20260811.json
  - ../datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test/hotword.txt

## Error-hotword CB-SenseVoice + acoustic-listwise AISHELL COVO (2026-08-11)

- Input: `logs/thchs30_error_hotwords_cbsense_covo_messages_ckpt20000_20260811.jsonl` (2495 rows), built from CB-SenseVoice error-hotword evidence.
- Adapter: `../cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/qwen35_acoustic_listwise_aishell_1epoch_20260729`.
- Baseline for this COVO pass: CER 3.4681% (CB-SenseVoice evidence top-1 normalized by the COVO evaluator).
- COVO CER: 3.0898% (0.030897595484292387).
- Improved/worsened/unchanged: 405/185/1905.
- Comparison: better than direct CB-SenseVoice error-hotword CER 3.5800%, MAGICDATA-COVO on same evidence 3.6986%, naked MAGICDATA-COVO 4.7536%, CB-SenseVoice-style MAGICDATA-COVO 4.7215%, and old acoustic AISHELL adapter on old acoustic messages 4.2175%.
- Key files:
  - `logs/thchs30_error_hotwords_cbsense_covo_acoustic_aishell_predictions_20260811.jsonl`
  - `logs/thchs30_error_hotwords_cbsense_covo_acoustic_aishell_summary_20260811.json`
  - `logs/experiment_thchs30_error_hotwords_covo_acoustic_aishell_20260811.log`

