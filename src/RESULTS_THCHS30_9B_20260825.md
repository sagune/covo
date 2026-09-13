# THCHS-30 zero-training 9B evaluation, 2026-08-25

All adapters were selected outside THCHS-30. No THCHS-30 training or checkpoint selection was performed.

## Standalone 9B on SenseVoice 10-best

| Adapter | Baseline CER | Output CER | Improved | Worsened | Unchanged |
|---|---:|---:|---:|---:|---:|
| MAGICDATA 9B ckpt-39185 | 5.1664% | 4.4023% | 735 | 413 | 1347 |
| ST-CMDS 9B ckpt-34400 | 5.1664% | 4.3358% | 662 | 243 | 1590 |
| AISHELL 9B ckpt-30022 | 5.1664% | 4.3111% | 813 | 415 | 1267 |

Classification: **Full cross-domain, no training**.

## CB-SenseVoice + 9B

| Adapter | CB-Sense CER | Output CER | Improved | Worsened | Unchanged |
|---|---:|---:|---:|---:|---:|
| MAGICDATA 9B ckpt-39185 | 3.4681% | 3.4176% | 437 | 433 | 1625 |
| ST-CMDS 9B ckpt-34400 | 3.4681% | 3.2931% | 329 | 214 | 1952 |
| AISHELL 9B ckpt-30022 | 3.4681% | 3.4804% | 472 | 464 | 1559 |

Classification: **Leaked/Diagnostic**. The CB-SenseVoice error-targeted hotword list was derived from THCHS-30 test references/errors; these numbers are an upper-bound diagnostic, not a held-out main result.

Full predictions, per-run summaries, the run manifest, and `final_summary.json` are stored in:

`/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825`
