#!/usr/bin/env bash
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
V="$WS/.dsh_checks/verify_guarantees.py"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
$PY "$V" --evidence "$WS/src/logs/hotword_lora_9b_20260908/dev.evidence.jsonl" --ctc-scores "$R/ctc_scores.jsonl" --uttid "$A/uttid" --label "AISHELL dev / original admission"
$PY "$V" --evidence "$R/dev_relax.jsonl" --ctc-scores "$R/relax_ctc_scores.jsonl" --uttid "$A/uttid" --label "AISHELL dev / relaxed admission"
$PY "$V" --evidence "$WS/cbwhisper_covo_migration_20260609_tar_extracted/covo/outputs/thchs30_zero_training_9b_suite_20260825/cb_sensevoice_thchs30_error_hotwords.evidence.jsonl" --ctc-scores "$R/thchs30_ctc_scores.jsonl" --uttid "$T/uttid" --label "THCHS-30 / original admission"
$PY "$V" --evidence "$R/thchs_relax.jsonl" --ctc-scores "$R/thchs_ctc_scores.jsonl" --uttid "$T/uttid" --label "THCHS-30 / relaxed admission"
$PY "$V" --evidence "$WS/.dsh_checks/stcmds/stcmds_evidence.jsonl" --ctc-scores "$R/stcmds_ctc_scores.jsonl" --uttid "$S/uttid" --label "ST-CMDS held-out / original admission"
