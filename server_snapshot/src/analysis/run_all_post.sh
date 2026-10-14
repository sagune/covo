#!/usr/bin/env bash
# ============================================================================
# Everything to run AFTER the AC-COVO pipeline finishes, in the right order.
#
# The pipeline itself already does P4 (per-checkpoint AISHELL-dev evaluation) and
# P5 (transfer with `final`). This adds the four things it does not do, in an order
# where each step can be read before the next is trusted:
#
#   1. pre-flight      -- confirm every input exists and is complete
#   2. final report    -- per-checkpoint dev table, gates, section-15 specificity
#   3. ablation        -- same trained adapter WITH and WITHOUT the acoustic channel,
#                         the only within-model control in the project
#   4. transfer with the checkpoint AISHELL dev actually selected (the pipeline used
#      `final` unconditionally, which deviates from the pre-registered protocol)
#   5. constrained arm -- optional, and only meaningful once 2-4 are read
#
# GPU steps are sequential; nothing here is destructive, and each writes to its own
# *_best / *_ablation / *_constrained files so the pipeline's own outputs survive.
set -uo pipefail
WS=/root/autodl-tmp
R=$WS/.dsh_checks/rerank
PY=/root/autodl-tmp/great/bin/python
LOG=$R/ac_postrun.log
say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

cd "$WS" || exit 1
say "================ AC-COVO POST-RUN START ================"

say "---------- 1. pre-flight ----------"
bash .dsh_checks/ac_preflight.sh 2>&1 | tee -a "$LOG"
if ! bash .dsh_checks/ac_preflight.sh >/dev/null 2>&1; then
  say "PRE-FLIGHT NOT CLEAN -- stopping before anything else; read the output above."
  exit 1
fi

say "---------- 2. final report ----------"
"$PY" src/analysis/ac_final_report.py 2>&1 | tee -a "$LOG"
"$PY" src/analysis/extract_dev_metrics.py --log "$R/acoustic_pipeline.log" \
  --out "$R/dev_metrics_rebuilt.tsv" 2>&1 | tee -a "$LOG"

say "---------- 3. within-model ablation (ST-CMDS) ----------"
"$PY" src/analysis/run_acoustic_ablation.py --corpus stcmds 2>&1 | tee -a "$LOG"

say "---------- 4. transfer using the checkpoint dev actually selected ----------"
"$PY" src/analysis/run_transfer_best.py 2>&1 | tee -a "$LOG"

say "---------- 5. constrained-decoding arm (ST-CMDS first) ----------"
say "run manually if wanted:  python src/analysis/run_constrained_arm.py --corpora stcmds"
say "then, for the section-15 test:  --corpora stcmds,thchs30,aishell_dev"

say "================ AC-COVO POST-RUN DONE ================"
say "read, in order:"
say "  $LOG"
say "  $R/dev_metrics_rebuilt.tsv"
say "  the AC-COVO FINAL REPORT block above"
