#!/usr/bin/env bash
# Integration-test the sequencer's MORNING REPORT block against a TEMP output dir.
#
# Why: every analysis command in that block has been tested individually on real files,
# but the shell glue (variable expansion, quoting, the `if [ -s ... ]` guards) has not,
# and that report is the only thing anyone reads in the morning.
#
# SAFETY: it must NOT write to the real $OUT/train_aishell_v1, because phase 4's `infer`
# skips any prediction file that already exists - a fake stcmds_final.predictions.jsonl
# there would make the real trained-adapter evaluation silently never run.  So $OUT is
# redirected to a temp tree of symlinks and the real directory is never touched.
set -uo pipefail
WS=/root/autodl-tmp
R="$WS/.dsh_checks/rerank"
PY="$WS/great/bin/python"
TMP=/tmp/mrtest/train_aishell_v1
SEQ="$WS/.dsh_checks/run_night_sequencer.sh"
BLOCK=/tmp/mrtest/report_block.sh
OUTLOG=/tmp/mrtest/report_stdout.txt

rm -rf /tmp/mrtest; mkdir -p "$TMP"
A="$WS/datasets/aishell/data_aishell_sensevoice/hotword/dev"
T="$WS/datasets/thchs30/cb_sensevoice_error_hotwords_20260811/hotword/test"
S="$WS/datasets/stcmds/cb_sensevoice_heldout/hotword/test"
TRLOG="$R/train_run_v2.log"
CMP="$R/compare_old_vs_new.txt"

# stand-ins so every guard in the block fires
ln -sf "$R/e2eSTCMDS_VA.predictions.jsonl"  "$TMP/stcmds_final.predictions.jsonl"
ln -sf "$R/dev_reranked.predictions.jsonl" "$TMP/dev_final.predictions.jsonl"
ln -sf "$R/dev_reranked.predictions.jsonl" "$TMP/dev_OLD.predictions.jsonl"
ln -sf "$R/e2eTHCHS.predictions.jsonl"     "$TMP/thchs_final.predictions.jsonl"
ln -sf "$R/e2eTHCHS.predictions.jsonl"     "$TMP/thchs_OLD.predictions.jsonl"
echo "temp OUT tree:"; ls -l "$TMP" | tail -6

# lift the REAL report block out of the sequencer, overriding only $OUT and the two
# filesystem side effects: the block runs to END OF FILE, so it also contains
# `touch $R/SEQUENCER_DONE` - and running this test once created the real completion
# marker, which made the watchdog exit immediately.  Both redirects are required.
awk '/6\. morning report/{f=1} f' "$SEQ" \
  | sed "s|\"\$R/MORNING_REPORT.txt\"|\"/tmp/mrtest/MORNING_REPORT.txt\"|" \
  | sed "s|\"\$R/SEQUENCER_DONE\"|\"/tmp/mrtest/SEQUENCER_DONE\"|" > /tmp/mrtest/body.sh
cat > "$BLOCK" <<EOF
#!/usr/bin/env bash
set -uo pipefail
WS=$WS
R=$R
PY=$PY
OUT=$TMP
A=$A
T=$T
S=$S
TRLOG=$TRLOG
CMP=$CMP
LOG=$R/sequencer.log
TARGET=4.50
BASELINE=4.9356
say() { echo "[test] \$*"; }
EOF
cat /tmp/mrtest/body.sh >> "$BLOCK"
echo "block lines: $(wc -l < "$BLOCK")"

echo "=== running the report block ==="
bash "$BLOCK" > "$OUTLOG" 2>&1
echo "rc=$?"
echo
echo "=== errors in stdout, if any ==="
grep -aiE "traceback|error|command not found|unbound|no such file" "$OUTLOG" | grep -v "editing-ability\|ERROR " | head -10 || echo "  (none)"
echo
echo "=== MORNING_REPORT.txt: section headers produced ==="
grep -nE "^#|^==|VERDICT|delta vs|TARGET|CER=" /tmp/mrtest/MORNING_REPORT.txt 2>/dev/null | head -30
echo
echo "=== size ==="
wc -l /tmp/mrtest/MORNING_REPORT.txt 2>/dev/null || echo "REPORT NOT WRITTEN"
echo
echo "=== confirm the real OUT was NOT touched ==="
ls -l "$R/train_aishell_v1"/stcmds_final.predictions.jsonl 2>/dev/null && echo "!!! REAL FILE EXISTS - PROBLEM" || echo "OK: no real stcmds_final.predictions.jsonl"
