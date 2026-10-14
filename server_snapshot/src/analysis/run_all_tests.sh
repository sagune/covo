#!/usr/bin/env bash
# Run every CPU-only test in the AC-COVO battery and summarise pass/fail.
#
# These are the checks that caught the checkpoint-format bug, the inputs_embeds slice,
# the right-padding bug, the ambiguous pairing, the exact-marker silent disable and the
# frame-offset question -- all without the GPU, so they can run while training owns it.
# Run after any edit to the coupling code.
set -uo pipefail
cd /root/autodl-tmp || exit 1
export CUDA_VISIBLE_DEVICES=""
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
PY=/root/autodl-tmp/great/bin/python
MODEL=/root/autodl-tmp/cbwhisper_covo_migration_20260609_tar_extracted/models/Qwen3.5-9B
R=.dsh_checks/rerank
pass=0; fail=0; failed=""

run() {
  local name="$1"; shift
  local out
  out=$(timeout 1800 "$PY" "$@" 2>&1)
  # NB the count phrase varies ("8/8 checks pass", "8/8 checkpoint round-trip checks
  # pass", "9/9 integration checks pass"), so allow words between the count and
  # "checks" -- an earlier regex required them adjacent and reported four passing
  # suites as failures.
  if echo "$out" | grep -qE "VERDICT: PASS|[0-9]+/[0-9]+[A-Za-z0-9 -]*checks pass|[0-9]+/[0-9]+ parse cases pass" \
     && ! echo "$out" | grep -qE "VERDICT: FAIL|\[FAIL\]"; then
    printf "  PASS  %-38s %s\n" "$name" "$(echo "$out" | grep -oE '[0-9]+/[0-9]+[A-Za-z0-9 -]*checks pass|[0-9]+/[0-9]+ parse cases pass' | tail -1)"
    pass=$((pass+1))
  else
    printf "  FAIL  %-38s\n" "$name"
    echo "$out" | tail -4 | sed 's/^/          /'
    fail=$((fail+1)); failed="$failed $name"
  fi
}

echo "=== AC-COVO CPU test battery ==="
run "prediction parser"        src/analysis/test_prediction_parser.py
run "constrained decoding"     src/analysis/test_constrained_decoding.py
run "constrained + generate"   src/analysis/test_constrained_generate.py
run "coupling mechanism (GPT2)" src/analysis/test_generate_mechanism.py
run "mechanism (real Qwen3.5)" src/analysis/test_qwen35_mechanism.py
run "checkpoint round-trip"    src/analysis/test_checkpoint_roundtrip.py
run "projector learning rate"  src/analysis/test_acoustic_lr.py
run "freeze-llm contingency"   src/analysis/test_freeze_llm.py
run "alignment + left padding" src/analysis/test_acoustic_alignment.py \
    --rows $R/acoustic_v1/eval_aishell.messages.ac.jsonl \
    --bin $R/ac_aishell_dev.bin --index $R/ac_aishell_dev.jsonl \
    --model "$MODEL" --limit 100
run "interface identity"       src/analysis/test_interface_identity.py \
    --rows $R/acoustic_v1/train_sft_ac.jsonl --bin $R/ac_aishell_train.bin \
    --index $R/ac_aishell_train.jsonl --model "$MODEL" --limit 150
run "frame offset (functional)" src/analysis/test_frame_offset.py --n 2
run "artifact integrity"       src/analysis/test_artifact_integrity.py --n 2
run "allowed-set coverage"     src/analysis/verify_allowed_sets.py \
    --sets $R/stcmds_allowed_sets.jsonl --input $R/stcmds_va_tok_input.jsonl

echo
echo "======================================"
echo "  PASS $pass   FAIL $fail"
[[ -n "$failed" ]] && echo "  failing:$failed"
echo "======================================"
[[ "$fail" -eq 0 ]] || exit 1
