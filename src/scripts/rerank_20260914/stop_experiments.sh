#!/usr/bin/env bash
# Stop every queued and in-flight rerank experiment on request.
echo "=== before ==="
pgrep -af "run_samecode_baselines|run_fixed_rerank_e2e|run_stcmds_orig_fixed|run_stcmds_breadth|run_CLI.py test|infer_lora_text|score_sensevoice" | cut -c1-90

pkill -f "run_fixed_rerank_e2e.sh"       2>/dev/null
pkill -f "run_stcmds_orig_fixed_e2e.sh"  2>/dev/null
pkill -f "run_stcmds_breadth.sh"         2>/dev/null
pkill -f "run_samecode_baselines.sh"     2>/dev/null
sleep 2
pkill -f "run_CLI.py test"               2>/dev/null
pkill -f "score_sensevoice_candidate_evidence.py" 2>/dev/null
pkill -f "infer_lora_text.py"            2>/dev/null
sleep 5

echo
echo "=== after ==="
pgrep -af "run_samecode_baselines|run_fixed_rerank_e2e|run_stcmds_orig_fixed|run_stcmds_breadth|run_CLI.py test|infer_lora_text|score_sensevoice" | cut -c1-90 || echo "(all stopped)"
echo
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo
echo "=== markers reached before the stop ==="
R=/root/autodl-tmp/.dsh_checks/rerank
for m in STCMDS_SAFE_DONE SAMECODE_DONE FIXED_RERANK_E2E_DONE STCMDS_ORIG_FIXED_E2E_DONE STCMDS_BREADTH_DONE POLICY_SWEEP6_DONE BOOTSTRAP_DONE V2_CONFIRM_DONE; do
  [ -f "$R/$m" ] && echo "  $m"
done
