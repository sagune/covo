#!/usr/bin/env bash
# One-shot health check for the backend training run.  Prints everything needed to
# judge "is it still healthy?" in one screen.
R=/root/autodl-tmp/.dsh_checks/rerank
OUT="$R/train_aishell_v1"
L="$R/train_out.log"
echo "===== $(date -Is) ====="
echo "-- markers --"
for m in P2_PHASE0_DONE P2_PHASE1_DONE P2_PHASE2_DONE P2_PHASE3_DONE P2_PHASE4_DONE TRAIN_PLAN2_DONE TRAIN_FAILED; do
  [ -f "$R/$m" ] && echo "  SET   $m"
done
echo "-- process --"
pgrep -af "train_lora_sft" | cut -c1-70 || echo "  (no trainer process)"
pgrep -af "run_train2" | cut -c1-50 || echo "  (no orchestrator)"
echo "-- progress --"
grep -aoE "[0-9]+/2163 \[[0-9:]+<[0-9:]+" "$L" 2>/dev/null | tail -1 || echo "  (no progress line yet)"
echo "-- loss trend (every 20 steps; first vs last 5) --"
grep -aoE "\"loss\": [0-9.]+, \"grad_norm\": [0-9.]+, \"learning_rate\": [0-9.e-]+, \"epoch\": [0-9.]+" "$L" 2>/dev/null | head -5
echo "  ..."
grep -aoE "\"loss\": [0-9.]+, \"grad_norm\": [0-9.]+, \"learning_rate\": [0-9.e-]+, \"epoch\": [0-9.]+" "$L" 2>/dev/null | tail -5
echo "-- loss statistics --"
grep -aoE "\"loss\": [0-9.]+" "$L" 2>/dev/null | grep -oE "[0-9.]+" | awk '
  {n++; s+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1; a[n]=$1}
  END{if(n>0) printf "  n=%d  mean=%.4f  min=%.4f  max=%.4f  last=%.4f\n", n, s/n, min, max, a[n]; else print "  (none yet)"}'
echo "-- checkpoints --"
ls -d "$OUT"/final/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -3 || echo "  (none yet)"
echo "-- gpu --"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "-- errors --"
grep -aiE "FATAL|Traceback|CUDA out of memory|nan|inf" "$L" 2>/dev/null | tail -4 || echo "  (none)"
