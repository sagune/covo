#!/usr/bin/env bash
# Wait for the ST-CMDS safe-subset end-to-end and the same-code base comparison.
R=/root/autodl-tmp/.dsh_checks/rerank
for i in $(seq 1 400); do
  [ -f "$R/STCMDS_SAFE_DONE" ] && [ -f "$R/SAMECODE_DONE" ] && break
  sleep 60
done
echo "=== ST-CMDS SAFE SUBSET (original admission + no anchor + rerank) ==="
awk '/ST-CMDS SAFE SUBSET/,0' "$R/stcmds_safe.log" | tail -8
echo
echo "=== ST-CMDS three-way front-end comparison (same-code) ==="
awk '/ST-CMDS held-out: 07-29 evidence/,0' "$R/samecode.log" | head -6
echo
echo "=== THCHS same-code baseline (already computed) ==="
awk '/THCHS-30 same-code baseline/,0' "$R/samecode.log" | head -8
