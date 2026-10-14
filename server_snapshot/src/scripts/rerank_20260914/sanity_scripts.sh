#!/usr/bin/env bash
# Non-GPU sanity pass: every committed rerank script must still parse, and the
# repository must contain no experiment artifacts.
set -uo pipefail
cd /root/autodl-tmp
D=src/scripts/rerank_20260914
PY=/root/autodl-tmp/great/bin/python

echo "=== python syntax ==="
fail=0
for f in "$D"/*.py; do
  if "$PY" -m py_compile "$f" 2>/tmp/_pc.err; then
    printf "  ok   %s\n" "$(basename "$f")"
  else
    printf "  FAIL %s\n" "$(basename "$f")"; cat /tmp/_pc.err; fail=1
  fi
done
echo "=== bash syntax ==="
for f in "$D"/*.sh; do
  if bash -n "$f" 2>/tmp/_bn.err; then
    printf "  ok   %s\n" "$(basename "$f")"
  else
    printf "  FAIL %s\n" "$(basename "$f")"; cat /tmp/_bn.err; fail=1
  fi
done
echo "=== artifact leak check (should print nothing) ==="
git ls-files "$D" | grep -E '\.(jsonl|gz|csv|log|pt|pth|bin|safetensors)$' || echo "  (clean: no artifacts tracked)"
echo "=== tracked file count in $D ==="
git ls-files "$D" | wc -l
echo "=== overall: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ==="
