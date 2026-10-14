#!/usr/bin/env bash
C=/root/autodl-tmp/.dsh_checks
echo "################ sampling analysis artifacts ################"
ls -l "$C/sampling/" | head -20
for f in "$C/sampling/"*.json "$C/sampling/"*.txt "$C/sampling/"*report* "$C/analyze_sampling.output" "$C/analyze_sampling.out"; do
  [ -f "$f" ] && { echo "--- $f ---"; head -60 "$f"; }
done 2>/dev/null
echo
echo "################ thinking smoke test (260 rows) ################"
PY=/root/autodl-tmp/great/bin/python
$PY - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/.dsh_checks/thinking/smoke_standalone_think.jsonl")
if not p.exists():
    print("(absent)"); raise SystemExit
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print("rows:", len(rows))
print("keys:", sorted(rows[0].keys()))
r = rows[0]
for k in sorted(r.keys()):
    v = r[k]
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    print("  %-18s %s" % (k, s[:300]))
PY
echo
echo "################ mid-session milestones ################"
dump() {
  echo "===== $1 ====="
  grep -aE "START|DONE|CER|recall|destroyed|oracle|rows=|ready|skip|WARNING|ERROR" "$1" 2>/dev/null \
    | grep -avE "Loading weights|torch_dtype|pynvml|flash-linear|funasr|Downloading|trust_remote|DEBUG" | tail -"${2:-14}"
  echo
}
dump "$C/rerank/p5.log" 10
dump "$C/rerank/round2.log" 12
dump "$C/rerank/round2de.log" 12
dump "$C/rerank/pool.log" 12
dump "$C/rerank/round3.log" 14
