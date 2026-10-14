#!/usr/bin/env bash
# Print only the metric-bearing lines of each early log, so the session's full
# experiment record can be reconstructed without the noise.
C=/root/autodl-tmp/.dsh_checks
dump() {
  echo "===== $1 ====="
  grep -aE "CER|recall|destroyed|oracle|DONE|START|written|rows|spurious|drift|edit|acc|match|json|score|sample_|changed|pp\b" "$1" 2>/dev/null \
    | grep -avE "Loading weights|torch_dtype|pynvml|flash-linear|funasr version|Downloading|trust_remote" \
    | tail -"${2:-30}"
  echo
}
echo "################ EARLY SESSION: model/interface baselines ################"
dump "$C/stcmds/infer.log" 20
dump "$C/aishell_selector/infer.log" 20
echo "################ SAMPLING EXPLORATION ################"
dump "$C/sampling/sampling.log" 25
ls -l "$C/thinking/" 2>/dev/null
echo "################ thinking dir logs ################"
for f in "$C/thinking/"*.log "$C/thinking/"*.txt "$C/thinking/"*.json; do [ -f "$f" ] && { echo "--- $f ---"; tail -20 "$f"; }; done 2>/dev/null
