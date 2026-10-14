#!/usr/bin/env bash
# Full inventory of this session's experiment artifacts and their key result lines.
echo "########## 1. all check dirs ##########"
ls -d /root/autodl-tmp/.dsh_checks/*/ 2>/dev/null

echo
echo "########## 2. every log file (size, mtime, lines) ##########"
find /root/autodl-tmp/.dsh_checks -maxdepth 2 -name "*.log" -printf "%TY-%Tm-%Td %TH:%TM %10s %p\n" 2>/dev/null | sort

echo
echo "########## 3. completion markers ##########"
find /root/autodl-tmp/.dsh_checks -maxdepth 2 -name "*_DONE" -printf "%TH:%TM %p\n" 2>/dev/null | sort

echo
echo "########## 4. prediction / evidence products (size, mtime) ##########"
find /root/autodl-tmp/.dsh_checks -maxdepth 2 \( -name "*.predictions.jsonl" -o -name "*.evidence.jsonl" -o -name "*reranked*.jsonl" -o -name "*.messages.jsonl" \) -printf "%TH:%TM %10s %p\n" 2>/dev/null | sort -k3 -n -r | head -50

echo
echo "########## 5. git log this session ##########"
cd /root/autodl-tmp && git log --oneline -20
