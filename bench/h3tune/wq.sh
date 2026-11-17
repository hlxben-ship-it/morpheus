#!/bin/bash
# Bench queue that lives inside WSL (a killed Windows-side wrapper loses nothing).
#   wq.sh add <variant>...   append to the queue, start the worker loop if none runs
#   wq.sh wait               block until the queue is empty and nothing runs
#   wq.sh rows <variant>...  result lines of finished variants
H=/mnt/c/dev/morpheus/bench/h3tune; Q=/root/bench_queue.txt
case $1 in
 add) shift; for v in "$@"; do echo "$v" >> $Q; done
      pgrep -f "wq.sh loop" >/dev/null || { setsid nohup bash $H/wq.sh loop >> /root/bench_queue.log 2>&1 < /dev/null & sleep 2; };;
 loop) while true; do
        while pgrep -f "h3tune/run.py|wsl_run.sh|solbench" >/dev/null; do sleep 10; done
        v=$(head -1 $Q 2>/dev/null); [ -z "$v" ] && break
        sed -i 1d $Q; bash $H/wsl_run.sh "$v"
      done;;
 wait) sleep 5; while pgrep -f "h3tune/run.py|wsl_run.sh|wq.sh loop" >/dev/null || [ -s $Q ]; do sleep 15; done;;
 rows) shift; for v in "$@"; do grep -h "## ROW\|## OUTSIDE\|## TIMING\|## captured\|Traceback\|Error" /mnt/c/dev/morpheus/output/bench/h3tune/$v/log.txt | cut -c1-420; done;;
esac
