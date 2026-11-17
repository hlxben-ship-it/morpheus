#!/bin/bash
# gobg.sh [variant...]: queue variants inside WSL (wq.sh) and wait for the queue to drain. Without arguments: just wait (re-attach).
W() { MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/dev/morpheus/bench/h3tune/wq.sh "$@" 2>&1 | tr -d '\0'; }
[ $# -gt 0 ] && W add "$@"; W wait; [ $# -gt 0 ] && W rows "$@"; tail -8 /c/dev/morpheus/bench/h3tune/results.csv | cut -c1-175
