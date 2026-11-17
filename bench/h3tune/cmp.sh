#!/bin/bash
# cmp.sh <out.jpg> <run>...   one row per run: frames 20, 45, 70 side by side (for judging a numerics change by eye)
O=$1; shift; D=C:/dev/morpheus/output/bench/h3tune; i=0; IN=(); F=""
for n in "$@"; do IN+=(-i $D/$n/$n.mp4); F+="[$i:v]select='eq(n,20)+eq(n,45)+eq(n,70)',scale=640:-1,tile=3x1[r$i];"; L+="[r$i]"; i=$((i+1)); done
ffmpeg -y -loglevel error "${IN[@]}" -filter_complex "${F}${L}vstack=inputs=$i" -frames:v 1 -q:v 3 "$O"
