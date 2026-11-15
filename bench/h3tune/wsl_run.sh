#!/bin/bash
# In WSL: wsl_run.sh <variant-name>...   One fresh process per variant, caches dropped before each.
HERE=/mnt/c/dev/morpheus/bench/h3tune
OUT=/mnt/c/dev/morpheus/output/bench/h3tune
for v in "$@"; do
  PY=$(python3 -c "import json;print(json.load(open('$HERE/variants/$v.json')).get('python','/root/venv/bin/python'))")
  if [[ $PY == /root/venv/* ]]; then   # system ROCm 7.2.4 (debs + librocdxg)
    export ROCM_HOME=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
    export PATH=/opt/rocm-7.2.4/bin:$(dirname $PY):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  else                                 # ROCm 10 pip wheels carry their own runtime
    unset ROCM_HOME ROCM_PATH LD_LIBRARY_PATH
    export PATH=$(dirname $PY):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  fi
  export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
  pkill -f warm_worker.py 2>/dev/null; pkill -f h3tune/run.py 2>/dev/null
  sync; echo 3 > /proc/sys/vm/drop_caches
  ( for i in $(seq 1 30); do sleep 5; sync; echo 1 > /proc/sys/vm/drop_caches; done ) & DROP=$!
  mkdir -p $OUT/$v
  timeout ${BENCH_TIMEOUT:-2700} $PY $HERE/run.py $HERE/variants/$v.json > $OUT/$v/log.txt 2>&1
  echo "== $v exit $?"; kill $DROP 2>/dev/null
  grep "## ROW" $OUT/$v/log.txt | cut -c1-700 || tail -5 $OUT/$v/log.txt
done
sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
