#!/bin/bash
# In WSL: sb.sh <script.py> args...   (venv10 env, like wsl_run.sh)
unset ROCM_HOME ROCM_PATH LD_LIBRARY_PATH
export PATH=/root/venv10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1
cd /root/wangp; S=$1; shift; python /mnt/c/dev/morpheus/bench/h3tune/$S "$@" 2>&1 | grep -v "^$"
