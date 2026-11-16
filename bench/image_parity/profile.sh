#!/bin/bash
# Same env as engine/wsl/start_warm.sh (stack 7), then a profiling script. $1 = script, $2 = 1 to enable AOTriton experimental
export ROCM_HOME=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:/root/venv/bin:/usr/bin:/bin
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST PYTHONUNBUFFERED=1
[ "$2" = "1" ] && export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
pkill -f warm_worker.py; sync; echo 3 > /proc/sys/vm/drop_caches
cd /root/wangp && /root/venv/bin/python "$1"
