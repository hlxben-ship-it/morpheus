#!/bin/bash
# Started by engine/wsl_runner.py: wsl.exe -d Ubuntu-24.04 -u root -- bash <this> <keep_warm_seconds> <preload_mb>
# Env of the WSL render setup (see Glia seed spike_wsl2_rocm_triton): ROCm 7.2.4 + librocdxg, venv /root/venv, WanGP copy /root/wangp.
export ROCM_HOME=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
export PATH=/opt/rocm-7.2.4/bin:/root/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export WARM_PRELOAD="${2:-14000}"
pkill -f warm_worker.py 2>/dev/null   # never two: each holds ~23 GB
sync; echo 3 > /proc/sys/vm/drop_caches
# keep the page cache flat while the model loads (else the VM holds the file twice and Windows runs out of RAM)
( for i in $(seq 1 40); do sleep 5; sync; echo 1 > /proc/sys/vm/drop_caches; done ) &
cd /root/wangp
exec /root/venv/bin/python "$(dirname "$0")/warm_worker.py" --keep-warm "${1:-600}"
