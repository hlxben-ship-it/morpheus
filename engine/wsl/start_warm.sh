#!/bin/bash
# Started by engine/wsl_runner.py: wsl.exe -d Ubuntu-24.04 -u root -- bash <this> <keep_warm_seconds> <preload_mb> [stack]
# Two Python stacks, same WanGP copy /root/wangp (Glia seeds spike_wsl2_rocm_triton, challenge_spike_h3_tuning):
#   7  (default) /root/venv: torch 2.10 on the system ROCm 7.2.4 + librocdxg. LTX and upscales.
#   10 /root/venv10: torch 2.13 + triton 3.8 on the ROCm 10 pip wheels (they carry their own runtime). H3: ~15-20 % faster per step.
PY=/root/venv/bin/python
if [ "$3" = "10" ] && [ -x /root/venv10/bin/python ]; then
  PY=/root/venv10/bin/python
  unset ROCM_HOME ROCM_PATH LD_LIBRARY_PATH
  export PATH=/root/venv10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
else
  export ROCM_HOME=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4
  export PATH=/opt/rocm-7.2.4/bin:/root/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
fi
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# VRAM preload: 0. Until 2026-11-15 the 14000 passed here never reached WanGP (warm_worker.py imported wgp before init). Now that it
# does: it buys no speed (the step is compute-bound) and leaves LTX too little VRAM ("unsufficient VRAM" at 512x896). $2 is ignored.
export WARM_PRELOAD=0
pkill -f warm_worker.py 2>/dev/null   # never two: each holds ~23 GB
sync; echo 3 > /proc/sys/vm/drop_caches
# keep the page cache flat while the model loads (else the VM holds the file twice and Windows runs out of RAM)
( for i in $(seq 1 40); do sleep 5; sync; echo 1 > /proc/sys/vm/drop_caches; done ) &
cd /root/wangp
$PY "$(dirname "$0")/warm_worker.py" --keep-warm "${1:-600}"
code=$?
# hand the memory back to Windows at once (else vmmemWSL keeps ~7 GB for a long while)
sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory
exit $code
