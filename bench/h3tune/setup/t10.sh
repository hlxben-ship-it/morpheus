# env for the ROCm 10 side venv: the wheels carry their own ROCm, so no /opt/rocm paths
export PATH=/root/venv10/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 PYTHONUNBUFFERED=1
unset ROCM_HOME ROCM_PATH LD_LIBRARY_PATH
"$@"
