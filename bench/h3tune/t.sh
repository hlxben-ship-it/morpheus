export ROCM_HOME=/opt/rocm-7.2.4 ROCM_PATH=/opt/rocm-7.2.4 PATH=/opt/rocm-7.2.4/bin:/root/venv/bin:/usr/bin:/bin
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE HSA_ENABLE_DXG_DETECTION=1 PYTHONUNBUFFERED=1
"$@"
