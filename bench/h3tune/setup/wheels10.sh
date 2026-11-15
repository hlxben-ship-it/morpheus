PY=/root/venv10/bin/python
$PY -m pip install --no-deps "https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch213rocm100py312-cp312-cp312-linux_x86_64.whl" 2>&1 | tail -2
$PY -m pip install --no-deps "https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.3/wgpkernels-1.4.3%2Brocm100-py3-none-linux_x86_64.whl" 2>&1 | tail -2
cd /root/wangp
$PY - <<'PY'
import torch
try:
    import llamacpp_gguf_cuda as g; print("gguf module ok", [n for n in dir(g) if not n.startswith("_")][:30])
except Exception as e: print("gguf import failed:", type(e).__name__, str(e)[:400])
try:
    from wgpkernels import int8; print("wgpkernels lib", int8.default_library_path(), int8.default_library_path().is_file())
    int8.load_library(); print("selftest:", int8.selftest())
except Exception as e: print("wgpkernels failed:", type(e).__name__, str(e)[:400])
PY
