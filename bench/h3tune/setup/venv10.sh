#!/bin/bash
# Side venv for the tuning bench: PyTorch 2.13 + ROCm 10 wheels (self-contained ROCm runtime) + Triton 3.7, next to the working /root/venv.
# Nothing system-wide changes; /root/venv (what Morpheus renders with) stays untouched.
set -x
cd /root
[ -d /root/venv10 ] || cp -a /root/venv /root/venv10
PY=/root/venv10/bin/python
sed -i 's|/root/venv\b|/root/venv10|g' /root/venv10/bin/activate /root/venv10/bin/pip* 2>/dev/null
$PY -m pip --version
$PY -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1100]==2.13.0+rocm10.0.0" "torchvision[device-gfx1100]==0.28.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0" 2>&1 | tail -25
$PY -m pip list 2>/dev/null | grep -i -E "^torch|triton|rocm|amd"
