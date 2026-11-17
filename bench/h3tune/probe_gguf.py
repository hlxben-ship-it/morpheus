import sys, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); sys.path.insert(0, "/root/wangp"); os.chdir("/root/wangp")
if "--spoof" in sys.argv: import amdspoof  # noqa
import torch
import shared.qtypes.gguf as G
print("probe:", G._probe_gguf_cuda_runtime(force=True)); print("error:", G._GGUF_CUDA_LOAD_ERROR)
try:
    import wgpkernels; print("wgpkernels", getattr(wgpkernels, "__version__", "?"))
    from shared.kernels import int8_backend as I
    print("wgp probe:", I._probe_wgpkernels())
except Exception as e: print("wgpkernels:", type(e).__name__, e)
