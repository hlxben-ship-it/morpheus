"""In WSL: the Triton GGUF dequant kernels against WanGP's PyTorch dequantisation, on random blocks + timing.
    /root/venv/bin/python test_gguf_triton.py [wangp root]"""
import os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa: F401,E401
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/root/wangp"; sys.argv = sys.argv[:1]
os.chdir(ROOT); sys.path.insert(0, ROOT)
import torch, gguf
import shared.qtypes.gguf as G
orig = G._gguf_dequantize_tensor
sys.path.insert(0, str(HERE)); import gguf_triton_mod  # noqa: E401,F401
fast = G._gguf_dequantize_tensor
torch.manual_seed(0); bad = 0
for name, size in (("Q4_K", 144), ("Q6_K", 210)):
    qt = getattr(gguf.GGMLQuantizationType, name)
    for rows, cols in ((3, 512), (5376, 5376), (21504, 5376)):
        nb = rows * cols // 256
        raw = torch.randint(0, 256, (nb, size), dtype=torch.uint8, device="cuda")
        scale = (torch.rand(nb, 2, device="cuda") * 0.02 - 0.01).to(torch.float16)
        if name == "Q4_K": raw[:, :4] = scale.view(torch.uint8).reshape(nb, 4)
        else: raw[:, 208:210] = scale[:, :1].contiguous().view(torch.uint8).reshape(nb, 2)
        raw = raw.reshape(rows, -1)
        for dtype in (torch.float32, torch.bfloat16):
            a = orig(raw, qt, (rows, cols), dtype=dtype); b = fast(raw, qt, (rows, cols), dtype=dtype)
            err = (a.float() - b.float()).abs().max().item(); ref = a.float().abs().max().item()
            same = torch.equal(a, b)
            if err > 1e-3 * ref: bad += 1
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): orig(raw, qt, (rows, cols), dtype=dtype)
            torch.cuda.synchronize(); t_o = (time.time() - t) / 5; t = time.time()
            for _ in range(5): fast(raw, qt, (rows, cols), dtype=dtype)
            torch.cuda.synchronize(); t_f = (time.time() - t) / 5
            print(f"{name} {rows}x{cols} {str(dtype)[6:]}: max err {err:.3g} (max {ref:.3g}) identical={same}  torch {t_o * 1e3:.1f} ms  triton {t_f * 1e3:.1f} ms", flush=True)
print("FAILED" if bad else "ALL OK")
