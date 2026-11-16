"""Qwen DiT alone (no text encoder): is the first step slow by itself? And what VRAM does WSL see free."""
import sys, time
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); sys.path.insert(0, "/root/wangp")
import torch
from image import qwen_edit
print("vram free/total GB at start", [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()], flush=True)
from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel
t = time.time()
dit = QwenImageTransformer2DModel.from_single_file(str(qwen_edit.GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=str(qwen_edit.ROOT), subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
print("dit load", round(time.time() - t, 1), "vram free", round(torch.cuda.mem_get_info()[0] / 2**30, 1), flush=True)
c = torch.randn(1, 207, 3584, device="cuda", dtype=torch.bfloat16)
x = torch.randn(1, 2 * 52 * 76, 64, device="cuda", dtype=torch.bfloat16)
for i in range(4):
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        qwen_edit.dit_forward(dit, x, c, torch.tensor([0.9], device="cuda", dtype=torch.bfloat16), [(1, 52, 76), (1, 52, 76)], 207)
    torch.cuda.synchronize(); print("step", i, round(time.time() - t, 2), "vram free", round(torch.cuda.mem_get_info()[0] / 2**30, 1), flush=True)
