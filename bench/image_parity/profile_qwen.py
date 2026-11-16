"""Time each part of one Qwen-Edit job inside WSL (run via profile.sh): load, VAE encode, text encode, per DiT step, decode."""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); sys.path.insert(0, "/root/wangp")
import torch
from image import qwen_edit, common
T = time.time
def memo(tag):
    m = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.split(":")[0] in ("MemAvailable", "Cached", "SwapTotal", "SwapFree", "AnonPages")}
    rss = int([l for l in open("/proc/self/status") if l.startswith("VmRSS")][0].split()[1]) // 1024
    print("  mem", tag, "avail", m["MemAvailable"], "cached", m["Cached"], "anon", m["AnonPages"], "swap_used", m["SwapTotal"] - m["SwapFree"], "rss", rss, "vram", round(torch.cuda.memory_allocated() / 2**30, 1), flush=True)
t = T(); q = qwen_edit.QwenEdit(); print("load", round(T() - t, 1), "resident blocks", q.resident_blocks, flush=True); memo("after load")
q.set_loras([{"path": "/mnt/c/dev/morpheus/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength": 1.0}])
img = common.load_image("/mnt/c/dev/morpheus/bench/image_parity/fox_in.jpg", (1216, 832)); t01 = qwen_edit.pil01(img)
torch.cuda.synchronize(); t = T(); lat = q.vae_encode(t01); torch.cuda.synchronize(); print("vae_encode", round(T() - t, 1), flush=True)
vw, vh = qwen_edit.vl_size(img.width, img.height)
vl = qwen_edit._to_pil(qwen_edit.area_resize(t01, vw, vh))
t = T(); c, _ = q.conditioning(["make it night, with warm window lights", None], [vl], "k1"); torch.cuda.synchronize(); print("conditioning", round(T() - t, 1), c.shape, flush=True); memo("after te")
t = T(); c, _ = q.conditioning(["make it winter, snow on the grass", None], [vl], "k2"); torch.cuda.synchronize(); print("conditioning_2", round(T() - t, 1), flush=True); memo("after te2")
ref = qwen_edit.pack(lat); x = torch.randn(1, 16, 104, 152, device="cuda"); xp = qwen_edit.pack(x)
shapes = [(1, 52, 76), (1, 52, 76)]
for i in range(3):
    torch.cuda.synchronize(); t = T()
    with torch.no_grad():
        o = qwen_edit.dit_forward(q.dit, torch.cat([xp, ref], 1).bfloat16(), c, torch.tensor([0.9], device="cuda").bfloat16(), shapes, c.shape[1])
    torch.cuda.synchronize(); print("dit_step", i, round(T() - t, 2), flush=True); memo("step")
print("sdpa backends", torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(), os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
qq = torch.randn(1, 24, 8200, 128, device="cuda", dtype=torch.bfloat16)
for name, ctx in [("default", None)]:
    torch.cuda.synchronize(); t = T()
    for _ in range(5): torch.nn.functional.scaled_dot_product_attention(qq, qq, qq)
    torch.cuda.synchronize(); print("sdpa x5", round(T() - t, 2), flush=True)
