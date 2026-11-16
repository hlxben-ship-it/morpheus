"""Shared bits of the WSL image pipelines: devices, layer-by-layer offload, the step preview, saving."""
import json
import time
from pathlib import Path

import torch

DEVICE = torch.device("cuda")

# ComfyUI latent_formats.SDXL.latent_rgb_factors / bias: a cheap picture of the latent for the step preview
SDXL_RGB = ([[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068], [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]],
            [0.1084, -0.0175, -0.0011])


def flow_rgb():
    """Wan 2.1 latent (Qwen-Image shares it) -> RGB factors from WanGP's table (the worker has /root/wangp on sys.path)."""
    from shared.RGB_factors import get_rgb_factors
    factors, bias = get_rgb_factors("qwen")
    return factors, bias


def preview(latent, factors, bias, path: Path, max_side=384):
    """latent [C, H, W] (unscaled model space) -> small JPEG. Never raises: a preview is a nicety."""
    try:
        from PIL import Image
        f = torch.tensor(factors, dtype=torch.float32, device=latent.device)
        b = torch.tensor(bias, dtype=torch.float32, device=latent.device)
        rgb = torch.einsum("chw,cr->hwr", latent.float(), f) + b
        rgb = ((rgb.clamp(-1, 1) + 1) * 127.5).byte().cpu().numpy()
        img = Image.fromarray(rgb)
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
        tmp = path.with_suffix(".tmp.jpg")
        img.save(tmp, quality=70)
        tmp.replace(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def to_pil(images):
    """[B, 3, H, W] in [-1, 1] -> PIL images."""
    from PIL import Image
    arr = ((images.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in arr]


def load_image(path, size=None):
    """A picture as RGB PIL, optionally resized (lanczos) + center-cropped to (w, h) like ComfyUI ImageScale crop=center."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if size:
        w, h = size
        scale = max(w / img.width, h / img.height)
        nw, nh = max(w, round(img.width * scale)), max(h, round(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        img = img.crop((left, top, left + w, top + h))
    return img


def pil_to_tensor(img):
    import numpy as np
    a = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    return a.permute(2, 0, 1).unsqueeze(0)


def save_png(img, path: Path, meta: dict):
    """PNG with the job's parameters in a text chunk (the sidecar json is written by the server)."""
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("morpheus", json.dumps(meta))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, pnginfo=info)
    return path


class Streamer:
    """Weights in pinned RAM, on the card only while their module runs (ComfyUI lowvram / WanGP-style block streaming).

    Each listed module's parameters and buffers keep a pinned CPU copy; a forward pre-hook points them at a fresh GPU copy
    (13 GB/s pinned in WSL, measured 2026-11-16), the forward hook points them back at the CPU copy and the GPU copy is
    dropped. No .to("cpu") on the way back: that made a new pageable copy each time and the process grew by GBs.
    Why at all: WSL cannot see what the Windows desktop holds on the card (it reports 19 of 20 GB free), so a resident
    15 GB DiT plus the 9 GB encoder spilled silently and a step took 50-180 s instead of 6 s. `skip` keeps tensors whose
    name contains it (LoRA weights) resident on the card instead.
    """

    def __init__(self, modules, skip="lora_", pin=True):
        self.handles = []
        self.bytes = 0
        self.pinned = 0
        for m in modules:
            tensors = []
            for name, p in list(m.named_parameters()) + list(m.named_buffers()):
                if skip and skip in name:
                    p.data = p.data.to(DEVICE)
                    continue
                cpu = p.data if p.data.device.type == "cpu" else p.data.cpu()
                if pin and not cpu.is_pinned():
                    try:  # WSL caps pinned host memory (the encoder's 9.4 GB on top of 5 GB failed): pageable then
                        cpu = cpu.pin_memory(); self.pinned += cpu.numel() * cpu.element_size()
                    except (RuntimeError, torch.AcceleratorError if hasattr(torch, "AcceleratorError") else RuntimeError):
                        pin = False
                p.data = cpu
                tensors.append((p, cpu))
                self.bytes += cpu.numel() * cpu.element_size()
            m._stream = tensors
            self.handles.append(m.register_forward_pre_hook(self._in))
            self.handles.append(m.register_forward_hook(self._out))

    @staticmethod
    def _in(module, args):
        for p, cpu in module._stream:
            p.data = cpu.to(DEVICE, non_blocking=True)

    @staticmethod
    def _out(module, args, out):
        for p, cpu in module._stream:
            p.data = cpu
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class Timer:
    def __init__(self):
        self.t0 = time.time(); self.marks = {}

    def mark(self, name):
        self.marks[name] = round(time.time() - self.t0, 2)


def free_vram():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:  # glibc keeps freed big blocks in the process; hand them back so the WSL VM (and Windows) gets the RAM
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
