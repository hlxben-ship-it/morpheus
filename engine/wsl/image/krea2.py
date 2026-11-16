"""Krea 2 (Muse and other finetunes) in our own loop: text to image, and reference images through the krea2edit path.

Sources this follows (Glia seed muse_krea2):
- the DiT: diffusers main `transformer_krea2.py` (Apache-2.0, Krea AI + HF), vendored here because diffusers 0.36 has no Krea 2;
  the checkpoint keys are ComfyUI's (`first.`, `blocks.N.attn.wq` ...), renamed as diffusers' `convert_krea2_transformer_checkpoint_to_diffusers`.
- sampling: ComfyUI `ModelSamplingFlux` shift 1.15 (sampling.FluxShift), velocity output, x0 = x - sigma * v.
- the text: ComfyUI `text_encoders/krea2.py`: Qwen3-VL-4B, hidden_states[2, 5, ..., 35] stacked, the system + "<|im_start|>user\\n"
  prefix cut, no padding. Needs transformers >= 4.57 (the "k2" worker stack puts /root/k2site first on sys.path).
- references: `lbouaraba/comfyui-krea2edit` 2.x (Apache-2.0) with the Identity / General Edit LoRAs: each reference is VAE-encoded
  ("fit" mode: resampled to the target's grid density, centred), its clean tokens go between the text and the target with RoPE
  frame 1, 2, ... (target = frame 0), only the target tokens come out; the text encoder sees the reference pictures too
  (grounded encode, longest side 768), and with CFG the negative is the empty prompt grounded on the same pictures.
"""
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common, sampling

ROOT = Path("/root/models/image/krea2")
TE_DIR = ROOT / "qwen3vl"                                  # Qwen/Qwen3-VL-4B-Instruct (what Krea 2 conditions on)
VAE_DIR = Path("/root/models/image/qwen-image-edit-2511/vae")  # the Qwen-Image VAE, shared with Qwen-Edit
TAPS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
SYSTEM = ("Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects "
          "and background:")
VISION = "<|vision_start|><|image_pad|><|vision_end|>"
IM_START, USER, NEWLINE = 151644, 872, 198
GROUNDING_PX = 768
SHIFT = 1.15
MAX_REFS = 2   # the edit LoRAs are trained on scene (picture 1) + person (picture 2)
DIT_GPU_GB = 10
CROP_TOL = 0.08
KEEP_TE = True   # keep the encoder in RAM between prompts (a new prompt cost ~80 s of disk read otherwise)
TE_8BIT = True   # ...as int8 linears (~4.5 GB instead of 8.9: the 24 GB VM also holds the DiT's streamed blocks)


class Int8Linear(nn.Module):
    """Weight-only int8 (per output row absmax), dequantised on the fly. Buffers, so common.Streamer moves them."""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data.float()
        scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
        self.register_buffer("q", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.register_buffer("bias", None if lin.bias is None else lin.bias.data.clone())
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        w = self.q.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))


def quantize_linears(module):
    """Swap every nn.Linear inside `module` for an Int8Linear, in place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Int8Linear(child))
        else:
            quantize_linears(child)
    return module


# ---------------------------------------------------------------- the DiT (diffusers transformer_krea2.py, trimmed)

class RMSNorm(nn.Module):
    """Zero-centred scale (multiplier 1 + weight), computed in float32."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim, self.eps = dim, eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.rms_norm(x.float(), (self.dim,), weight=self.weight.float() + 1.0, eps=self.eps).to(x.dtype)


def rope_apply(x, rope):
    """x [B, S, H, D]; rope (cos, sin) [S, D], pairs interleaved (diffusers apply_rotary_emb, use_real, unbind -1)."""
    cos, sin = rope
    xr = x.float().unflatten(-1, (-1, 2))
    rot = torch.stack([-xr[..., 1], xr[..., 0]], dim=-1).flatten(-2)
    return (x.float() * cos[None, :, None] + rot * sin[None, :, None]).to(x.dtype)


def rope_freqs(ids, axes=(32, 48, 48), theta=1000.0):
    """ids [S, 3] -> (cos, sin) [S, 128] (diffusers get_1d_rotary_pos_embed, repeat_interleave_real, float64)."""
    cos, sin = [], []
    pos = ids.double()
    for i, d in enumerate(axes):
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device=ids.device) / d))
        f = pos[:, i:i + 1] * inv[None]
        cos.append(f.cos().repeat_interleave(2, dim=1)); sin.append(f.sin().repeat_interleave(2, dim=1))
    return torch.cat(cos, 1).float(), torch.cat(sin, 1).float()


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads, eps=1e-5):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.to_v = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.to_gate = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps)
        self.norm_k = RMSNorm(self.head_dim, eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])

    def forward(self, x, rope=None):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, self.head_dim)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.kv_heads, self.head_dim)))
        v = self.to_v(x).unflatten(-1, (self.kv_heads, self.head_dim))
        gate = self.to_gate(x)
        if rope is not None:
            q, k = rope_apply(q, rope), rope_apply(k, rope)
        g = self.heads // self.kv_heads
        if g > 1:
            k, v = k.repeat_interleave(g, dim=2), v.repeat_interleave(g, dim=2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        return self.to_out[0](o.flatten(2) * torch.sigmoid(gate))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False), nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class FusionBlock(nn.Module):
    def __init__(self, dim, heads, kv_heads, inter, eps):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, kv_heads, eps)
        self.ff = SwiGLU(dim, inter)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class TextFusion(nn.Module):
    def __init__(self, layers, dim, heads, kv_heads, inter, n_layerwise, n_refiner, eps):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([FusionBlock(dim, heads, kv_heads, inter, eps) for _ in range(n_layerwise)])
        self.projector = nn.Linear(layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([FusionBlock(dim, heads, kv_heads, inter, eps) for _ in range(n_refiner)])

    def forward(self, h):  # [B, S, L, D]
        b, s, l, d = h.shape
        x = h.reshape(b * s, l, d)
        for blk in self.layerwise_blocks:
            x = blk(x.contiguous())
        x = self.projector(x.reshape(b, s, l, d).permute(0, 1, 3, 2)).squeeze(-1)
        for blk in self.refiner_blocks:
            x = blk(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, inter, heads, kv_heads, eps):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(6, dim))
        self.norm1, self.norm2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.attn = Attention(dim, heads, kv_heads, eps)
        self.ff = SwiGLU(dim, inter)

    def forward(self, x, temb, rope):
        m = temb.unflatten(-1, (6, -1)) + self.scale_shift_table.to(temb.dtype)
        sc1, sh1, g1, sc2, sh2, g2 = m.unbind(-2)
        x = x + g1 * self.attn((1 + sc1) * self.norm1(x) + sh1, rope)
        return x + g2 * self.ff((1 + sc2) * self.norm2(x) + sh2)


class TimeEmbed(nn.Module):
    def __init__(self, embed_dim, dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1, self.linear_2 = nn.Linear(embed_dim, dim), nn.Linear(dim, dim)

    def forward(self, t, dtype):
        half = self.embed_dim // 2
        freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        a = (t.float() * 1e3)[:, None, None] * freqs
        return self.linear_2(F.gelu(self.linear_1(torch.cat([a.cos(), a.sin()], -1).to(dtype)), approximate="tanh"))


class TextProj(nn.Module):
    def __init__(self, text_dim, dim, eps):
        super().__init__()
        self.norm = RMSNorm(text_dim, eps)
        self.linear_1, self.linear_2 = nn.Linear(text_dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(self.norm(x)), approximate="tanh"))


class FinalLayer(nn.Module):
    def __init__(self, dim, out, eps):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(2, dim))
        self.norm = RMSNorm(dim, eps)
        self.linear = nn.Linear(dim, out)

    def forward(self, x, temb):
        scale, shift = (temb + self.scale_shift_table.to(temb.dtype)).chunk(2, dim=1)
        return self.linear((1 + scale) * self.norm(x) + shift)


class Krea2DiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, eps = cfg["hidden"], 1e-5
        self.cfg = cfg
        self.img_in = nn.Linear(cfg["in_channels"], d)
        self.time_embed = TimeEmbed(256, d)
        self.time_mod_proj = nn.Linear(d, 6 * d)
        self.text_fusion = TextFusion(cfg["text_layers"], cfg["text_dim"], cfg["text_heads"], cfg["text_kv_heads"], cfg["text_inter"],
                                      cfg["n_layerwise"], cfg["n_refiner"], eps)
        self.txt_in = TextProj(cfg["text_dim"], d, eps)
        self.transformer_blocks = nn.ModuleList([Block(d, cfg["inter"], cfg["heads"], cfg["kv_heads"], eps) for _ in range(cfg["layers"])])
        self.final_layer = FinalLayer(d, cfg["in_channels"], eps)


def dit_config(sd):
    """The model's shape from the checkpoint (diffusers names)."""
    shape = lambda k: tuple(getattr(sd[k], "quant_shape", None) or sd[k].shape)
    hidden, in_ch = shape("img_in.weight")
    kv = shape("transformer_blocks.0.attn.to_k.weight")[0]
    tkv = shape("text_fusion.layerwise_blocks.0.attn.to_k.weight")[0]
    text_dim = shape("txt_in.linear_1.weight")[1]
    blocks = lambda pre: len({int(m.group(1)) for k in sd for m in [re.match(pre + r"\.(\d+)\.", k)] if m})
    return {"hidden": hidden, "in_channels": in_ch, "heads": hidden // 128, "kv_heads": kv // 128,
            "inter": shape("transformer_blocks.0.ff.gate.weight")[0], "layers": blocks("transformer_blocks"),
            "text_dim": text_dim, "text_heads": text_dim // 128, "text_kv_heads": tkv // 128,
            "text_inter": shape("text_fusion.layerwise_blocks.0.ff.gate.weight")[0],
            "text_layers": shape("text_fusion.projector.weight")[1],
            "n_layerwise": blocks("text_fusion.layerwise_blocks"), "n_refiner": blocks("text_fusion.refiner_blocks")}


# ---------------------------------------------------------------- checkpoint names (ComfyUI -> ours)

PREFIX = (("first.", "img_in."), ("tmlp.0.", "time_embed.linear_1."), ("tmlp.2.", "time_embed.linear_2."),
          ("tproj.1.", "time_mod_proj."), ("txtmlp.0.scale", "txt_in.norm.weight"), ("txtmlp.1.", "txt_in.linear_1."),
          ("txtmlp.3.", "txt_in.linear_2."), ("txtfusion.", "text_fusion."), ("blocks.", "transformer_blocks."),
          ("last.linear.", "final_layer.linear."), ("last.norm.scale", "final_layer.norm.weight"),
          ("last.modulation.lin", "final_layer.scale_shift_table"))
INNER = ((".attn.wq.", ".attn.to_q."), (".attn.wk.", ".attn.to_k."), (".attn.wv.", ".attn.to_v."), (".attn.wo.", ".attn.to_out.0."),
         (".attn.gate.", ".attn.to_gate."), (".attn.qknorm.qnorm.scale", ".attn.norm_q.weight"),
         (".attn.qknorm.knorm.scale", ".attn.norm_k.weight"), (".mlp.", ".ff."), (".prenorm.scale", ".norm1.weight"),
         (".postnorm.scale", ".norm2.weight"), (".mod.lin", ".scale_shift_table"))
# the text fusion blocks of ComfyUI's Krea 2 use the same inner names as the main blocks (attn.wq ..., prenorm, mlp)


def rename(key):
    """A ComfyUI / original Krea 2 tensor name -> this module's name; None when it is not a DiT tensor."""
    k = key
    for pre in ("model.diffusion_model.", "diffusion_model.", "transformer.", "model."):
        if k.startswith(pre):
            k = k[len(pre):]
            break
    for old, new in PREFIX:
        if k.startswith(old):
            k = new + k[len(old):]
            break
    else:
        if not k.startswith(("img_in.", "time_embed.", "time_mod_proj.", "text_fusion.", "txt_in.", "transformer_blocks.", "final_layer.")):
            return None
    for old, new in INNER:
        k = k.replace(old, new)
    return k


def _dequant(v):
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    return dequantize_gguf_tensor(v).float() if isinstance(v, GGUFParameter) else v.float()


def load_dit(path):
    """The GGUF DiT on the CPU: quantised linears stay quantised (diffusers GGUFLinear), the rest in bf16, norms in fp32."""
    from accelerate import init_empty_weights
    from diffusers.models.model_loading_utils import load_gguf_checkpoint
    from diffusers.quantizers.gguf.utils import GGUFParameter, _replace_with_gguf_linear
    raw = load_gguf_checkpoint(str(path), return_tensors=True)
    sd = {}
    for k, v in raw.items():
        nk = rename(k)
        if nk is None:
            continue
        if nk.endswith("scale_shift_table"):
            v = _dequant(v)
            v = v.reshape(6, -1) if nk.startswith("transformer_blocks.") else v.reshape(2, -1)
        sd[nk] = v
    cfg = dit_config(sd)
    with init_empty_weights():
        model = Krea2DiT(cfg)
    _replace_with_gguf_linear(model, torch.bfloat16, sd)
    for k, v in list(sd.items()):
        if isinstance(v, GGUFParameter):
            continue
        sd[k] = v.float() if (v.ndim == 1 and "norm" in k) else v.to(torch.bfloat16)  # RMSNorm scales stay fp32
    _assign(model, sd)  # not load_state_dict: a GGUF weight's byte shape differs from the linear's logical shape
    return model.eval(), cfg


def _assign(model, sd):
    """Put every tensor of sd in place (as diffusers' load_model_dict_into_meta does for GGUF); refuse a gap."""
    mods = dict(model.named_modules())
    want = {n for n, _ in model.named_parameters()}
    extra = sorted(set(sd) - want)
    if extra:
        print(f"[krea2] unused tensors: {extra[:5]}", flush=True)
    missing = sorted(want - set(sd))
    if missing:
        raise RuntimeError(f"Krea 2 weights missing: {missing[:5]}")
    for name in want:
        mod, leaf = name.rsplit(".", 1)
        v = sd[name]
        setattr(mods[mod], leaf, v if isinstance(v, nn.Parameter) else nn.Parameter(v, requires_grad=False))


# ---------------------------------------------------------------- LoRA (own, as forward hooks: works on GGUF and streamed linears)

def lora_pairs(sd):
    """{module name: (down [r, in], up [out, r], alpha or None)} from kohya (lora_down/up), PEFT (lora_A/B) or ai-toolkit names."""
    out = {}
    for k in sd:
        m = re.match(r"(.+?)\.(lora_down|lora_A)(?:\.default)?\.weight$", k)
        if not m:
            continue
        base, kind = m.group(1), m.group(2)
        up_k = k.replace("lora_down", "lora_up") if kind == "lora_down" else k.replace("lora_A", "lora_B")
        if up_k not in sd:
            continue
        alpha = sd.get(base + ".alpha")
        name = rename(base if base.endswith(".weight") else base + ".weight")
        if name:
            out[name[:-len(".weight")]] = (sd[k], sd[up_k], float(alpha) if alpha is not None else None)
    return out


class Loras:
    def __init__(self, dit):
        self.dit, self.handles, self.key = dit, [], ()

    def set(self, loras):
        from safetensors.torch import load_file
        key = tuple((l["path"], float(l["strength"])) for l in loras)
        if key == self.key:
            return
        for h in self.handles:
            h.remove()
        self.handles = []
        mods = dict(self.dit.named_modules())
        for l in loras:
            pairs = lora_pairs(load_file(l["path"]))
            hit = 0
            for name, (down, up, alpha) in pairs.items():
                mod = mods.get(name)
                if mod is None:
                    continue
                r = down.shape[0]
                scale = float(l["strength"]) * ((alpha / r) if alpha else 1.0)
                d, u = down.to(common.DEVICE, torch.bfloat16), up.to(common.DEVICE, torch.bfloat16)
                self.handles.append(mod.register_forward_hook(
                    lambda m, inp, out, d=d, u=u, s=scale: out + F.linear(F.linear(inp[0].to(torch.bfloat16), d), u).to(out.dtype) * s))
                hit += 1
            if not hit:
                raise ValueError(f"No layer of this LoRA matches Krea 2: {Path(l['path']).name}")
            print(f"[krea2] LoRA {Path(l['path']).name}: {hit}/{len(pairs)} layers", flush=True)
        self.key = key


# ---------------------------------------------------------------- references (krea2edit "fit" pixel path)

def fit_pixels(iw, ih, px_w, px_h):
    """krea2edit _fit_encode_image(fit): the crop box (x0, y0, cw, ch) of the source and the size (nw, nh) it is resampled to."""
    sc = min(px_h / ih, px_w / iw)
    if ih * sc >= px_h * (1 - CROP_TOL) and iw * sc >= px_w * (1 - CROP_TOL):  # near the target's aspect: fill it exactly
        s = max(px_h / ih, px_w / iw)
        ch, cw = min(ih, int(round(px_h / s))), min(iw, int(round(px_w / s)))
        return ((iw - cw) // 2, (ih - ch) // 2, cw, ch), (px_w, px_h)
    nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
    nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
    ch, cw = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
    return ((iw - cw) // 2, (ih - ch) // 2, cw, ch), (nw, nh)


def ref_ids(frame, gh, gw, th, tw, device, stride=1.0):
    """Positions of a reference grid centred (fractional offset allowed) on the target grid, on RoPE frame `frame`.
    stride 1 = krea2edit's trained "fit" geometry; a reference shrunk by s (ref_scale) gets stride 1/s, so it still spans the
    area it would have covered at full size (RoPE is continuous; krea2edit warns this can add artifacts: hence an option)."""
    oh, ow = max(0.0, (th - gh * stride) / 2), max(0.0, (tw - gw * stride) / 2)
    ids = torch.zeros(gh, gw, 3, device=device)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) * stride + oh)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) * stride + ow)[None, :]
    return ids.reshape(gh * gw, 3)


def target_ids(th, tw, device):
    return ref_ids(0, th, tw, th, tw, device)


def pack(lat):  # [B, C, H, W] -> [B, H/2*W/2, C*4] (ComfyUI "b c (h ph) (w pw) -> b (h w) (c ph pw)")
    b, c, h, w = lat.shape
    return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(x, h, w):
    b, n, c4 = x.shape
    c = c4 // 4
    return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def dit_forward(dit, x, sigma, text, refs, rope):
    """x: packed target [1, N, 64]; text: [1, S, 12, 2560]; refs: list of packed clean reference tokens; rope: for [text|refs|target]."""
    temb = dit.time_embed(sigma, x.dtype)
    tmod = dit.time_mod_proj(F.gelu(temb, approximate="tanh"))
    txt = dit.txt_in(dit.text_fusion(text))
    parts = [txt] + [dit.img_in(r) for r in refs] + [dit.img_in(x)]
    h = torch.cat(parts, dim=1)
    for blk in dit.transformer_blocks:
        h = blk(h, tmod, rope)
    n = x.shape[1]
    return dit.final_layer(h[:, -n:], temb)


# ---------------------------------------------------------------- the pipeline

class Krea2:
    family = "krea2"

    def __init__(self, gguf):
        from diffusers import AutoencoderKLQwenImage
        from transformers import AutoProcessor
        self.dit, self.cfg = load_dit(gguf)
        self._place()
        self.vae = AutoencoderKLQwenImage.from_pretrained(VAE_DIR, torch_dtype=torch.bfloat16).to(common.DEVICE).eval()
        self.vae.enable_tiling()
        self.processor = AutoProcessor.from_pretrained(TE_DIR)
        self.te = self.te_stream = None
        self._cond = (None, None)
        self.mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(common.DEVICE)
        self.std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(common.DEVICE)
        self.ms = sampling.FluxShift(SHIFT)
        self.loras = Loras(self.dit)

    def _place(self):
        """First blocks resident up to DIT_GPU_GB, the rest streamed from pinned RAM each step (as Qwen-Edit)."""
        blocks = list(self.dit.transformer_blocks)
        size = lambda m: sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
        budget, used, resident = DIT_GPU_GB * 2**30, 0, 0
        for blk in blocks:
            if used + size(blk) > budget:
                break
            used += size(blk); resident += 1
        for name, m in self.dit.named_children():
            if name != "transformer_blocks":
                m.to(common.DEVICE)
        for blk in blocks[:resident]:
            blk.to(common.DEVICE)
        self.dit_stream = common.Streamer(blocks[resident:], skip=None)
        self.resident_blocks = resident
        common.free_vram()

    def gb(self):
        return 16.0

    def set_loras(self, loras):
        self.loras.set(loras)

    # ---- text ----
    def _load_te(self):
        """Place the encoder: small parts on the GPU, layers streamed. The weights stay in RAM between prompts (no
        second 8.9 GB read); _drop_te only takes it off the GPU."""
        if self.te_stream is not None:
            return
        if self.te is None:
            from transformers import Qwen3VLForConditionalGeneration  # the checkpoint's names are model.language_model.*: Qwen3VLModel loads none
            self.te = Qwen3VLForConditionalGeneration.from_pretrained(TE_DIR, dtype=torch.bfloat16).model.eval()
            if TE_8BIT:
                for layer in self.te.language_model.layers:
                    quantize_linears(layer)
                common.free_vram()
        lm, vis = self.te.language_model, self.te.visual
        streamed = [lm.embed_tokens, *lm.layers, *vis.blocks]
        inside = {id(x) for m in streamed for x in m.modules()}
        for m in self.te.modules():
            if id(m) not in inside and not any(id(x) in inside for x in m.modules()):
                m.to(common.DEVICE)
        self.te_stream = common.Streamer(streamed, skip=None, pin=False)

    def _drop_te(self):
        if self.te_stream is None:
            return
        self.te_stream.remove()
        self.te_stream = None
        self.te.to("cpu")
        if not KEEP_TE:
            self.te = None
        common.free_vram()

    @torch.no_grad()
    def conditioning(self, texts, images, key):
        if self._cond[0] == key:
            return self._cond[1]
        self._load_te()
        try:
            out = [self.encode(t, images) if t is not None else None for t in texts]
        finally:
            self._drop_te()
        self._cond = (key, out)
        return out

    @torch.no_grad()
    def encode(self, prompt, images):
        """[1, S, 12, 2560]: the 12 tapped layers of Qwen3-VL, system + user-opening prefix cut (ComfyUI Krea2TEModel)."""
        text = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{VISION * len(images)}{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt")
        inputs = {k: v.to(common.DEVICE) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        out = self.te(**inputs, output_hidden_states=True)
        ids = inputs["input_ids"][0].tolist()
        cut = prefix_len(ids)
        h = torch.stack([out.hidden_states[i][0] for i in TAPS], dim=1)  # [S, 12, 2560]
        return h[cut:].unsqueeze(0).to(torch.bfloat16)

    # ---- pictures ----
    @torch.no_grad()
    def vae_encode(self, t01):
        lat = self.vae.encode((t01 * 2 - 1).unsqueeze(2).to(common.DEVICE, torch.bfloat16)).latent_dist.mode()
        return ((lat.float() - self.mean) / self.std)[:, :, 0]

    def reference(self, path, w, h):
        """(packed clean tokens, grid h, grid w) of one reference, fitted the krea2edit way to a w x h target (w, h: multiples of 16)."""
        import numpy as np
        from PIL import Image
        img = Image.open(path).convert("RGB")
        (x0, y0, cw, ch), (nw, nh) = fit_pixels(img.width, img.height, w, h)
        t = torch.from_numpy(np.asarray(img.crop((x0, y0, x0 + cw, y0 + ch)), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(nh, nw), mode="bicubic", antialias=True).clamp(0, 1)
        lat = self.vae_encode(t)
        return pack(lat), lat.shape[2] // 2, lat.shape[3] // 2

    @staticmethod
    def grounding(path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        s = GROUNDING_PX / max(img.size)
        return img.resize((round(img.width * s), round(img.height * s)), Image.BOX) if s < 1 else img

    @torch.no_grad()
    def generate(self, p, emit, stopped):
        """p: prompt, negative, width, height, seed, steps, cfg, sampler, scheduler, images [paths] (0-2), loras."""
        import os
        self.set_loras(p.get("loras") or [])
        w, h = int(p["width"]) // 16 * 16, int(p["height"]) // 16 * 16
        paths = list(p.get("images") or [])[:MAX_REFS]
        cfg = float(p.get("cfg") or 1.0)
        key = (p["prompt"], p.get("negative") or "", cfg != 1.0, tuple((x, os.path.getmtime(x)) for x in paths))
        ground = [self.grounding(x) for x in paths]
        neg = (p.get("negative") or "") if cfg != 1.0 else None
        cond, uncond = self.conditioning([p["prompt"], neg], ground, key)
        lh, lw = h // 8, w // 8
        th, tw = lh // 2, lw // 2
        refs, ids = [], []
        rs = float(p.get("ref_scale") or 0.75)  # < 1: references at a smaller grid, fewer tokens, faster steps (Glia seed muse_krea2)
        rw, rh = (max(16, round(w * rs / 16) * 16), max(16, round(h * rs / 16) * 16)) if rs < 1 else (w, h)
        for n, path in enumerate(paths):
            tok, gh, gw = self.reference(path, rw, rh)
            refs.append(tok.to(torch.bfloat16)); ids.append(ref_ids(n + 1, gh, gw, th, tw, common.DEVICE, stride=(w / rw) if rs < 1 else 1.0))
        ropes = {}

        def rope_for(text):
            s = text.shape[1]
            if s not in ropes:
                pos = torch.cat([torch.zeros(s, 3, device=common.DEVICE)] + ids + [target_ids(th, tw, common.DEVICE)])
                ropes[s] = rope_freqs(pos)
            return ropes[s]

        def velocity(xp, sigma, text):
            return dit_forward(self.dit, xp.to(torch.bfloat16), sigma.to(torch.bfloat16).view(1), text, refs, rope_for(text)).float()

        def denoise(x, sigma):
            xp = pack(x)
            v = velocity(xp, sigma, cond)
            if uncond is not None:
                v = _cfg(v, velocity(xp, sigma, uncond), cfg)
            return unpack(xp - v * sigma.view(-1, 1, 1), lh, lw)

        sig = sampling.sigmas_for(self.ms, p.get("scheduler") or "simple", int(p["steps"])).to(common.DEVICE)
        noise = sampling.initial_noise([1, 16, lh, lw], int(p["seed"])).to(common.DEVICE)
        x = sampling.noise_scaling("flow", sig[0], noise, torch.zeros_like(noise), True)
        x = sampling.sample(p.get("sampler") or "euler", denoise, x, sig, "flow", int(p["seed"]), stopped=stopped, ms=self.ms,
                            eta=float(p.get("eta", 1.0)), callback=lambda i, n, x0: emit(i, n, x0[0] * self.std[0, :, 0] + self.mean[0, :, 0]))
        lat = (x.unsqueeze(2) * self.std + self.mean).to(torch.bfloat16)
        return common.to_pil(self.vae.decode(lat).sample[:, :, 0])[0]

    def preview_factors(self):
        return common.flow_rgb()


def _cfg(cond, uncond, scale):
    """ComfyUI CFG on the prediction: uncond + scale * (cond - uncond)."""
    return uncond + scale * (cond - uncond)


def prefix_len(ids):
    """ComfyUI Krea2TEModel: cut at the second <|im_start|>, plus "user\\n" after it."""
    seen, end = 0, 0
    for i, t in enumerate(ids):
        if t == IM_START and seen < 2:
            end, seen = i, seen + 1
    if len(ids) > end + 3 and ids[end + 1] == USER and ids[end + 2] == NEWLINE:
        end += 3
    return end
