"""Qwen-Image-Edit-2511 in our own loop, on the same files ComfyUI uses (Q5_K_M GGUF DiT, Qwen2.5-VL-7B fp8_scaled encoder,
Lightning LoRA) so a picture can be compared with the same seed there.

What is ours: the fp8 encoder (weights stay fp8 in RAM, upcast per layer, each layer visits the GPU only while it runs),
ComfyUI's image scaling for the vision and reference inputs, the DiT forward (written here because diffusers 0.36 has no
2511 "zero_cond_t": reference-image tokens are modulated at timestep 0, ComfyUI's index_timestep_zero), the flow sampling
(AuraFlow shift 3.1, ComfyUI simple scheduler), CFG with CFGNorm. diffusers supplies the module classes and the GGUF loader.
"""
import json
import math
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import common, sampling

ROOT = Path("/root/models/image/qwen-image-edit-2511")
GGUF = ROOT / "qwen-image-edit-2511-Q5_K_M.gguf"
TE = ROOT / "qwen_2.5_vl_7b_fp8_scaled.safetensors"
TEMPLATE = ("<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then "
            "explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's "
            "requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}"
            "<|im_end|>\n<|im_start|>assistant\n")
DROP = 64  # template tokens before the user's text (ComfyUI cuts at "<|im_start|>user\n" too)
PICTURE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
VL_AREA, REF_AREA = 384 * 384, 1024 * 1024
SHIFT = 3.1
MAX_REFS = 3
DIT_GPU_GB = 10  # resident DiT blocks; the rest streams (measured: Q5_K_M ~15 GB total, ~0.25 GB a block)


class Fp8Linear(nn.Module):
    """ComfyUI "fp8_scaled" linear: weight float8_e4m3fn * scale_weight, computed in bf16."""

    def __init__(self, inf, outf, bias):
        super().__init__()
        self.in_features, self.out_features = inf, outf
        self.weight = nn.Parameter(torch.empty(outf, inf, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.register_buffer("scale_weight", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(outf, dtype=torch.bfloat16), requires_grad=False) if bias else None

    def forward(self, x):
        w = self.weight.to(torch.bfloat16) * self.scale_weight.to(torch.bfloat16)
        return F.linear(x.to(torch.bfloat16), w, self.bias)


def _swap_linears(module, fp8_names, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and full in fp8_names:
            setattr(module, name, Fp8Linear(child.in_features, child.out_features, child.bias is not None))
        else:
            _swap_linears(child, fp8_names, full + ".")


def _te_key(k):
    """ComfyUI checkpoint names -> transformers 4.54 Qwen2_5_VLModel names."""
    if k.startswith("visual."):
        return k
    if k.startswith("model."):
        return "language_model." + k[len("model."):]
    return None  # lm_head: not needed for hidden states


def load_text_encoder(device="cuda"):
    """The fp8 encoder straight from the ext4 file onto `device` (no RAM copy: the file is mmapped)."""
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLModel
    cfg = Qwen2_5_VLConfig.from_pretrained(ROOT / "text_encoder")
    raw = load_file(str(TE), device=str(device))
    sd = {}
    for k, v in raw.items():
        nk = _te_key(k)
        if nk is None or nk.endswith(".scale_input"):
            continue
        sd[nk] = v
    fp8 = {k[:-len(".weight")] for k, v in sd.items() if k.endswith(".weight") and v.dtype == torch.float8_e4m3fn}
    with init_empty_weights():
        model = Qwen2_5_VLModel(cfg)
    _swap_linears(model, fp8)
    for k, v in sd.items():  # everything that is not fp8 or a scale computes in bf16
        if v.dtype in (torch.float32, torch.float16) and not k.endswith(".scale_weight"):
            sd[k] = v.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        raise RuntimeError(f"Qwen2.5-VL weights missing: {missing[:5]}")
    for mod in model.modules():  # rotary inv_freq buffers are made at init (on meta here): rebuild them on the device
        if hasattr(mod, "inv_freq") and mod.inv_freq.device.type == "meta":
            if hasattr(mod, "dim") and hasattr(mod, "theta"):  # vision rotary
                mod.inv_freq = 1.0 / (mod.theta ** (torch.arange(0, mod.dim, 2, dtype=torch.float, device=device) / mod.dim))
            else:  # text rotary
                inv, mod.attention_scaling = mod.rope_init_fn(mod.config, torch.device(device))
                mod.register_buffer("inv_freq", inv, persistent=False); mod.original_inv_freq = inv
    left = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if left:
        raise RuntimeError(f"Qwen2.5-VL buffers still on meta: {left[:3]}")
    return model.to(device).eval()


def area_resize(t, w, h):
    """[1, 3, H, W] in [0, 1] -> (w, h) like ComfyUI common_upscale(..., "area")."""
    return F.interpolate(t, size=(h, w), mode="area")


def vl_size(w, h):
    s = math.sqrt(VL_AREA / (w * h))
    return round(w * s), round(h * s)


def ref_size(w, h):
    s = math.sqrt(REF_AREA / (w * h))
    return round(w * s / 8) * 8, round(h * s / 8) * 8


def pil01(img):
    import numpy as np
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def pad2(lat):
    """ComfyUI common_dit.pad_to_patch_size: odd latent sides (references are rounded to 8 px) padded circularly to 2."""
    ph, pw = (-lat.shape[2]) % 2, (-lat.shape[3]) % 2
    return F.pad(lat, (0, pw, 0, ph), mode="circular") if ph or pw else lat


def pack(lat):  # [B, C, H, W] -> [B, H/2*W/2, C*4]
    lat = pad2(lat)
    b, c, h, w = lat.shape
    return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def unpack(x, h, w):  # inverse of pack, h, w in latent pixels
    b, n, c4 = x.shape
    c = c4 // 4
    return x.view(b, h // 2, w // 2, c, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def dit_forward(dit, hidden, text, timestep, img_shapes, txt_len):
    """QwenImageTransformer2DModel forward with 2511's zero_cond_t: the first image (the one being made) is modulated at
    `timestep`, every reference image at 0. Same math as diffusers 0.43, on 0.36's modules."""
    n_main = math.prod(img_shapes[0])
    index = torch.zeros(1, hidden.shape[1], dtype=torch.bool, device=hidden.device)
    index[:, n_main:] = True
    x = dit.img_in(hidden)
    t = torch.cat([timestep, timestep * 0]).to(x.dtype)
    txt = dit.txt_in(dit.txt_norm(text))
    temb = dit.time_text_embed(t, x)  # [2, D]: row 0 = the real timestep, row 1 = 0
    rope = dit.pos_embed([img_shapes], [txt_len], device=x.device)
    idx = index.unsqueeze(-1)

    def mod(v, params, per_token):
        shift, scale, gate = params.chunk(3, dim=-1)
        if per_token:
            sel = lambda p: torch.where(idx, p[1:2].unsqueeze(1), p[0:1].unsqueeze(1))
            shift, scale, gate = sel(shift), sel(scale), sel(gate)
        else:
            shift, scale, gate = shift[:1].unsqueeze(1), scale[:1].unsqueeze(1), gate[:1].unsqueeze(1)
        return v * (1 + scale) + shift, gate

    for blk in dit.transformer_blocks:
        streamed = hasattr(blk, "_stream")  # common.Streamer: its hooks fire on blk(...), but this loop calls the parts
        if streamed:
            common.Streamer._in(blk, None)
        try:
            x, txt = _block(blk, x, txt, temb, rope, mod)
        finally:
            if streamed:
                common.Streamer._out(blk, None, None)
    x = dit.norm_out(x, temb[:1])
    return dit.proj_out(x)


def _block(blk, x, txt, temb, rope, mod):
    """One QwenImageTransformerBlock; the image stream is modulated per token (main vs. reference), the text stream not."""
    im1, im2 = blk.img_mod(temb).chunk(2, dim=-1)
    tm1, tm2 = blk.txt_mod(temb[:1]).chunk(2, dim=-1)
    a, g1 = mod(blk.img_norm1(x), im1, True)
    b, tg1 = mod(blk.txt_norm1(txt), tm1, False)
    ia, ta = blk.attn(hidden_states=a, encoder_hidden_states=b, encoder_hidden_states_mask=None, image_rotary_emb=rope)
    x = x + g1 * ia
    txt = txt + tg1 * ta
    a2, g2 = mod(blk.img_norm2(x), im2, True)
    x = x + g2 * blk.img_mlp(a2)
    b2, tg2 = mod(blk.txt_norm2(txt), tm2, False)
    txt = txt + tg2 * blk.txt_mlp(b2)
    return x, txt


class QwenEdit:
    family = "qwen_edit"

    def __init__(self):
        from diffusers import AutoencoderKLQwenImage, GGUFQuantizationConfig, QwenImageTransformer2DModel
        from transformers import Qwen2VLProcessor
        self.dit = QwenImageTransformer2DModel.from_single_file(
            str(GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(ROOT), subfolder="transformer", torch_dtype=torch.bfloat16).eval()  # on the CPU: placed by _place()
        self._place()  # first: leaves only the streamed ~5 GB in RAM before the 9.4 GB encoder loads (24 GB VM)
        self.vae = AutoencoderKLQwenImage.from_pretrained(ROOT / "vae", torch_dtype=torch.bfloat16).to(common.DEVICE).eval()
        self.vae.enable_tiling()  # whole-picture encode wants ~5 GB beside the DiT: OOM on 20 GB (2026-11-16)
        self.processor = Qwen2VLProcessor.from_pretrained(ROOT / "processor")
        self.te = None  # the 9.4 GB encoder: loaded per new prompt, freed right after (_load_te / _drop_te)
        self._cond = (None, None)
        self.mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(common.DEVICE)
        self.std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(common.DEVICE)
        self.ms = sampling.Flow(SHIFT)
        self.loras = ()

    def _place(self):
        """DiT: the first blocks resident up to DIT_GPU_GB, the rest streamed from pinned RAM each step; LoRA weights resident."""
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
        self.dit_stream = common.Streamer(blocks[resident:], skip="lora_")
        self.resident_blocks = resident
        common.free_vram()

    def _load_te(self):
        """The fp8 encoder in RAM, streamed through the card layer by layer (9.4 GB pageable, ~1.3 s of copies).
        Only while a prompt is read: holding it through keep-warm left Windows ~1 GB free (2026-11-16)."""
        if self.te is not None:
            return
        self.te = load_text_encoder("cpu")
        lm, vis = self.te.language_model, self.te.visual
        streamed = [lm.embed_tokens, *lm.layers, vis.patch_embed, *vis.blocks, vis.merger]
        inside = {id(x) for m in streamed for x in m.modules()}
        for m in self.te.modules():  # the small rest (final norm, rotary buffers) lives on the card
            if id(m) not in inside and not any(id(x) in inside for x in m.modules()):
                m.to(common.DEVICE)
        self.te_stream = common.Streamer(streamed, skip=None, pin=False)  # pinned RAM is capped in WSL: the DiT tail gets it

    def _drop_te(self):
        if self.te is None:
            return
        self.te_stream.remove()
        self.te = self.te_stream = None
        common.free_vram()

    def gb(self):
        return 17.0

    def set_loras(self, loras):
        from diffusers import QwenImageEditPlusPipeline as P
        key = tuple((l["path"], float(l["strength"])) for l in loras)
        if key == self.loras:
            return
        if self.loras:
            self.dit.delete_adapters([f"l{n}" for n in range(len(self.loras))])
        names = []
        for n, l in enumerate(loras):
            sd = P.lora_state_dict(l["path"])
            sd = sd[0] if isinstance(sd, tuple) else sd
            P.load_lora_into_transformer(sd, transformer=self.dit, adapter_name=f"l{n}")
            names.append(f"l{n}")
        if names:
            self.dit.set_adapters(names, weights=[float(l["strength"]) for l in loras])
        for name, prm in self.dit.named_parameters():  # new LoRA weights land beside their base (pinned RAM): to the card
            if "lora_" in name and prm.data.device.type != "cuda":
                prm.data = prm.data.to(common.DEVICE)
        self.loras = key

    # ---- conditioning ----
    @torch.no_grad()
    def conditioning(self, texts, vl_images, key):
        """Encode all texts of a job (the encoder streams through the card). Cached per job key: a batch encodes once."""
        if self._cond[0] == key:
            return self._cond[1]
        self._load_te()
        try:
            out = [self.encode(t, vl_images) if t is not None else None for t in texts]
        finally:
            self._drop_te()
        self._cond = (key, out)
        return out

    @torch.no_grad()
    def encode(self, prompt, vl_images):
        pics = "".join(PICTURE.format(i + 1) for i in range(len(vl_images)))
        inputs = self.processor(text=[TEMPLATE.format(pics + prompt)], images=vl_images or None, padding=True, return_tensors="pt")
        inputs = {k: v.to(common.DEVICE) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        out = self.te(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs.get("pixel_values"),
                      image_grid_thw=inputs.get("image_grid_thw"), output_hidden_states=True)
        h = out.hidden_states[-1][0][inputs["attention_mask"][0].bool()][DROP:]
        return h.unsqueeze(0).to(torch.bfloat16)

    @torch.no_grad()
    def vae_encode(self, t01):
        lat = self.vae.encode((t01 * 2 - 1).unsqueeze(2).to(common.DEVICE, torch.bfloat16)).latent_dist.mode()
        return ((lat.float() - self.mean) / self.std)[:, :, 0]

    @torch.no_grad()
    def generate(self, p, emit, stopped):
        """p: prompt, negative, width, height, seed, steps, cfg, scheduler, sampler, denoise, images [paths] (1-3), loras."""
        self.set_loras(p.get("loras") or [])
        w, h = int(p["width"]) // 16 * 16, int(p["height"]) // 16 * 16
        paths = list(p.get("images") or [])[:MAX_REFS]
        vl, refs = [], []
        first = None
        for n, path in enumerate(paths):
            # Morpheus' ComfyUI graph scales the first picture to the output size first (ImageScale lanczos, center crop)
            img = common.load_image(path, (w, h) if n == 0 else None)
            t = pil01(img)
            vw, vh = vl_size(img.width, img.height)
            vl.append(_to_pil(area_resize(t, vw, vh)))
            rw, rh = ref_size(img.width, img.height)
            refs.append(self.vae_encode(area_resize(t, rw, rh)))
            if n == 0:
                first = self.vae_encode(t)
        cfg = float(p.get("cfg") or 1.0)
        import os
        key = (p["prompt"], p.get("negative") or "", cfg != 1.0, w, h, tuple((x, os.path.getmtime(x)) for x in paths))
        cond, uncond = self.conditioning([p["prompt"], (p.get("negative") or "") if cfg != 1.0 else None], vl, key)
        lh, lw = h // 8, w // 8
        ref_tokens = torch.cat([pack(r) for r in refs], dim=1) if refs else None
        shapes = [(1, lh // 2, lw // 2)] + [(1, (r.shape[2] + 1) // 2, (r.shape[3] + 1) // 2) for r in refs]
        n_main = (lh // 2) * (lw // 2)

        def velocity(xp, sigma, text):
            inp = torch.cat([xp, ref_tokens], dim=1) if ref_tokens is not None else xp
            out = dit_forward(self.dit, inp.to(torch.bfloat16), text, sigma.to(torch.bfloat16).view(1), shapes, text.shape[1])
            return out[:, :n_main].float()

        def denoise(x, sigma):
            xp = pack(x)
            v = velocity(xp, sigma, cond)
            if uncond is not None:  # CFG + ComfyUI CFGNorm (strength 1): keep the norm of the conditional prediction
                vu = velocity(xp, sigma, uncond)
                comb = vu + cfg * (v - vu)
                v = comb * (v.norm(dim=-1, keepdim=True) / comb.norm(dim=-1, keepdim=True))
            return unpack(xp - v * sigma.view(-1, 1, 1), lh, lw)  # x0 = x - sigma * v

        denoise_s = float(p.get("denoise") or 1.0)
        sig = sampling.sigmas_for(self.ms, p.get("scheduler") or "simple", int(p["steps"]), denoise_s).to(common.DEVICE)
        latent = first if first is not None else torch.zeros([1, 16, lh, lw], device=common.DEVICE)
        noise = sampling.initial_noise([1, 16, lh, lw], int(p["seed"])).to(common.DEVICE)
        x = sampling.noise_scaling("flow", sig[0], noise, latent, True)
        x = sampling.sample(p.get("sampler") or "euler", denoise, x, sig, "flow", int(p["seed"]), stopped=stopped,
                            eta=float(p.get("eta", 1.0)), callback=lambda i, n, x0: emit(i, n, x0[0] * self.std[0, :, 0] + self.mean[0, :, 0]))
        lat = (x.unsqueeze(2) * self.std + self.mean).to(torch.bfloat16)
        img = self.vae.decode(lat).sample[:, :, 0]
        return common.to_pil(img)[0]

    def preview_factors(self):
        return common.flow_rgb()


def _to_pil(t01):
    from PIL import Image
    a = (t01[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a)
