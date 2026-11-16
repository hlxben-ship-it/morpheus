"""SDXL (JANKU and friends) in our own loop: diffusers loads the UNet, VAE and both CLIPs from the single checkpoint;
prompt parsing, conditioning, sampling, img2img and the hires pass are ours and follow ComfyUI (CLIPTextEncode on SDXL,
KSampler, LatentUpscaleBy) so the same seed gives the same picture.
"""
import re

import torch
import torch.nn.functional as F

from . import common, sampling

SCALE = 0.13025  # SDXL latent scaling factor
MAX_TOKENS = 75   # per CLIP chunk, plus start and end token


# ---- prompt weights: ComfyUI's "(word:1.2)" / "(word)" syntax ----

def _parse_parens(s):
    out, cur, depth = [], "", 0
    for c in s:
        if c == "(":
            if depth == 0:
                if cur: out.append(cur)
                cur = "("
            else:
                cur += c
            depth += 1
        elif c == ")":
            depth -= 1
            cur += c
            if depth == 0:
                out.append(cur); cur = ""
        else:
            cur += c
    if cur: out.append(cur)
    return out


def token_weights(text, weight=1.0):
    """ComfyUI sd1_clip.token_weights: [(text, weight)]; nested parentheses multiply by 1.1, "(x:w)" sets w."""
    out = []
    for x in _parse_parens(text):
        w = weight
        if len(x) >= 2 and x[0] == "(" and x[-1] == ")":
            x = x[1:-1]
            xx = x.rfind(":")
            w *= 1.1
            if xx > 0:
                try:
                    w = float(x[xx + 1:]); x = x[:xx]
                except ValueError:
                    pass
            out += token_weights(x, w)
        else:
            out.append((x, weight))
    return out


def _unescape(t):
    return t.replace("\0\1", ")").replace("\0\2", "(")


def tokenize(tokenizer, text, pad_with_end):
    """ComfyUI SDTokenizer.tokenize_with_weights: words are kept whole in a 75-token chunk when they fit."""
    start, end = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = end if pad_with_end else 0
    text = text.replace("\\)", "\0\1").replace("\\(", "\0\2")
    words = []
    for seg, w in token_weights(text):
        for word in _unescape(seg).split(" "):
            if not word:
                continue
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if ids:
                words.append([(t, w) for t in ids])
    batches, batch = [], []
    for word in words:
        while word:
            if len(word) + len(batch) > MAX_TOKENS:
                remain = MAX_TOKENS - len(batch)
                if len(word) > MAX_TOKENS // 2:  # ComfyUI splits long words; short ones go whole into the next chunk
                    batch.extend(word[:remain]); word = word[remain:]
                batches.append(batch); batch = []
            else:
                batch.extend(word); word = []
    if batch or not batches:
        batches.append(batch)
    out = []
    for b in batches:
        b = [(start, 1.0)] + b + [(end, 1.0)]
        b += [(pad, 1.0)] * (MAX_TOKENS + 2 - len(b))
        out.append(b)
    return out


class SDXL:
    family = "sdxl"

    def __init__(self, path):
        from diffusers import StableDiffusionXLPipeline
        self.path = path
        pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=torch.float16)
        self.unet = pipe.unet.to(common.DEVICE).eval()
        self.vae = pipe.vae.to(common.DEVICE, torch.bfloat16).eval()  # bf16: SDXL's VAE overflows in fp16
        self.te1 = pipe.text_encoder.to(common.DEVICE).eval()
        self.te2 = pipe.text_encoder_2.to(common.DEVICE).eval()
        self.tok1, self.tok2 = pipe.tokenizer, pipe.tokenizer_2
        self.pipe = pipe  # keeps the LoRA loader
        self.loras = ()
        self.ms = sampling.Discrete()

    def gb(self):
        return 7.0

    # ---- LoRA ----
    def set_loras(self, loras):
        key = tuple((l["path"], float(l["strength"])) for l in loras)
        if key == self.loras:
            return
        if self.loras:
            self.pipe.unload_lora_weights()
        names = []
        for n, l in enumerate(loras):
            name = f"l{n}"
            self.pipe.load_lora_weights(l["path"], adapter_name=name)
            names.append(name)
        if names:
            self.pipe.set_adapters(names, adapter_weights=[float(l["strength"]) for l in loras])
        self.loras = key

    # ---- conditioning ----
    @torch.no_grad()
    def _encode_chunks(self, chunks1, chunks2):
        """Each chunk: penultimate hidden states of both CLIPs, concatenated (2048). Pooled = CLIP-G of the first chunk."""
        zs, pooled = [], None
        for c1, c2 in zip(chunks1, chunks2):
            t1 = torch.tensor([[t for t, _ in c1]], device=common.DEVICE)
            t2 = torch.tensor([[t for t, _ in c2]], device=common.DEVICE)
            h1 = self.te1(t1, output_hidden_states=True).hidden_states[-2]
            o2 = self.te2(t2, output_hidden_states=True)
            h2 = o2.hidden_states[-2]
            if pooled is None:
                pooled = o2.text_embeds
            zs.append(torch.cat([h1, h2], dim=-1))
        return zs, pooled

    @torch.no_grad()
    def encode(self, text):
        c1 = tokenize(self.tok1, text, pad_with_end=True)
        c2 = tokenize(self.tok2, text, pad_with_end=False)
        n = max(len(c1), len(c2))  # both tokenizers split the same words; guard anyway
        c1 += [c1[-1]] * (n - len(c1)); c2 += [c2[-1]] * (n - len(c2))
        zs, pooled = self._encode_chunks(c1, c2)
        if any(w != 1.0 for c in c1 for _, w in c):  # ComfyUI: weight relative to the empty prompt
            e1, e2 = tokenize(self.tok1, "", True), tokenize(self.tok2, "", False)
            (ze,), _ = self._encode_chunks(e1, e2)
            for i, c in enumerate(c1):
                w = torch.tensor([wt for _, wt in c], device=common.DEVICE, dtype=zs[i].dtype).view(1, -1, 1)
                zs[i] = (zs[i] - ze) * w + ze
        return torch.cat(zs, dim=1), pooled

    # ---- sampling ----
    def _denoiser(self, cond, uncond, cfg, time_ids, rescale=0.0):
        (c, cp), (u, up) = cond, uncond
        if c.shape[1] != u.shape[1]:  # ComfyUI repeats to the lcm length; attention over repeats is unchanged
            import math
            L = math.lcm(c.shape[1], u.shape[1])
            c = c.repeat(1, L // c.shape[1], 1); u = u.repeat(1, L // u.shape[1], 1)
        ctx = torch.cat([c, u]); pooled = torch.cat([cp, up]); tids = torch.cat([time_ids, time_ids])

        def denoise(x, sigma):
            s = sigma[0]
            c_in = 1.0 / torch.sqrt(s ** 2 + 1)
            t = self.ms.timestep(s.cpu()).float().to(common.DEVICE)
            xin = torch.cat([x, x]) * c_in
            eps = self.unet(xin.half(), t.expand(2), encoder_hidden_states=ctx,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": tids}).sample.float()
            x0c, x0u = (x - eps[:1] * s), (x - eps[1:] * s)
            if cfg == 1.0:
                return x0c
            out = x0u + (x0c - x0u) * cfg
            if rescale:  # CFG rescale (ComfyUI RescaleCFG): match the std of the conditional prediction
                xc, xr = (x - x0c) / s, (x - out) / s  # eps space
                ro = xr * (xc.std(dim=(1, 2, 3), keepdim=True) / xr.std(dim=(1, 2, 3), keepdim=True))
                xr = rescale * ro + (1 - rescale) * xr
                out = x - xr * s
            return out
        return denoise

    @torch.no_grad()
    def generate(self, p, emit, stopped):
        """p: prompt, negative, width, height, seed, steps, cfg, sampler, scheduler, denoise, init (path|None),
        hires (factor, 1 = off), hires_denoise, cfg_rescale, eta, loras. emit(step, total, x0) per step. -> PIL image."""
        self.set_loras(p.get("loras") or [])
        w, h = int(p["width"]), int(p["height"])
        cond = self.encode(p["prompt"]); uncond = self.encode(p.get("negative") or "")

        def run(latent, width, height, steps, denoise, seed, offset, total):
            sig = sampling.sigmas_for(self.ms, p["scheduler"], steps, denoise).to(common.DEVICE)
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=common.DEVICE, dtype=torch.float16)
            fn = self._denoiser(cond, uncond, float(p["cfg"]), time_ids, float(p.get("cfg_rescale") or 0))
            noise = sampling.initial_noise(latent.shape, seed).to(common.DEVICE)
            max_den = denoise >= 0.9999 and abs(float(sig[0]) - float(self.ms.sigma_max)) < 1e-5
            x = sampling.noise_scaling("eps", sig[0], noise, latent, max_den)
            return sampling.sample(p["sampler"], fn, x, sig, "eps", seed, eta=float(p.get("eta", 1.0)), stopped=stopped,
                                   callback=lambda i, n, x0: emit(offset + i, total, x0[0]))

        seed = int(p["seed"]); steps = int(p["steps"])
        denoise = float(p.get("denoise") or 1.0)
        if p.get("init"):
            img = common.load_image(p["init"], (w, h))
            latent = self.vae.encode(common.pil_to_tensor(img).to(common.DEVICE, torch.bfloat16)).latent_dist.mean.float() * SCALE
        else:
            latent = torch.zeros([1, 4, h // 8, w // 8], device=common.DEVICE)
            denoise = 1.0
        hires = float(p.get("hires") or 1.0)
        h_steps = max(10, steps // 2) if hires > 1 else 0
        first = steps if denoise >= 0.9999 else steps  # sigmas_for keeps `steps` sigmas either way
        total = first + h_steps
        x = run(latent, w, h, steps, denoise, seed, 0, total)
        if hires > 1:  # ComfyUI LatentUpscaleBy nearest-exact, then KSampler at hires_denoise with the same seed
            H, W = round(x.shape[2] * hires), round(x.shape[3] * hires)
            x = F.interpolate(x, size=(H, W), mode="nearest-exact")
            x = run(x, W * 8, H * 8, h_steps, float(p.get("hires_denoise") or 0.5), seed, first, total)
        img = self.vae.decode((x / SCALE).to(torch.bfloat16)).sample
        return common.to_pil(img)[0]

    def preview_factors(self):
        return common.SDXL_RGB
