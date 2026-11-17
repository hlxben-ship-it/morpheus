"""Reference latent cache for H3-Image (Glia seed h3_image_cache_warmup).

Every H3-Image job VAE-encodes each reference again: ~6.5 s per 1664x2496 reference on the RX 7900 XT (bench img_prep2), so a
repeat edit with 2 refs spends ~13 s re-making latents it made a minute ago. The encoder is deterministic, so the same pixels at the
same size give the same latent: keep the last few, keyed by a hash of the input tensor, per loaded pipeline. No torch import here,
so the logic is testable on Windows; the worker passes the hash function.
"""
from collections import OrderedDict

MAX_ENTRIES = 8  # a latent of a 4 MP reference is a few MB on the CPU


class LatentCache:
    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def get_or_make(self, key, make, copy=lambda v: v):
        if key in self.entries:
            self.entries.move_to_end(key); self.hits += 1
            return copy(self.entries[key])
        value = make(); self.misses += 1
        self.entries[key] = copy(value)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return value


def wrap(encode, key_of, use_for, cache: LatentCache, copy=lambda v: v):
    """encode(self, video, keep_all_latents=False) cached when use_for(self) is true; key_of(self, video, keep_all_latents) names it.
    The key must include the pipeline instance (another loaded model = another VAE)."""
    def wrapped(self, video, keep_all_latents=False):
        if not use_for(self):
            return encode(self, video, keep_all_latents=keep_all_latents)
        return cache.get_or_make(key_of(self, video, keep_all_latents), lambda: encode(self, video, keep_all_latents=keep_all_latents), copy)
    return wrapped


def install(pipeline_cls, cache: LatentCache = None):
    """Wrap pipeline_cls._encode_video (WanGP MiniMaxH3Pipeline) for H3-Image pipelines. Returns the cache (hits / misses)."""
    import hashlib
    import torch
    cache = cache or LatentCache()

    def key_of(self, video, keep_all_latents):
        t = video.detach().contiguous().cpu()
        d = hashlib.sha256(); d.update(f"{t.dtype}{tuple(t.shape)}".encode()); d.update(memoryview(t.view(torch.uint8).reshape(-1).numpy()))
        return id(self), id(getattr(self, "vae", None)), bool(keep_all_latents), d.digest()

    pipeline_cls._encode_video = wrap(pipeline_cls._encode_video, key_of, lambda self: bool(getattr(self, "h3_image", False)), cache,
                                      copy=lambda v: v.clone() if hasattr(v, "clone") else v)
    return cache
