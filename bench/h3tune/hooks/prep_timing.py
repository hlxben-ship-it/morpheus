# Split the phase before denoising (seed h3_image_cache_warmup): ref VAE encode, prompt encode (cache hit or not), text encoder, other.
import time as _t, collections as _c, torch as _torch
from models.minimax_h3 import pipeline as _pl
_P = _c.defaultdict(float); _PN = _c.defaultdict(int)
def _pw(owner, attr, key):
    orig = getattr(owner, attr)
    def f(*a, **k):
        _torch.cuda.synchronize(); t = _t.perf_counter()
        try: return orig(*a, **k)
        finally: _torch.cuda.synchronize(); _P[key] += _t.perf_counter() - t; _PN[key] += 1
    setattr(owner, attr, f)
_pw(_pl.MiniMaxH3Pipeline if hasattr(_pl, "MiniMaxH3Pipeline") else [v for v in vars(_pl).values() if isinstance(v, type) and hasattr(v, "_encode_prompt")][0], "_encode_video", "vae_encode")
_cls = [v for v in vars(_pl).values() if isinstance(v, type) and hasattr(v, "_encode_prompt")][0]
_pw(_cls, "_encode_prompt", "encode_prompt")
_pw(_cls, "_prepare_image_reference", "ref_resize")
_pw(_cls, "generate", "generate")
def _prep_report():
    for k in ("generate", "vae_encode", "encode_prompt", "ref_resize"):
        print("## PREP %-14s %7.1fs n=%d" % (k, _P[k], _PN[k]), flush=True)
after.append(_prep_report)
