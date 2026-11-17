# Reference latent cache (seed h3_image_cache_warmup), the worker's module installed into the bench session.
import sys as _s; _s.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl")
import ref_latents as _rl
from models.minimax_h3 import pipeline as _pl
_rlc = _rl.install(_pl.MiniMaxH3Pipeline)
after.append(lambda: print("## LATENTS hits", _rlc.hits, "misses", _rlc.misses, flush=True))
