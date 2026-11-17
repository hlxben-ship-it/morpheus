# Cold-start probe (seed h3_image_cache_warmup): per-call block times of the first two forwards, and VAE decode calls.
import time as _t, torch as _torch
from models.minimax_h3 import transformer as _tr
_calls = []
_ob = _tr.DiTBlock.forward
def _blk(*a, **k):
    if len(_calls) >= 200: return _ob(*a, **k)
    _torch.cuda.synchronize(); t = _t.perf_counter(); r = _ob(*a, **k); _torch.cuda.synchronize(); _calls.append(_t.perf_counter() - t); return r
_tr.DiTBlock.forward = _blk
def _fc_report():
    f = [round(x * 1000) for x in _calls]
    print("## FIRST step1 blocks ms", f[:100], flush=True); print("## FIRST step2 blocks ms", f[100:200], flush=True)
after.append(_fc_report)
