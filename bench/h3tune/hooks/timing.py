# Wall-clock split of the denoise phase (cuda.synchronize around each part): where does a step go?
# block = DiTBlock.forward; attn / mlp inside it; sol_fwd / sol_prep = the Sol kernel and its K/V preparation inside attn.
# "outside" (denoise minus blocks) = weight streaming (mmgp hooks sit outside the class forward), input prep, sampler.
import time as _time, collections as _c
import torch as _torch
from models.minimax_h3 import transformer as _tr
import shared.sol_attn as _sol
_T = _c.defaultdict(float); _N = _c.defaultdict(int); _info = {}
_t_first = [None]; _t_last = [0.0]
def _wrap(owner, attr, key, pre=None):
    orig = getattr(owner, attr)
    def timed(*a, **k):
        if pre: pre(*a, **k)
        _torch.cuda.synchronize(); t = _time.perf_counter()
        if _t_first[0] is None: _t_first[0] = t
        try: return orig(*a, **k)
        finally:
            _torch.cuda.synchronize(); e = _time.perf_counter(); _T[key] += e - t; _N[key] += 1; _t_last[0] = e
    setattr(owner, attr, timed)
def _seq(self, x_list, *a, **k):
    try: _info.setdefault("seq", set()).add(int(x_list[0].shape[0]))
    except Exception: pass
_wrap(_tr.DiTBlock, "forward", "block")
_wrap(_tr.Attention, "forward", "attn", pre=_seq)
_wrap(_tr.MLP, "forward", "mlp")
for _cls, _key in (("MiniMaxH3Model", "model"), ("InputPreparation", "input_prep"), ("FinalLayer", "final"), ("TokenRefiner", "refiner")):
    if hasattr(_tr, _cls): _wrap(getattr(_tr, _cls), "forward", _key)
_wrap(_sol, "sol_attn_prepared", "sol_fwd")
_wrap(_sol, "prepare_kv", "sol_prep")
_wrap(_sol, "sol_attn", "sol_unstaged")
def _report():
    span = _t_last[0] - (_t_first[0] or 0)
    print("## TIMING seq_len", sorted(_info.get("seq", [])), "span first..last block %.1fs (all jobs)" % span)
    for k in ("model", "input_prep", "refiner", "final", "block", "attn", "sol_fwd", "sol_prep", "sol_unstaged", "mlp"):
        if _N[k]: print("## TIMING %-12s %8.1fs  n=%5d  avg %7.1f ms" % (k, _T[k], _N[k], 1e3 * _T[k] / _N[k]))
    print("## TIMING block-attn-mlp %.1fs, attn-sol %.1fs" % (_T["block"] - _T["attn"] - _T["mlp"], _T["attn"] - _T["sol_fwd"] - _T["sol_prep"] - _T["sol_unstaged"]), flush=True)
after.append(_report)
