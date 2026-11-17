# Share of GGUF dequantisation (after gguf_triton) in a job: synchronised wall clock around shared.qtypes.gguf._gguf_dequantize_tensor.
import time as _t, torch as _torch
import shared.qtypes.gguf as _g
_o = _g._gguf_dequantize_tensor; _acc = {"s": 0.0, "n": 0, "bytes": 0}
def _timed(*a, **k):
    _torch.cuda.synchronize(); t = _t.perf_counter(); r = _o(*a, **k); _torch.cuda.synchronize()
    _acc["s"] += _t.perf_counter() - t; _acc["n"] += 1
    try: _acc["bytes"] += r.numel() * r.element_size()
    except Exception: pass
    return r
_g._gguf_dequantize_tensor = _timed
def _rep(): print("## TIMING dequant %.1fs n=%d dense %.1f GB written (all jobs)" % (_acc["s"], _acc["n"], _acc["bytes"] / 2**30), flush=True)
after = list(globals().get("after", [])) + [_rep]
