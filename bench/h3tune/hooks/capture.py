# Save the inputs of one Sol attention call (q, k, v, prepared K/V, tau, sinks) for the kernel micro-benchmark (solbench.py).
import os as _os, torch as _torch
import shared.sol_attn as _sol
_n = [0]; _k = [None]; _want = int(var.get("capture_call", 100)); _dst = "/root/solbench/%s_%s.pt" % (var["model"], var["clip"])
_os.makedirs("/root/solbench", exist_ok=True)
_p0, _f0 = _sol.prepare_kv, _sol.sol_attn_prepared
def _prep(k, v):
    if _n[0] + 1 == _want: _k[0] = k.detach().cpu().clone()
    return _p0(k, v)
def _fwd(q, v, prepared, **kw):
    _n[0] += 1
    if _n[0] == _want:
        _torch.save({"q": q.cpu().clone(), "k": _k[0], "v": v.cpu().clone(), "prepared": tuple(t.cpu() for t in prepared),
                     "kw": {k: x for k, x in kw.items() if k != "out"}}, _dst)
        print("## captured sol call", _want, tuple(q.shape), kw.get("tau"), kw.get("sink_start"), kw.get("sink_tokens"), "->", _dst, flush=True)
    return _f0(q, v, prepared, **kw)
_sol.prepare_kv, _sol.sol_attn_prepared = _prep, _fwd
