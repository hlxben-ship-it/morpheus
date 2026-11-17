# Sol attention: do not force the conditioning rows ("sink" K/V blocks: start frame + reference images) exact for every query
# in the chosen steps / layers; there they are routed like any other block. Full routing (all steps, all layers) is 22 % faster
# at 1080p but smears the picture (2026-11-17), hence the partial forms.
#   var["nosink_steps"]: step indices where sinks are routed (default: all);  var["nosink_layers"]: [first, last) Sol-call index
#   within a forward (default: all);  var["sol_tau_shift"]: added to tau.
from models.minimax_h3 import sol_attention as _sa
_C = _sa.MiniMaxH3SolAttention
_bf, _att, _call = _C.begin_forward, _C.attention, _C.__call__
_shift = float(var.get("sol_tau_shift", 0.0)); _steps = var.get("nosink_steps"); _lay = var.get("nosink_layers")
_tsteps = var.get("sol_tau_steps", {})   # {"2": 0.5}: tau shift per step index
_st = {"step": -1, "tau": None, "layer": 0, "sink": 0}
def _begin_forward(self, layout, device, dtype, tau, grouped_video=False):
    _st["step"] = 0 if _st["tau"] is None or float(tau) >= _st["tau"] else _st["step"] + 1   # tau falls from step to step
    _st["tau"] = float(tau); _st["layer"] = 0
    _bf(self, layout, device, dtype, float(tau) + _shift + float(_tsteps.get(str(_st["step"]), 0.0)), grouped_video)
    _st["sink"] = self.sink_tokens
def _pick(self):
    i = _st["layer"]; _st["layer"] += 1
    routed = (_steps is None or _st["step"] in _steps) and (_lay is None or _lay[0] <= i < _lay[1])
    self.sink_tokens = 0 if routed else _st["sink"]
def _attention(self, *a, **k):
    _pick(self); return _att(self, *a, **k)
def _callf(self, qkv_list, use_sol):
    if use_sol: _pick(self)
    return _call(self, qkv_list, use_sol)
_C.begin_forward, _C.attention, _C.__call__ = _begin_forward, _attention, _callf
print("## hook sol_nosink: sink blocks routed in steps", _steps, "layers", _lay, "tau shift", _shift, flush=True)
