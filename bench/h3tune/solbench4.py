"""Head-major variants on a captured call: INT8 P.V and centroid routing. Time and error against all-exact.  (via sb.sh)"""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa
sys.path.insert(0, "/root/wangp"); sys.path.insert(0, "/mnt/c/dev/morpheus/bench/h3tune"); os.chdir("/root/wangp")
import torch, shared.qtypes.gguf  # noqa
from shared.sol_attn.saganaki import fwd
from shared.sol_attn.saganaki.preprocess import prepare_int8_kv
from shared.sol_attn.saganaki.quant import quantize_v_per_channel
from shared.sol_attn.interface import _sink_block_range
import sol_fast
d = torch.load(sys.argv[1], weights_only=False); P = lambda t: torch.Tensor.as_subclass(t, torch.Tensor).cuda()
q, k, v = P(d["q"]), P(d["k"]), P(d["v"]); prepared = tuple(P(t) for t in d["prepared"]); kw = d["kw"]; tau = kw["tau"]
sb = _sink_block_range(q.shape[1], kw.get("sink_start"), kw.get("sink_tokens", 0))
def bench(f, reps=4):
    out = f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): out = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
ref = fwd.sol_attn_prepared(q, v, prepared, tau=-100.0, sink_blocks=sb).float()
err = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
def pv8():
    r = prepare_int8_kv(k, v, int8_pv=True); return quantize_v_per_channel(v, absmax=r[-1])
t_q, vq = bench(pv8); print(f"V quantisation alone: {t_q:.1f} ms; vi {tuple(vq[0].shape)} {vq[0].dtype}, scale {tuple(vq[1].shape)}", flush=True)
for name, f in (("head-major (production)", lambda s_: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=s_, head_major=2)),
                ("+ centroid routing", lambda s_: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=s_, head_major=2, croute=True)),
                ("+ INT8 P.V", lambda s_: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=s_, head_major=2, pv8=vq)),
                ("+ INT8 P.V + centroid routing", lambda s_: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=s_, head_major=2, pv8=vq, croute=True))):
    for label, s_ in (("sinks forced", sb), ("sinks routed", (0, 0))):
        try:
            ms, out = bench(lambda: f(s_)); print(f"{name:32s} {label}: {ms:7.1f} ms   err vs all-exact {err(out):.4f}", flush=True)
        except Exception as e: print(f"{name} {label}: {type(e).__name__} {str(e)[:400]}", flush=True)
