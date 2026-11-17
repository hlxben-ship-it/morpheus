"""Micro-benchmark of the Sol attention kernel on captured inputs (hooks/capture.py).
   python solbench.py /root/solbench/q4_720s.pt [tau ...]     run from /root/wangp with the venv10 env."""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa
sys.path.insert(0, "/root/wangp"); os.chdir("/root/wangp")
import torch
from shared.sol_attn.saganaki import fwd
from shared.sol_attn.interface import _sink_block_range
import shared.qtypes.gguf  # noqa
d = torch.load(sys.argv[1], weights_only=False); dev = "cuda"
P = lambda t: torch.Tensor.as_subclass(t, torch.Tensor)
d = {**d, "q": P(d["q"]), "k": P(d["k"]), "v": P(d["v"]), "prepared": tuple(P(t) for t in d["prepared"])}
q, k, v = d["q"].to(dev), d["k"].to(dev), d["v"].to(dev); prepared = tuple(t.to(dev) for t in d["prepared"]); kw = d["kw"]
B, T, H, D = q.shape; NT = (T + 63) // 64
print("q", tuple(q.shape), "kw", kw, "prepared", [(tuple(t.shape), str(t.dtype)) for t in prepared])
kc, vc, mean_kc, var_kc, k8, k_scale = prepared
sb = _sink_block_range(T, kw.get("sink_start"), kw.get("sink_tokens", 0))
sl = D ** -0.5 * 1.4426950408889634
def routed_fraction(tau):
    pad = NT * 64 - T
    qf = torch.nn.functional.pad(q[0].float(), (0, 0, 0, 0, 0, pad)).view(NT, 64, H, D).sum(1)
    cnt = torch.full((NT, 1, 1), 64.0, device=dev); cnt[-1] = 64 - pad
    qc = qf / cnt                                                     # [NT,H,D]
    s = torch.einsum("qhd,khd->hqk", qc, kc.view(NT, H, D).float()) * sl
    thr = (qc * mean_kc.view(1, H, D).float()).sum(-1) * sl + tau * torch.sqrt(((qc * qc) * var_kc.view(1, H, D).float()).sum(-1) * sl * sl + 1e-6)
    r = s > thr.t()[:, :, None]
    i = torch.arange(NT, device=dev)
    r |= ((i[:, None] - i[None, :]).abs() <= 1)[None]
    r[:, :, sb[0]:sb[1]] = True
    return r.float().mean().item(), r.float().mean((1, 2))
def run(tau, reps=3):
    out = torch.empty_like(q)
    f = lambda: fwd.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sb, out=out)
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
print("sink blocks", sb, "of", NT)
def run2(tau, sink, reps=2):
    out = torch.empty_like(q)
    f = lambda: fwd.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sink, out=out)
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
_, ref = run2(-100.0, sb, 1); ref = ref.float(); gen = slice(sb[1] * 64, None)
def err(o):
    o = o.float(); e = lambda a, b: ((a - b).norm() / b.norm()).item()
    return e(o, ref), e(o[:, gen], ref[:, gen])
print("columns: tau, sink mode, kernel ms, rel. error vs all-exact (all rows / generated rows)")
for spec in sys.argv[2:]:
    tau, mode = spec.split(":") if ":" in spec else (spec, "sink")
    sink = sb if mode == "sink" else (0, 0)
    ms, out = run2(float(tau), sink); ea, eg = err(out)
    print(f"tau {float(tau):6.2f} {mode:7s} {ms:7.1f} ms   err {ea:.4f} / {eg:.4f}", flush=True)
