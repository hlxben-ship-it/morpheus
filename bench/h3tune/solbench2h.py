"""Kernel experiments on a captured Sol call: python solbench2.py <capture.pt> [tau]   (via sb.sh)
Upstream kernel vs the copy in sol_fast.py: same output?, warps / stages scan, ablations (what the exact loop is made of)."""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa
sys.path.insert(0, "/root/wangp"); sys.path.insert(0, "/mnt/c/dev/morpheus/bench/h3tune"); os.chdir("/root/wangp")
import torch, shared.qtypes.gguf  # noqa
from shared.sol_attn.saganaki import fwd
from shared.sol_attn.interface import _sink_block_range
import sol_fast
d = torch.load(sys.argv[1], weights_only=False); P = lambda t: torch.Tensor.as_subclass(t, torch.Tensor).cuda()
q, v = P(d["q"]), P(d["v"]); prepared = tuple(P(t) for t in d["prepared"]); kw = d["kw"]
tau = float(sys.argv[2]) if len(sys.argv) > 2 else kw["tau"]
sb = _sink_block_range(q.shape[1], kw.get("sink_start"), kw.get("sink_tokens", 0))
def bench(f, reps=3):
    out = f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): out = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
ms0, ref = bench(lambda: fwd.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sb))
print(f"upstream kernel: {ms0:.1f} ms  (T {q.shape[1]}, tau {tau:.3f}, sink blocks {sb})", flush=True)
for hm in (0, 2, 5, 6, 2, 5, 6):
    try:
        ms, out = bench(lambda: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sb, head_major=hm), 5)
        print(f"head-major {hm} (0 no, 1 V, 2 V+K), permute copies included: {ms:7.1f} ms  max diff {(out.float() - ref.float()).abs().max().item():.2e}", flush=True)
    except Exception as e: print(f"head-major {hm}: {type(e).__name__} {str(e)[:300]}", flush=True)

def perm():
    a = v.permute(0, 2, 1, 3).contiguous(); c = prepared[4].permute(0, 2, 1, 3).contiguous(); return a
ms, _ = bench(perm); print(f"head-major permute copies of V and K8 alone: {ms:.1f} ms", flush=True)
for name, kw2 in (("sinks routed, upstream layout", dict(head_major=0)), ("sinks routed, head-major", dict(head_major=2))):
    ms, out = bench(lambda: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=(0, 0), **kw2)); print(f"{name}: {ms:7.1f} ms", flush=True)

k = P(d["k"]); from shared.sol_attn import interface as itf
ms, o1 = bench(lambda: itf.sol_attn(q, k, v, tau=tau, thresh_type="diag", sink_start=0, sink_tokens=kw.get("sink_tokens", 0), int8_qk=True))
ms2, o2 = bench(lambda: sol_fast.sol_attn_prepared(q, v, itf.prepare_kv(k, v), tau=tau, sink_blocks=sb, head_major=2))
print(f"unstaged upstream {ms:.1f} ms vs prepare_kv + head-major {ms2:.1f} ms, max diff {(o1.float() - o2.float()).abs().max().item():.2e}", flush=True)
