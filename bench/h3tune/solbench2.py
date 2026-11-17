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
for w, s in ((4, 1), (8, 1), (4, 2), (8, 2), (2, 1), (16, 1), (4, 3)):
    try:
        ms, out = bench(lambda: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sb, num_warps=w, num_stages=s))
        print(f"copy warps {w:2d} stages {s}: {ms:7.1f} ms  max diff {(out.float() - ref.float()).abs().max().item():.2e}", flush=True)
    except Exception as e: print(f"copy warps {w} stages {s}: {type(e).__name__} {str(e)[:120]}", flush=True)
names = {1: "no int8 Q.K dot", 2: "no P.V dot", 3: "no exp2", 4: "no K/V loads", 5: "empty exact loop"}
for a in (1, 2, 3, 4, 5):
    try:
        ms, _ = bench(lambda: sol_fast.sol_attn_prepared(q, v, prepared, tau=tau, sink_blocks=sb, abl=a))
        print(f"ablation {a} ({names[a]}): {ms:7.1f} ms  (saves {ms0 - ms:6.1f})", flush=True)
    except Exception as e: print(f"ablation {a}: {type(e).__name__} {str(e)[:200]}", flush=True)
