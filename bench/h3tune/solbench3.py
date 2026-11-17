"""int8 P.V in the Sol kernel: time and error against all-exact, on a captured call.  python solbench3.py <capture.pt>  (via sb.sh)"""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa
sys.path.insert(0, "/root/wangp"); os.chdir("/root/wangp")
import torch, shared.qtypes.gguf  # noqa
from shared.sol_attn.saganaki import fwd
from shared.sol_attn.interface import _sink_block_range
d = torch.load(sys.argv[1], weights_only=False); P = lambda t: torch.Tensor.as_subclass(t, torch.Tensor).cuda()
q, k, v = P(d["q"]), P(d["k"]), P(d["v"]); kw = d["kw"]; tau = kw["tau"]
sb = _sink_block_range(q.shape[1], kw.get("sink_start"), kw.get("sink_tokens", 0))
def bench(f, reps=3):
    out = f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): out = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
_, ref = bench(lambda: fwd.sol_attn(q, k, v, tau=-100.0, int8_qk=True, sink_blocks=sb), 1); ref = ref.float()
err = lambda o: ((o.float() - ref).norm() / ref.norm()).item()
for name, kwargs in (("bf16 P.V, sinks forced", dict(tau=tau, sink_blocks=sb)), ("int8 P.V, sinks forced", dict(tau=tau, sink_blocks=sb, int8_pv=True)),
                     ("bf16 P.V, sinks routed", dict(tau=tau)), ("int8 P.V, sinks routed", dict(tau=tau, int8_pv=True)),
                     ("int8 P.V, all exact", dict(tau=-100.0, sink_blocks=sb, int8_pv=True))):
    ms, out = bench(lambda: fwd.sol_attn(q, k, v, int8_qk=True, **kwargs))
    print(f"{name:26s} {ms:7.1f} ms (with K/V preparation)  err vs all-exact bf16 {err(out):.4f}", flush=True)
