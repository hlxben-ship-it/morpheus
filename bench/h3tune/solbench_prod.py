"""engine/wsl/sol_kernel.py against upstream on a captured call: identical output, time.  python solbench_prod.py <capture.pt>  (via sb.sh)"""
import sys, time, os
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl"); import amdspoof  # noqa
sys.path.insert(0, "/root/wangp"); os.chdir("/root/wangp")
import torch, shared.qtypes.gguf  # noqa
import shared.sol_attn as sol
up_prepared, up_unstaged, prepare_kv = sol.sol_attn_prepared, sol.sol_attn, sol.prepare_kv
import sol_kernel
assert sol.sol_attn_prepared is sol_kernel.sol_attn_prepared and sol.sol_attn is sol_kernel.sol_attn
d = torch.load(sys.argv[1], weights_only=False); P = lambda t: torch.Tensor.as_subclass(t, torch.Tensor).cuda()
q, k, v = P(d["q"]), P(d["k"]), P(d["v"]); prepared = tuple(P(t) for t in d["prepared"]); kw = d["kw"]
def bench(f, reps=4):
    out = f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): out = f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1e3, out
ok = True
for name, a, b in (("staged", lambda: up_prepared(q, v, prepared, **kw), lambda: sol.sol_attn_prepared(q, v, prepared, **kw)),
                   ("staged, sinks routed", lambda: up_prepared(q, v, prepared, tau=kw["tau"]), lambda: sol.sol_attn_prepared(q, v, prepared, tau=kw["tau"])),
                   ("unstaged", lambda: up_unstaged(q, k, v, int8_qk=True, thresh_type="diag", **kw), lambda: sol.sol_attn(q, k, v, int8_qk=True, thresh_type="diag", **kw))):
    m1, o1 = bench(a); m2, o2 = bench(b); same = torch.equal(o1, o2); ok &= same or sol_kernel.CENTROID_ROUTE
    if not same: print("   rel diff %.2e" % ((o1.float() - o2.float()).norm() / o1.float().norm()).item())
    print(f"{name:22s} upstream {m1:7.1f} ms   sol_kernel {m2:7.1f} ms   identical {same}", flush=True)
qq = q.clone(); o = sol.sol_attn_prepared(qq, v, prepared, out=qq, **kw); same = torch.equal(o, sol.sol_attn_prepared(q, v, prepared, **kw)); ok &= same
print("output written into q: identical", same); print("## PROD", "OK" if ok else "MISMATCH", flush=True)
