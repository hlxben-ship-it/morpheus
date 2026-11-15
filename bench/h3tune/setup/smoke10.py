import time, torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), round(torch.cuda.mem_get_info()[1] / 2**30, 1), "GB")
x = torch.randn(8192, 5376, device="cuda", dtype=torch.bfloat16); w = torch.randn(21504, 5376, device="cuda", dtype=torch.bfloat16)
for _ in range(3): y = torch.nn.functional.linear(x, w)
torch.cuda.synchronize(); t = time.time()
for _ in range(10): y = torch.nn.functional.linear(x, w)
torch.cuda.synchronize(); dt = (time.time() - t) / 10
print(f"bf16 linear 8192x5376->21504: {dt * 1e3:.1f} ms = {2 * 8192 * 5376 * 21504 / dt / 1e12:.1f} TFLOPS")
import triton, triton.language as tl
@triton.jit
def add(X, Y, Z, n, B: tl.constexpr):
    o = tl.program_id(0) * B + tl.arange(0, B); m = o < n
    tl.store(Z + o, tl.load(X + o, mask=m) + tl.load(Y + o, mask=m), mask=m)
a = torch.rand(100000, device="cuda"); b = torch.rand(100000, device="cuda"); c = torch.empty_like(a)
add[(triton.cdiv(100000, 1024),)](a, b, c, 100000, B=1024)
print("triton", triton.__version__, "add ok", torch.allclose(c, a + b))
q = torch.randn(1, 24, 8192, 128, device="cuda", dtype=torch.bfloat16)
for _ in range(2): o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize(); t = time.time()
for _ in range(5): o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
torch.cuda.synchronize(); print(f"sdpa 8k tokens 24 heads: {(time.time() - t) / 5 * 1e3:.0f} ms")
