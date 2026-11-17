# int8 checkpoint without the int8 matmul kernels: PyTorch's own path (the timing split showed the Kitchen HIP int8 MLP 45 % slower than GGUF's bf16 GEMM)
wgp.apply_int8_kernel_setting("disabled")
print("## hook int8 backend ->", __import__("shared.kernels.int8_backend", fromlist=["x"])._backend, flush=True)
