wgp.apply_int8_kernel_setting("triton")
print("## hook int8 backend ->", __import__("shared.kernels.int8_backend", fromlist=["x"])._backend, flush=True)
