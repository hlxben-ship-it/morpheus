# Overnight master Q5: force WanGP's int8 kernel backend (auto = Comfy Kitchen HIP here) to var["int8_kernels"]: triton | disabled (PyTorch) | wgpkernels | kitchen.
_sel = var["int8_kernels"]
print("## int8_backend set", _sel, "->", wgp.apply_int8_kernel_setting(_sel), flush=True)
