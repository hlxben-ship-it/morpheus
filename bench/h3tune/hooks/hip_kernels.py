# WanGP's own fused Triton kernels (norm + RoPE, SwiGLU, norm modulation, Conv3D, GroupNorm + SiLU) are gated to NVIDIA:
# dispatch.capability() is None on HIP. Report RDNA3 as Ampere (8, 6), the same lie amdspoof tells the Sol kernels.
from shared.kernels import dispatch
dispatch.capability = lambda device_index=None: (8, 6)
print("## hook hip_kernels: dispatch.capability -> (8, 6)", flush=True)
