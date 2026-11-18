"""int8 kernel backend of the warm worker (Glia seed h3_overnight_master, Q5, 2026-11-18).

WanGP's "auto" lands on Comfy Kitchen HIP on gfx1100. Triton measured faster: H3-Image 2 refs 127.5 -> 119.1 s warm, H3 int8
video 250 -> 229 s, and its pictures sit closer to plain PyTorch (Kitchen is the numeric outlier). MORPHEUS_INT8_KERNELS overrides:
auto | kitchen | triton | wgpkernels | disabled (PyTorch). Anything else falls back to the default."""
DEFAULT = "triton"
CHOICES = ("auto", "kitchen", "triton", "wgpkernels", "disabled")


def int8_kernels(env):
    value = (env.get("MORPHEUS_INT8_KERNELS") or "").strip().lower()
    return value if value in CHOICES else DEFAULT
