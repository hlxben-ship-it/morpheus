# Make NVIDIA-gated Triton paths pick their Ampere/pointer variants on RDNA3 (no TMA, no fp8).
import torch
if torch.version.hip:
    _orig = torch.cuda.get_device_capability
    def _cap(device=None): return (8, 6)
    torch.cuda.get_device_capability = _cap
