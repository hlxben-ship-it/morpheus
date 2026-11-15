# Sol's pointer kernels autotune over num_warps 4 / 8 only (NVIDIA: 32-thread warps). On RDNA3 (64-wide wavefronts) 4 beat 8 by 2x
# in the cached timings, so offer smaller ones too.
import triton
from shared.sol_attn.saganaki import fwd as _fwd
for _k in (_fwd._forward_int8_ptr, _fwd._forward_ptr):
    _have = {(c.num_warps, c.num_stages) for c in _k.configs}
    for _w, _s in ((2, 1), (2, 2), (1, 1), (1, 2), (2, 3)):
        if (_w, _s) not in _have: _k.configs.append(triton.Config({}, num_warps=_w, num_stages=_s))
print("## hook sol_warps:", sorted((c.num_warps, c.num_stages) for c in _fwd._forward_int8_ptr.configs), flush=True)
