# Sol attention for MiniMax H3: stop forcing the conditioning rows exact in the second half of the denoising steps.
#
# WanGP's Sol attention computes a K/V block exactly where its centroid score passes a threshold and approximates it elsewhere,
# except for the "sink" blocks: the conditioning rows (start frame + reference images) are exact for every query, in every layer
# of every step. At 1920x1088 that is 16 % of all blocks and close to half of the kernel's time (Glia seed
# h3_performance_hunt_2: 440 of 1040 ms per call), and the kernel is 62 % of the denoise phase.
# Measured on the 7900 XT (Q4, 4 steps): routing the sinks like any other block in ALL steps is 22 % faster but smears the picture
# and loses the look of the reference; routing them only once the structure is there (steps 2 and 3 of 4) gives a clip that matches
# the unpatched one (same motion, sharpness 0.99, SSIM 0.83 at 720p) and takes 7 % off denoise at 720p, more at 1080p.
# Numerics change: the same render seed gives a slightly different clip than without this module.
#
# MORPHEUS_SOL_ROUTE_FROM: share of the denoising schedule after which sinks are routed (default 0.5; 1.1 or more = never, the
# unpatched behaviour). The position in the schedule is read from tau, which WanGP lowers linearly from the job's
# attention_sparsity to SOL_ATTN_TAU_END over the steps, so this holds for any step count.
import os
from models.minimax_h3 import sol_attention as _sa  # needs the WanGP root on sys.path (warm_worker.py imports this after init)
from models.minimax_h3 import pipeline as _pl

ROUTE_FROM = float(os.environ.get("MORPHEUS_SOL_ROUTE_FROM", "0.5"))
_C = _sa.MiniMaxH3SolAttention
_begin_forward = _C.begin_forward
_state = {"start": None, "last": None}
_said = set()


def progress(tau):
    """0..1 position of this forward in the denoising schedule; a tau that rises starts a new job (several forwards of one step share a tau)."""
    tau = float(tau)
    if _state["last"] is None or tau > _state["last"]: _state["start"] = tau
    _state["last"] = tau
    span = _state["start"] - float(_pl.SOL_ATTN_TAU_END)
    return (_state["start"] - tau) / span if span > 1e-6 else 0.0


def begin_forward(self, layout, device, dtype, tau, grouped_video=False):
    _begin_forward(self, layout, device, dtype, tau, grouped_video)
    if self.enabled and self.sink_tokens and progress(tau) > ROUTE_FROM:
        self.sink_tokens = 0
        _sa._graph_state = (self.tau, 0)  # what CUDA graphs key on
        if "routed" not in _said:
            _said.add("routed"); print(f"[Morpheus] Sol attention: conditioning rows routed from {ROUTE_FROM:g} of the schedule on.", flush=True)


_C.begin_forward = begin_forward

print("## sol_route loaded", flush=True)
