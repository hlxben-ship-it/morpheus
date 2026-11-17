# Sol attention for MiniMax H3 on RDNA3: the exact K/V blocks read from a head-major copy of K and V.
#
# The kernel below is WanGP's pointer INT8 forward (shared/sol_attn/saganaki/fwd.py::_forward_int8_ptr, inline-Q, BF16 P.V; bundled
# there from ComfyUI-sol-attn v0.6.2, Apache-2.0) with one change: the INT8 K blocks and their scales are addressed head-major
# ([B, H, T, D]) and V is passed as a head-major copy through its strides. Upstream keeps K and V token-major ([B, T, H, D]): the 64
# rows of one head's block then lie 7 KB (K) / 14 KB (V) apart, and the exact loop spends its time loading them.
# Same arithmetic, same order: with CENTROID_ROUTE off the output is bit-identical to upstream (max diff 0 on a captured 1080p call, Glia seed
# h3_performance_hunt_2, 2026-11-17). Kernel time at 1920x1088 (53 490 tokens, 56 heads): 1040 -> 864 ms per call with the two
# copies (9 ms) included. Costs VRAM for the copies: 1.1 GB at 1080p / 73 frames.
# Addressing matters as much as the layout (the same layout with K through runtime strides, or V through a constant formula, is no
# faster or slower than upstream), so do not "clean up" the index expressions without re-running bench/h3tune/solbench2h.py.
# Fixed num_warps=4, num_stages=1: what upstream's autotune picks on the 7900 XT at every size measured.
#
# MORPHEUS_SOL_KERNEL=0 leaves upstream's kernel in place.
import os
import torch, triton, triton.language as tl
import shared.sol_attn as _sol  # needs the WanGP root on sys.path (warm_worker.py imports this after init)
from shared.sol_attn import interface as _itf
from shared.sol_attn.saganaki import fwd as _fwd
from shared.sol_attn.saganaki.quant import _round_to_int8

BLOCK, GROUP = _fwd.BLOCK, _fwd.GROUP
# Route score from the block's centroid instead of the column sums of the 64 x 32 score tile: the same quantity up to rounding (a
# borderline block may flip: relative output difference ~4e-6 per call, error against all-exact unchanged at 0.0349), 6 % less
# kernel time on that call. Over a whole job it gives a different clip (SSIM 0.53 at 720p, other pose, tattoo drifts): off by default,
# MORPHEUS_SOL_CENTROID_ROUTE=1 turns it on for every job (1080p Q4: 271 -> 250 s); the Playground's routing switch does it per job.
CENTROID_ROUTE = os.environ.get("MORPHEUS_SOL_CENTROID_ROUTE", "0") == "1"
CENTROID_ROUTE_DEFAULT = CENTROID_ROUTE  # warm_worker.py sets CENTROID_ROUTE per job (the Playground's "Fast" routing)
_NS = ['T', 'NT', 'sink_start', 'sink_end', 'sink_q_start', 'sink_q_end', 'sq_b', 'sq_t', 'sq_h', 'sv_b', 'sv_t', 'sv_h']


@triton.jit(do_not_specialize=_NS, do_not_specialize_on_alignment=_NS)
def _forward_hm(q_ptr, v_ptr, kc_ptr, vc_ptr, ki_ptr, k_scale, kc_mean_ptr, kc_var_ptr, o_ptr, scale, tau,
                sink_start, sink_end, sink_q_start, sink_q_end, T, sq_b, sq_t, sq_h, sv_b, sv_t, sv_h,
                H: tl.constexpr, D: tl.constexpr, NT, BV: tl.constexpr, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, CENTROID_ROUTE: tl.constexpr):
    v_tile, q_block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    d_offsets = tl.arange(0, D)
    bv_offsets = v_tile * BV + tl.arange(0, BV)
    q_start = q_block * BLOCK_SIZE
    q_rows = q_start + token_offsets
    q_valid = q_rows < T
    q = tl.load(q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d_offsets[None, :], mask=q_valid[:, None], other=0.0)
    q_fp32 = q.to(tl.float32)
    q_amax = tl.max(tl.abs(q_fp32), axis=1)
    qs = tl.maximum(q_amax / 127.0, 1e-8)
    qi = _round_to_int8(q_fp32 / qs[:, None])
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)
    output = tl.zeros([BLOCK_SIZE, BV], dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    row_max = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
    centroid = tl.sum(q.to(tl.float32), axis=0) / q_len
    mean_kc = tl.load(kc_mean_ptr + batch_head * D + d_offsets)
    var_kc = tl.load(kc_var_ptr + batch_head * D + d_offsets)
    threshold_mean = tl.sum(centroid * mean_kc, axis=0) * scale_log2
    threshold_variance = tl.sum(centroid * centroid * var_kc, axis=0) * (scale_log2 * scale_log2)
    route_threshold = threshold_mean + tau * tl.sqrt(tl.maximum(threshold_variance, 0.0) + 1.0e-6)
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)

    for group_start in range(0, NT, GROUP_SIZE):
        block_indices = group_start + group_offsets
        valid = block_indices < NT
        kc = tl.load(kc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + d_offsets[None, :], mask=valid[:, None], other=0.0)
        vc = tl.load(vc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + bv_offsets[None, :], mask=valid[:, None], other=0.0)
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        if CENTROID_ROUTE:  # the mean over the rows of q.kc is centroid.kc: the route score without the column sums, in FP32
            routed = ((tl.sum(centroid[None, :] * kc.to(tl.float32), axis=1) * scale_log2 > route_threshold) | (tl.abs(q_block - block_indices) <= 1) | sink_kv) & valid
        else:
            routed = ((tl.sum(scores, axis=0) / q_len > route_threshold) | (tl.abs(q_block - block_indices) <= 1) | sink_kv) & valid
        exact = tl.where(q_in_sink, valid, routed)
        approximate = valid & ~exact
        approximate_scores = tl.where(approximate[None, :], scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(approximate_scores, axis=1))
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        approximate_probability = tl.where(approximate[None, :], tl.math.exp2(approximate_scores - new_max[:, None]), 0.0)
        output = output * alpha[:, None] + tl.dot(approximate_probability.to(vc.dtype), vc)
        lengths = tl.where(block_indices == NT - 1, tail_length, BLOCK_SIZE).to(tl.float32)
        row_sum = row_sum * alpha + tl.sum(approximate_probability * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
        for _ in range(tl.sum(exact.to(tl.int32))):
            offset = tl.min(exact_offsets)
            block = group_start + offset
            exact_offsets = tl.where(group_offsets == offset, GROUP_SIZE, exact_offsets)
            kv_start = block * BLOCK_SIZE
            k_rows = kv_start + token_offsets
            k_valid = k_rows < T
            approx_col = tl.sum(tl.where(group_offsets[None, :] == offset, scores, 0.0), axis=1)
            # K INT8 and its scales head-major: [B, H, T, D] / [B, H, T]
            ki = tl.load(ki_ptr + ((batch * H + head) * T + k_rows[:, None]).to(tl.int64) * D + d_offsets[None, :], mask=k_valid[:, None], other=0)
            ks = tl.load(k_scale + (batch * H + head) * T + k_rows, mask=k_valid, other=1.0)
            s32 = tl.dot(qi, ki.T, out_dtype=tl.int32)
            exact_scores = s32.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2 + approx_col[:, None]
            exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))
            block_max = tl.max(exact_scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.math.exp2(row_max - new_max)
            exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
            # V through its strides (the caller passes the head-major copy)
            vb = tl.load(v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + bv_offsets[None, :], mask=k_valid[:, None], other=0.0)
            output = output * alpha[:, None] + tl.dot(exact_probability.to(vb.dtype), vb)
            row_max = new_max

    tl.store(o_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + bv_offsets[None, :], (output / row_sum[:, None]).to(tl.bfloat16), mask=q_valid[:, None])


def attention(q, v, prepared, *, scale=None, tau=1.0, sink_blocks=(0, 0), out=None):
    """saganaki.fwd.sol_attn_prepared with head-major K/V: q, v [B, T, H, 128] BF16, prepared = prepare_kv(k, v); out may be q."""
    kc, vc, stat_mean, stat_var, k8, k_scale = prepared
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype) if out is None else out
    v = v.permute(0, 2, 1, 3).contiguous()
    k8, k_scale = k8.permute(0, 2, 1, 3).contiguous(), k_scale.permute(0, 2, 1).contiguous()
    _forward_hm[(1, blocks, batch * heads)](q, v, kc, vc, k8, k_scale, stat_mean, stat_var, output, scale, float(tau),
                                            int(sink_blocks[0]), int(sink_blocks[1]), 0, 0, tokens,
                                            q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(2), v.stride(1),
                                            H=heads, D=head_dim, NT=blocks, BV=head_dim, BLOCK_SIZE=BLOCK, GROUP_SIZE=GROUP, CENTROID_ROUTE=CENTROID_ROUTE,
                                            num_warps=4, num_stages=1)
    return output


_upstream_prepared, _upstream_sol_attn = _sol.sol_attn_prepared, _sol.sol_attn
_said = set()


def _usable(q, v):
    return q.is_cuda and q.dtype == torch.bfloat16 and v.dtype == torch.bfloat16 and q.shape[-1] == 128 and q.stride(-1) == 1 and v.stride(-1) == 1


def sol_attn_prepared(q, v, prepared, *, scale=None, tau=1.0, sink_tokens=0, sink_start=None, out=None):
    """shared.sol_attn.sol_attn_prepared (the staged path: GGUF, and the INT8 layers whose projections can be staged)."""
    if not _usable(q, v):
        return _upstream_prepared(q, v, prepared, scale=scale, tau=tau, sink_tokens=sink_tokens, sink_start=sink_start, out=out)
    if "on" not in _said:
        _said.add("on"); print("[Morpheus] Sol attention: head-major K/V kernel.", flush=True)
    return attention(q, v, prepared, scale=scale, tau=tau, sink_blocks=_itf._sink_block_range(q.shape[1], sink_start, sink_tokens), out=out)


def sol_attn(q, k, v, *, scale=None, tau=1.0, thresh_type="diag", sink_tokens=0, sink_start=None, query_start=0,
             recycle_q=False, int8_qk=False, int8_pv=False):
    """shared.sol_attn.sol_attn (the unstaged path); only upstream's inline-Q INT8 case is taken over, the rest goes upstream."""
    if not (int8_qk and not int8_pv and not query_start and thresh_type == "diag" and _usable(q, v) and k.dtype == torch.bfloat16
            and _fwd.inline_q_path(torch.cuda.get_device_capability(q.device), q.shape[1], thresh_type)):
        return _upstream_sol_attn(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type, sink_tokens=sink_tokens, sink_start=sink_start,
                                  query_start=query_start, recycle_q=recycle_q, int8_qk=int8_qk, int8_pv=int8_pv)
    _itf._validate_inputs(q, k, v, thresh_type, sink_tokens, sink_start, query_start)
    return sol_attn_prepared(q, v, _sol.prepare_kv(k, v), scale=scale, tau=tau, sink_tokens=sink_tokens, sink_start=sink_start)


if os.environ.get("MORPHEUS_SOL_KERNEL", "1") != "0":
    _sol.sol_attn_prepared, _sol.sol_attn = sol_attn_prepared, sol_attn

print("## sol_kernel loaded", flush=True)
