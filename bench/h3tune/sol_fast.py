"""Experimental copy of WanGP's Sol pointer kernel (shared/sol_attn/saganaki/fwd.py::_forward_int8_ptr, inline-Q, bf16 P.V only)
for kernel experiments in solbench.py. ABL switches parts off to see what the exact loop is made of (results are then wrong):
  0 = the kernel as upstream     1 = no INT8 Q.K dot (centroid score only)     2 = no P.V dot
  3 = no exp2 in the exact loop  4 = no K / V loads in the exact loop (zeros)  5 = exact loop body empty (loop + bookkeeping only)
"""
import triton, triton.language as tl
from shared.sol_attn.saganaki.quant import _round_to_int8

NS = ['T', 'NT', 'sink_start', 'sink_end', 'sink_q_start', 'sink_q_end', 'sq_b', 'sq_t', 'sq_h', 'sv_b', 'sv_t', 'sv_h', 'sk_t', 'sk_h', 'ss_t', 'ss_h']


@triton.jit(do_not_specialize=NS, do_not_specialize_on_alignment=NS)
def forward(q_ptr, v_ptr, vsc_ptr, kc_ptr, vc_ptr, ki_ptr, k_scale, kc_mean_ptr, kc_var_ptr, o_ptr, scale, tau,
            sink_start, sink_end, sink_q_start, sink_q_end, T, sq_b, sq_t, sq_h, sv_b, sv_t, sv_h, sk_t, sk_h, ss_t, ss_h,
            H: tl.constexpr, D: tl.constexpr, NT, BV: tl.constexpr, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, ABL: tl.constexpr, KHM: tl.constexpr, CHM: tl.constexpr, VHM: tl.constexpr, KST: tl.constexpr, PV8: tl.constexpr, CROUTE: tl.constexpr):
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
        if CHM:   # centroids head-major too: [B, H, NT, D]
            kc = tl.load(kc_ptr + ((batch * H + head) * NT + block_indices[:, None]) * D + d_offsets[None, :], mask=valid[:, None], other=0.0)
            vc = tl.load(vc_ptr + ((batch * H + head) * NT + block_indices[:, None]) * D + bv_offsets[None, :], mask=valid[:, None], other=0.0)
        else:
            kc = tl.load(kc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + d_offsets[None, :], mask=valid[:, None], other=0.0)
            vc = tl.load(vc_ptr + ((batch * NT + block_indices[:, None]) * H + head) * D + bv_offsets[None, :], mask=valid[:, None], other=0.0)
        scores = tl.dot(q, kc.T).to(tl.float32) * scale_log2
        sink_kv = (block_indices >= sink_start) & (block_indices < sink_end)
        if CROUTE:   # sum over the rows of q.kc = q_len * centroid.kc: the route score from the centroid, in FP32
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
            if ABL != 5:
                kv_start = block * BLOCK_SIZE
                k_rows = kv_start + token_offsets
                k_valid = k_rows < T
                approx_col = tl.sum(tl.where(group_offsets[None, :] == offset, scores, 0.0), axis=1)
                if ABL == 1:
                    exact_scores = approx_col[:, None] + tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
                else:
                    if ABL == 4:
                        ki = tl.zeros([BLOCK_SIZE, D], dtype=tl.int8) + (block % 3).to(tl.int8)
                        ks = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + 1.0
                    else:
                        if KST:   # K int8 and its scales through runtime strides (any layout)
                            ki = tl.load(ki_ptr + k_rows[:, None].to(tl.int64) * sk_t + head * sk_h + d_offsets[None, :], mask=k_valid[:, None], other=0)
                            ks = tl.load(k_scale + k_rows.to(tl.int64) * ss_t + head * ss_h, mask=k_valid, other=1.0)
                        elif KHM:   # K int8 and its scales head-major: [B, H, T, D] / [B, H, T]
                            ki = tl.load(ki_ptr + ((batch * H + head) * T + k_rows[:, None]).to(tl.int64) * D + d_offsets[None, :], mask=k_valid[:, None], other=0)
                            ks = tl.load(k_scale + (batch * H + head) * T + k_rows, mask=k_valid, other=1.0)
                        else:
                            ki = tl.load(ki_ptr + ((batch * T + k_rows[:, None]) * H + head) * D + d_offsets[None, :], mask=k_valid[:, None], other=0)
                            ks = tl.load(k_scale + (batch * T + k_rows) * H + head, mask=k_valid, other=1.0)
                    s32 = tl.dot(qi, ki.T, out_dtype=tl.int32)
                    exact_scores = s32.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2 + approx_col[:, None]
                exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))
                block_max = tl.max(exact_scores, axis=1)
                new_max = tl.maximum(row_max, block_max)
                if PV8:   # INT8 P.V on head-major INT8 V (upstream's INT8_PV arithmetic): P against the block max lands in [0, 1]
                    alpha = tl.math.exp2(row_max - new_max)
                    vi = tl.load(v_ptr + ((batch * H + head) * T + k_rows[:, None]).to(tl.int64) * D + bv_offsets[None, :], mask=k_valid[:, None], other=0)
                    pe = tl.math.exp2(exact_scores - block_max[:, None])
                    beta = tl.math.exp2(block_max - new_max)
                    row_sum = row_sum * alpha + beta * tl.sum(pe, axis=1)
                    pi = tl.minimum(pe * 127.0 + 0.5, 127.0).to(tl.int8)
                    v_scale = tl.load(vsc_ptr + batch_head * D + bv_offsets)
                    output = output * alpha[:, None] + tl.dot(pi, vi, out_dtype=tl.int32).to(tl.float32) * ((beta * (1.0 / 127.0))[:, None] * v_scale[None, :])
                elif ABL == 3:
                    alpha = tl.where(row_max == new_max, 1.0, 0.5)
                    exact_probability = tl.where(exact_scores == new_max[:, None], 1.0, 0.25)
                else:
                    alpha = tl.math.exp2(row_max - new_max)
                    exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
                if not PV8:
                    row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
                if PV8:
                    pass
                elif ABL == 2:
                    output = output * alpha[:, None] + tl.sum(exact_probability, axis=1)[:, None]
                else:
                    if ABL == 4:
                        vb = tl.zeros([BLOCK_SIZE, BV], dtype=tl.bfloat16) + 1.0
                    else:
                        if VHM:   # V head-major with constant strides
                            vb = tl.load(v_ptr + ((batch * H + head) * T + k_rows[:, None]).to(tl.int64) * D + bv_offsets[None, :], mask=k_valid[:, None], other=0.0)
                        else:
                            vb = tl.load(v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + bv_offsets[None, :], mask=k_valid[:, None], other=0.0)
                    output = output * alpha[:, None] + tl.dot(exact_probability.to(vb.dtype), vb)
                row_max = new_max

    tl.store(o_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + bv_offsets[None, :], (output / row_sum[:, None]).to(tl.bfloat16), mask=q_valid[:, None])


@triton.jit(do_not_specialize=NS, do_not_specialize_on_alignment=NS)
def forward_split(q_ptr, v_ptr, kc_ptr, vc_ptr, ki_ptr, k_scale, kc_mean_ptr, kc_var_ptr, o_ptr, scale, tau,
            sink_start, sink_end, sink_q_start, sink_q_end, T, sq_b, sq_t, sq_h, sv_b, sv_t, sv_h,
            H: tl.constexpr, D: tl.constexpr, NT, BV: tl.constexpr, BLOCK_SIZE: tl.constexpr, GROUP_SIZE: tl.constexpr, ABL: tl.constexpr, ROWS: tl.constexpr):
    v_tile, q_block, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = batch_head // H, batch_head % H
    group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    row_offsets = v_tile * ROWS + tl.arange(0, ROWS)
    d_offsets = tl.arange(0, D)
    bv_offsets = tl.arange(0, BV)
    q_start = q_block * BLOCK_SIZE
    full_rows = q_start + token_offsets
    full_valid = full_rows < T
    q_full = tl.load(q_ptr + batch * sq_b + full_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d_offsets[None, :], mask=full_valid[:, None], other=0.0)
    q_len = tl.minimum(BLOCK_SIZE, T - q_start).to(tl.float32)
    centroid = tl.sum(q_full.to(tl.float32), axis=0) / q_len
    q_rows = q_start + row_offsets
    q_valid = q_rows < T
    q = tl.load(q_ptr + batch * sq_b + q_rows[:, None].to(tl.int64) * sq_t + head * sq_h + d_offsets[None, :], mask=q_valid[:, None], other=0.0)
    q_fp32 = q.to(tl.float32)
    q_amax = tl.max(tl.abs(q_fp32), axis=1)
    qs = tl.maximum(q_amax / 127.0, 1e-8)
    qi = _round_to_int8(q_fp32 / qs[:, None])
    output = tl.zeros([ROWS, BV], dtype=tl.float32)
    row_sum = tl.zeros((ROWS,), dtype=tl.float32)
    row_max = tl.full((ROWS,), -float("inf"), tl.float32)
    scale_log2 = scale * 1.4426950408889634
    tail_length = T - (NT - 1) * BLOCK_SIZE
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
        route_score = tl.sum(centroid[None, :] * kc.to(tl.float32), axis=1) * scale_log2
        routed = ((route_score > route_threshold) | (tl.abs(q_block - block_indices) <= 1) | sink_kv) & valid
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
            if ABL != 5:
                kv_start = block * BLOCK_SIZE
                k_rows = kv_start + token_offsets
                k_valid = k_rows < T
                approx_col = tl.sum(tl.where(group_offsets[None, :] == offset, scores, 0.0), axis=1)
                if ABL == 1:
                    exact_scores = approx_col[:, None] + tl.zeros([ROWS, BLOCK_SIZE], dtype=tl.float32)
                else:
                    if ABL == 4:
                        ki = tl.zeros([BLOCK_SIZE, D], dtype=tl.int8) + (block % 3).to(tl.int8)
                        ks = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + 1.0
                    else:
                        ki = tl.load(ki_ptr + ((batch * T + k_rows[:, None]) * H + head) * D + d_offsets[None, :], mask=k_valid[:, None], other=0)
                        ks = tl.load(k_scale + (batch * T + k_rows) * H + head, mask=k_valid, other=1.0)
                    s32 = tl.dot(qi, ki.T, out_dtype=tl.int32)
                    exact_scores = s32.to(tl.float32) * (qs[:, None] * ks[None, :]) * scale_log2 + approx_col[:, None]
                exact_scores += tl.where(k_valid[None, :], 0.0, -float("inf"))
                block_max = tl.max(exact_scores, axis=1)
                new_max = tl.maximum(row_max, block_max)
                if ABL == 3:
                    alpha = tl.where(row_max == new_max, 1.0, 0.5)
                    exact_probability = tl.where(exact_scores == new_max[:, None], 1.0, 0.25)
                else:
                    alpha = tl.math.exp2(row_max - new_max)
                    exact_probability = tl.math.exp2(exact_scores - new_max[:, None])
                row_sum = row_sum * alpha + tl.sum(exact_probability, axis=1)
                if ABL == 2:
                    output = output * alpha[:, None] + tl.sum(exact_probability, axis=1)[:, None]
                else:
                    if ABL == 4:
                        vb = tl.zeros([BLOCK_SIZE, BV], dtype=tl.bfloat16) + 1.0
                    else:
                        vb = tl.load(v_ptr + batch * sv_b + k_rows[:, None].to(tl.int64) * sv_t + head * sv_h + bv_offsets[None, :], mask=k_valid[:, None], other=0.0)
                    output = output * alpha[:, None] + tl.dot(exact_probability.to(vb.dtype), vb)
                row_max = new_max

    tl.store(o_ptr + ((batch * T + q_rows[:, None]) * H + head) * D + bv_offsets[None, :], (output / row_sum[:, None]).to(tl.bfloat16), mask=q_valid[:, None])


def sol_attn_prepared(q, v, prepared, *, tau, sink_blocks=(0, 0), out=None, abl=0, num_warps=4, num_stages=1, kernel=forward, rows=64, head_major=0, pv8=None, croute=False):   # head_major: 1 = V, 2 = V and K, 3 = and centroids, 4 = V and K with constant V strides, 5 = V and K head-major both through strides, 6 = upstream layout, K through strides
    import torch
    kc, vc, stat_mean, stat_var, k8, k_scale = prepared
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, 64)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype) if out is None else out
    extra = {"KHM": head_major >= 2, "CHM": head_major == 3, "VHM": head_major == 4, "KST": head_major in (5, 6), "PV8": pv8 is not None, "CROUTE": croute} if kernel is forward else {"ROWS": rows}
    if head_major == 6: head_major_layout = False
    else: head_major_layout = bool(head_major)
    if head_major_layout:   # contiguous per head: a K/V block of one head is one run of memory instead of 64 rows 7 / 14 KB apart
        v = v.permute(0, 2, 1, 3).contiguous()
        sv = (v.stride(0), v.stride(2), v.stride(1))
        if head_major >= 2: k8, k_scale = k8.permute(0, 2, 1, 3).contiguous(), k_scale.permute(0, 2, 1).contiguous()
        if head_major == 3: kc, vc = kc.permute(0, 2, 1, 3).contiguous(), vc.permute(0, 2, 1, 3).contiguous()
    else: sv = (v.stride(0), v.stride(1), v.stride(2))
    if head_major_layout and head_major >= 2: sk = (k8.stride(2), k8.stride(1), k_scale.stride(2), k_scale.stride(1))
    else: sk = (k8.stride(1), k8.stride(2), k_scale.stride(1), k_scale.stride(2))
    vsc = q
    if pv8 is not None:   # pv8 = (vi [B, T, H, D] INT8, v_scale): V is replaced by its head-major INT8 form
        v, vsc = pv8[0].permute(0, 2, 1, 3).contiguous(), pv8[1]
    head = (q, v, vsc) if kernel is forward else (q, v)
    kernel[(1 if kernel is forward else 64 // rows, blocks, batch * heads)](*head, kc, vc, k8, k_scale, stat_mean, stat_var, output, head_dim ** -0.5, float(tau),
                                       int(sink_blocks[0]), int(sink_blocks[1]), 0, 0, tokens,
                                       q.stride(0), q.stride(1), q.stride(2), sv[0], sv[1], sv[2], sk[0], sk[1], sk[2], sk[3],
                                       H=heads, D=head_dim, NT=blocks, BV=head_dim, BLOCK_SIZE=64, GROUP_SIZE=32, ABL=abl, **extra,
                                       num_warps=num_warps, num_stages=num_stages)
    return output
