# GGUF Q4_K / Q6_K -> dense weights in ONE Triton kernel launch per linear.
# WanGP's fast GGUF path is the llamacpp_gguf_cuda wheel (NVIDIA, or HIP for PyTorch 2.13 + ROCm 10 only). Without it every
# linear dequantises with ~10 full-size PyTorch elementwise passes: 37 % of the GPU time of a HyperFlow Q4 render on the 7900 XT
# (profile 2026-11-15). Same arithmetic as shared/qtypes/gguf.py (_dequantize_blocks_Q4_K / _Q6_K): float32, then the target dtype.
import torch, triton, triton.language as tl
import shared.qtypes.gguf as _G  # needs the WanGP root on sys.path (warm_worker.py imports this after init)


@triton.jit
def _deq_q4_k(RAW, OUT, n_blocks, BLK: tl.constexpr):
    b = tl.program_id(0) * BLK + tl.arange(0, BLK)
    ok = b < n_blocks
    base = RAW + b.to(tl.int64) * 144                                   # d f16 | dmin f16 | 12 scale bytes | 128 nibble bytes
    d = tl.load(base.to(tl.pointer_type(tl.float16)), mask=ok, other=0.0).to(tl.float32)
    dmin = tl.load((base + 2).to(tl.pointer_type(tl.float16)), mask=ok, other=0.0).to(tl.float32)
    j = tl.arange(0, 8)                                                 # 8 sub-blocks of 32 values
    lo = (j % 4)[None, :]
    ok2 = ok[:, None]
    s_d = tl.load(base[:, None] + 4 + lo, mask=ok2, other=0).to(tl.int32)
    s_m = tl.load(base[:, None] + 8 + lo, mask=ok2, other=0).to(tl.int32)
    s_md = tl.load(base[:, None] + 12 + lo, mask=ok2, other=0).to(tl.int32)
    hi = (j >= 4)[None, :]
    sc = tl.where(hi, (s_md & 0x0F) | ((s_d >> 2) & 0x30), s_d & 0x3F)
    mn = tl.where(hi, (s_md >> 4) | ((s_m >> 2) & 0x30), s_m & 0x3F)
    scale = d[:, None] * sc.to(tl.float32)
    off = dmin[:, None] * mn.to(tl.float32)
    i = tl.arange(0, 32)
    ok3 = ok[:, None, None]
    q = tl.load(base[:, None, None] + 16 + (j // 2)[None, :, None] * 32 + i[None, None, :], mask=ok3, other=0).to(tl.int32)
    q = (q >> (4 * (j % 2))[None, :, None]) & 0x0F
    val = scale[:, :, None] * q.to(tl.float32) - off[:, :, None]
    tl.store(OUT + b.to(tl.int64)[:, None, None] * 256 + j[None, :, None] * 32 + i[None, None, :], val.to(OUT.dtype.element_ty), mask=ok3)


@triton.jit
def _deq_q6_k(RAW, OUT, n_blocks, BLK: tl.constexpr):
    b = tl.program_id(0) * BLK + tl.arange(0, BLK)
    ok = b < n_blocks
    base = RAW + b.to(tl.int64) * 210                                   # 128 ql | 64 qh | 16 int8 scales | d f16
    d = tl.load((base + 208).to(tl.pointer_type(tl.float16)), mask=ok, other=0.0).to(tl.float32)
    r = tl.arange(0, 8)                                                 # 8 rows of 32 values (2 sub-blocks of 16 each)
    i = tl.arange(0, 32)
    c, s = r // 4, r % 4
    ok3 = ok[:, None, None]
    ql = tl.load(base[:, None, None] + (c * 64 + (s % 2) * 32)[None, :, None] + i[None, None, :], mask=ok3, other=0).to(tl.int32)
    ql = (ql >> (4 * (s // 2))[None, :, None]) & 0x0F
    qh = tl.load(base[:, None, None] + 128 + (c * 32)[None, :, None] + i[None, None, :], mask=ok3, other=0).to(tl.int32)
    qh = (qh >> (2 * s)[None, :, None]) & 0x03
    q = (ql | (qh << 4)) - 32
    sc = tl.load((base[:, None, None] + 192 + (r * 2)[None, :, None] + (i // 16)[None, None, :]).to(tl.pointer_type(tl.int8)), mask=ok3, other=0).to(tl.float32)
    val = (d[:, None, None] * sc) * q.to(tl.float32)
    tl.store(OUT + b.to(tl.int64)[:, None, None] * 256 + r[None, :, None] * 32 + i[None, None, :], val.to(OUT.dtype.element_ty), mask=ok3)


_KERNELS = {"Q4_K": (_deq_q4_k, 144), "Q6_K": (_deq_q6_k, 210)}
_BLK = int(__import__("os").environ.get("GGUF_TRITON_BLK", "8"))
_WARPS = int(__import__("os").environ.get("GGUF_TRITON_WARPS", "4"))
_orig_dequantize = _G._gguf_dequantize_tensor
_said = set()


def _gguf_dequantize_tensor(raw, qtype_obj, oshape, dtype=None):
    entry = _KERNELS.get(getattr(qtype_obj, "name", None))
    if entry is None or not raw.is_cuda or dtype not in (torch.bfloat16, torch.float16, torch.float32) or torch.compiler.is_compiling():
        return _orig_dequantize(raw, qtype_obj, oshape, dtype=dtype)
    kernel, type_size = entry
    data = raw.reshape(-1).view(torch.uint8)
    if not data.is_contiguous() or data.data_ptr() % 2:
        data = data.contiguous()
    n_blocks = data.numel() // type_size
    out = torch.empty(n_blocks * 256, dtype=dtype, device=raw.device)
    kernel[(triton.cdiv(n_blocks, _BLK),)](data, out, n_blocks, BLK=_BLK, num_warps=_WARPS)
    if qtype_obj.name not in _said:
        _said.add(qtype_obj.name); print(f"[GGUF] Triton dequant kernel active for {qtype_obj.name} (blocks/program {_BLK}, warps {_WARPS}).", flush=True)
    return out.reshape(oshape)


_G._gguf_dequantize_tensor = _gguf_dequantize_tensor
print("## hook gguf_triton installed", flush=True)
