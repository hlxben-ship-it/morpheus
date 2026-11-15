# Sol attention with the P.V product in int8 too (upstream opt-in "int8_pv": faster, costs accuracy). WanGP's staged H3 path
# hard-codes INT8_PV=False; this routes prepare_kv / sol_attn_prepared through the int8 V path.
import triton, shared.sol_attn as _S
from shared.sol_attn import interface as _I
from shared.sol_attn.saganaki import fwd as _F
from shared.sol_attn.saganaki.preprocess import prepare_int8_kv as _prep
from shared.sol_attn.saganaki.quant import quantize_v_per_channel as _qv


def _prepare_kv(k, v):
    kc, vc, sm, sv, k8, ks, vabs = _prep(k, v, int8_pv=True)
    vi, vscale = _qv(v, absmax=vabs)
    return kc, vc, sm, sv, k8, ks, vi, vscale


def _sol_attn_prepared(q, v, prepared, *, scale=None, tau=1.0, sink_tokens=0, sink_start=None, out=None):
    kc, vc, sm, sv, k8, ks, vi, vscale = prepared
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, _F.BLOCK)
    output = q.new_empty(q.shape) if out is None else out
    sb = _I._sink_block_range(tokens, sink_start, sink_tokens)
    _F._forward_int8_ptr[(1, blocks, batch * heads)](
        q, v, kc, vc, q, k8, q, ks, vi, vscale, q, sm, sv, output, scale, float(tau), int(sb[0]), int(sb[1]), 0, 0, tokens,
        q.stride(0), q.stride(1), q.stride(2), v.stride(0), v.stride(1), v.stride(2),
        H=heads, D=head_dim, NT=blocks, BV=head_dim, BLOCK_SIZE=_F.BLOCK, GROUP_SIZE=_F.GROUP, INT8_PV=True, INLINE_Q=True)
    return output


_S.prepare_kv = _I.prepare_kv = _prepare_kv
_S.sol_attn_prepared = _I.sol_attn_prepared = _sol_attn_prepared
print("## hook sol_int8pv installed", flush=True)
