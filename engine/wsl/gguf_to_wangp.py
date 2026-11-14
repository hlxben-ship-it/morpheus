"""Make Video Rebirth's HyperFlow H3 GGUF (VideoRebirthAI/hyperflow-h3-flash, *_gguf/) loadable by WanGP's MiniMax H3 code.

    /root/venv/bin/python gguf_to_wangp.py <in.gguf> <out.gguf>          (inside WSL)

Checked tensor by tensor against the bf16 source (2026-11-14): the GGUF already has WanGP's fc1 order (gate | value)
and WanGP's names, but (1) a fused attn.qkv_proj in the dense blocks + token refiner (rows q | k | v), which is split
into q/k/v_proj like WanGP's int8, and (2) its blocks numbered 0..48. GGUF
quantizes along a row, so whole rows move as raw bytes: no dequantize, no quality change. Its blocks are renumbered past
the ones HyperFlow removed (H3_DROPPED_BLOCK, default 9 = ref2va; fl2va: 6,21,22), and its hyperflow_gate / hyperflow_sigmas tensors are left out. Low-rank ranks stay padded
to multiples of 256 (K-quants need it); WanGP reads each rank from the file.
Everything else is copied as is. The HyperFlow gate (0.25, absent from the GGUF) is a default in minimax_h3_main.py.
Glia seed: seed_2026_11_14_hyperflow_gguf_fl2va.
"""
import os
import re
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter


HEAD_DIM = 128
INTERLEAVED = os.environ.get("QKV_LAYOUT") == "heads"  # rows (heads, q/k/v, head_dim) instead of (q/k/v, heads, head_dim)


def split_qkv(data: np.ndarray) -> list:
    """A fused qkv_proj (dense blocks + token refiner in the GGUF; WanGP's int8 has q/k/v_proj) -> q, k, v row blocks."""
    rows = data.shape[0]
    assert rows % 3 == 0, data.shape
    if not INTERLEAVED:
        return [data[i * rows // 3:(i + 1) * rows // 3] for i in range(3)]
    per = data.reshape(rows // (3 * HEAD_DIM), 3, HEAD_DIM, *data.shape[1:])
    return [per[:, i].reshape(rows // 3, *data.shape[1:]) for i in range(3)]


# H3 has 50 blocks; HyperFlow removed some (ref2va: 9; fl2va: 6, 21, 22 — see each partition's bf16 index). The GGUF
# numbers the rest 0..n straight through, WanGP (and its int8) keep H3's indices with the removed ones missing.
DROPPED = {int(x) for x in os.environ.get("H3_DROPPED_BLOCK", "9").split(",") if x.strip()}
KEPT = [i for i in range(50) if i not in DROPPED]


def wangp_name(name: str) -> str:
    m = re.match(r"blocks\.(\d+)\.(.*)", name)
    if not m:
        return name
    return f"blocks.{KEPT[int(m.group(1))]}.{m.group(2)}"


def main(src: str, dst: str) -> None:
    r = GGUFReader(src)
    w = GGUFWriter(dst, arch=None)
    split = 0
    for t in r.tensors:
        if t.name in ("hyperflow_gate", "hyperflow_sigmas"):  # 0.25 is the default in minimax_h3_main; WanGP has the sigmas
            continue
        name = wangp_name(t.name)
        data = np.asarray(t.data)
        quantized = t.tensor_type.name not in ("F32", "F16", "BF16")
        raw = t.tensor_type if quantized or t.tensor_type.name == "BF16" else None
        # quantized: the byte rows as read, the writer works the logical shape out from them
        if name.endswith(".attn.qkv_proj.weight"):
            for part, d in zip(("q_proj", "k_proj", "v_proj"), split_qkv(data)):
                w.add_tensor(name.replace("qkv_proj", part), np.ascontiguousarray(d), raw_dtype=raw)
            split += 1
            continue
        w.add_tensor(name, data, raw_dtype=raw)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{len(r.tensors)} tensors, qkv split in {split} ({'heads' if INTERLEAVED else 'blocks'}) -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
