"""Image jobs in the warm worker: pick the pipeline, keep it warm, run a batch, report progress + step previews.

Residency (seed G2): one image model is active on the GPU. The previous one stays parked in RAM only when the parked RAM
fits PARK_BUDGET_GB; otherwise it is freed. A video model never sits beside an image model (the worker releases WanGP's
model before an own pipeline runs, and these before a WanGP job).
"""
import time
from pathlib import Path

import torch

from . import common

PARK_BUDGET_GB = 12   # RAM an inactive image model may keep inside the 24 GB VM
PARK_GB = {"sdxl": 7.0, "qwen_edit": 25.0, "krea2": 20.0}  # what a family holds in RAM when parked (Qwen, Krea 2 never fit: freed)
SDXL_ROOT = Path("/root/models/image/sdxl")
KREA2_ROOT = Path("/root/models/image/krea2")

_loaded = {}   # family -> (key, pipeline)
_active = None


def _build(family, key):
    if family == "sdxl":
        from .sdxl import SDXL
        return SDXL(key)
    if family == "qwen_edit":
        from .qwen_edit import QwenEdit
        return QwenEdit()
    if family == "krea2":
        from .krea2 import Krea2
        return Krea2(key)
    raise ValueError(f"Unknown image family {family!r}.")


def _park(family):
    key, pipe = _loaded[family]
    if PARK_GB.get(family, 99) <= PARK_BUDGET_GB:
        for m in ("unet", "vae", "te1", "te2", "dit"):
            if hasattr(pipe, m):
                getattr(pipe, m).to("cpu")
        pipe._parked = True
    else:
        del _loaded[family]
    common.free_vram()


def _wake(pipe):
    if getattr(pipe, "_parked", False):
        for m in ("unet", "te1", "te2", "dit"):
            if hasattr(pipe, m):
                getattr(pipe, m).to(common.DEVICE)
        if hasattr(pipe, "vae"):
            pipe.vae.to(common.DEVICE)
        pipe._parked = False


def pipeline(family, key):
    """The pipeline for (family, checkpoint key), loading or waking it and parking/freeing the others."""
    global _active
    for fam in list(_loaded):
        if fam != family and not getattr(_loaded[fam][1], "_parked", False):
            _park(fam)
    if family in _loaded and _loaded[family][0] != key:
        del _loaded[family]; common.free_vram()
    if family not in _loaded:
        _loaded[family] = (key, _build(family, key))
    _active = family
    pipe = _loaded[family][1]
    _wake(pipe)
    return pipe


def release_all():
    """Before a WanGP job: no image model stays (RAM goes to the video/H3 model)."""
    global _active
    _loaded.clear(); _active = None
    common.free_vram()


HEAVY = {"qwen_edit", "krea2"}  # own pipelines that hold >10 GB of RAM: freed after HEAVY_IDLE (warm_worker.py)


def release_heavy():
    """Free the heavy image models; the light ones (SDXL ~7 GB) stay. Returns whether anything is left."""
    global _active
    for fam in [f for f in _loaded if f in HEAVY]:
        del _loaded[fam]
        if _active == fam:
            _active = None
    common.free_vram()
    return bool(_loaded)


def loaded():
    return {fam: ("parked" if getattr(p, "_parked", False) else "active") for fam, (_, p) in _loaded.items()}


def run(job, say, stopped=lambda: False):
    """job: {id, family, params, count, out_dir}. Writes img_<n>_<seed>.png into out_dir; previews to out_dir/preview.jpg."""
    jid, fam, p = job.get("id"), job["family"], dict(job["params"])
    out = Path(job["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    key = str(SDXL_ROOT / p["checkpoint"]) if fam == "sdxl" else str(KREA2_ROOT / p["checkpoint"]) if fam == "krea2" else "qwen-image-edit-2511"
    t0 = time.time()
    say({"event": "progress", "id": jid, "phase": "loading_model", "step": None, "total": None})
    pipe = pipeline(fam, key)
    load_s = round(time.time() - t0, 1)
    factors, bias = pipe.preview_factors()
    count = max(1, int(job.get("count") or 1))
    files, seeds = [], []
    prev = out / "preview.jpg"
    last = [0.0]
    for n in range(count):
        seed = int(p["seed"]) + n
        q = {**p, "seed": seed}

        def emit(step, total, x0, n=n):
            g_step, g_total = n * total + step, count * total
            msg = {"event": "progress", "id": jid, "phase": "inference", "step": g_step, "total": g_total, "image": n + 1}
            now = time.time()
            if now - last[0] > 0.4 or step == total:  # previews at most ~2/s
                if common.preview(x0, factors, bias, prev):
                    msg["preview"] = str(prev)
                last[0] = now
            say(msg)

        img = pipe.generate(q, emit, stopped)
        say({"event": "progress", "id": jid, "phase": "decoding", "step": None, "total": None})
        f = common.save_png(img, out / f"img_{n + 1}_{seed}.png", {"family": fam, **q})
        files.append(str(f)); seeds.append(seed)
    try:
        prev.unlink()
    except OSError:
        pass
    return {"files": files, "seeds": seeds, "load_seconds": load_s, "resident": loaded()}
