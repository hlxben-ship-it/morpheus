"""ComfyUI vs the own WSL pipelines at the same seed and settings (seed image_playground G4: "match ComfyUI, then tune").

    python bench/image_parity/parity.py sdxl [sampler] [scheduler]          JANKU text-to-image
    python bench/image_parity/parity.py qwen <picture> [prompt]              Qwen-Image-Edit-2511 + Lightning

Renders in ComfyUI first (started if needed), stops it, renders the same job in WSL, and prints the mean / p99 absolute
pixel difference (0-255) and both times. Pictures + a side-by-side land in bench/image_parity/out/. Run only when nothing renders.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "out"

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import morpheus  # noqa: E402
import server  # noqa: E402
from engine import wsl_runner  # noqa: E402

PROMPT = "masterpiece, best quality, absurdres, 1girl, silver hair, rain, neon city, night"
NEG = "nsfw, worst quality, low quality, lowres, blurry"
LIGHTNING = str(ROOT / "ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")


def comfy(graph):
    server.ensure_comfy()
    t = time.time()
    img = morpheus.generate(graph, ghost=True)
    return img, time.time() - t


def wsl(job):
    morpheus.stop_comfy()
    wsl_runner.wait_for_ram()
    t = time.time()
    try:
        out = wsl_runner.run(job, lambda ev: None, lambda: False, keep_warm=60)
    finally:
        wsl_runner.release()
    return Image.open(out["file"]).convert("RGB"), time.time() - t, out


def compare(name, a, b, ta, tb, extra=None):
    OUT.mkdir(exist_ok=True)
    a.save(OUT / f"{name}_comfy.png"); b.save(OUT / f"{name}_wsl.png")
    x, y = np.asarray(a, dtype=np.int16), np.asarray(b.resize(a.size), dtype=np.int16)
    d = np.abs(x - y)
    side = Image.new("RGB", (a.width * 2, a.height)); side.paste(a, (0, 0)); side.paste(b.resize(a.size), (a.width, 0))
    side.thumbnail((1600, 1600)); side.save(OUT / f"{name}_side.jpg", quality=85)
    res = {"case": name, "mean_abs_diff": round(float(d.mean()), 2), "p99_abs_diff": int(np.percentile(d, 99)),
           "comfy_s": round(ta, 1), "wsl_s": round(tb, 1), **(extra or {})}
    print(json.dumps(res))
    with open(OUT / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**res, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def sdxl(sampler="euler_ancestral", scheduler="normal", steps=30, seed=42, w=832, h=1216, cfg=5.0):
    build = morpheus.sdxl_workflow("JANKUTrainedChenkinNoobai_v50.safetensors", sampler, scheduler, cfg, NEG)
    a, ta = comfy(build(PROMPT, w, h, seed, steps, negative=NEG, prefix=""))
    job = {"kind": "image", "family": "sdxl", "count": 1, "params": {
        "checkpoint": "JANKUTrainedChenkinNoobai_v50.safetensors", "prompt": PROMPT, "negative": NEG, "width": w, "height": h,
        "seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler, "scheduler": scheduler, "denoise": 1.0, "loras": []}}
    b, tb, out = wsl(job)
    compare(f"sdxl_{sampler}_{scheduler}", a, b, ta, tb, {"wsl_load_s": out.get("load_seconds"), "wsl_render_s": out.get("seconds")})


def qwen(picture, prompt="make it night, with warm window lights", seed=42, w=1216, h=832):
    name = f"parity_in_{int(time.time())}.png"
    Image.open(picture).convert("RGB").save(ROOT / "ComfyUI" / "input" / name)
    a, ta = comfy(morpheus.qwen_edit_workflow(prompt, name, w, h, seed, 4, loras=["qwen-lightning"]))
    job = {"kind": "image", "family": "qwen_edit", "count": 1, "params": {
        "prompt": prompt, "negative": "", "width": w, "height": h, "seed": seed, "steps": 4, "cfg": 1.0, "sampler": "euler",
        "scheduler": "simple", "denoise": 1.0, "images": [str(Path(picture).resolve())], "loras": [{"path": LIGHTNING, "strength": 1.0}]}}
    b, tb, out = wsl(job)
    compare("qwen_edit", a, b, ta, tb, {"wsl_load_s": out.get("load_seconds"), "wsl_render_s": out.get("seconds")})


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "sdxl":
        sdxl(*sys.argv[2:4])
    elif kind == "qwen":
        qwen(*sys.argv[2:4])
    else:
        raise SystemExit(__doc__)
