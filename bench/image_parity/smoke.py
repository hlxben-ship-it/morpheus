"""Smoke-run one image job through the real WSL path (engine.wsl_runner -> warm_worker), print every event.

    python bench/image_parity/smoke.py h3  [prompt]
    python bench/image_parity/smoke.py h3edit <picture> [prompt]
    python bench/image_parity/smoke.py sdxl [prompt]
    python bench/image_parity/smoke.py qwen <picture> [prompt]

Kills any warm worker another process holds (the server's too): run it only when nothing renders.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import wsl_runner  # noqa: E402

JANKU = "JANKUTrainedChenkinNoobai_v50.safetensors"
LIGHTNING = "C:/dev/morpheus/ComfyUI/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"


def job(kind, args):
    prompt = args[-1] if args and not Path(args[-1]).is_file() else None
    pic = next((a for a in args if Path(a).is_file()), None)
    if kind in ("h3", "h3edit"):
        s = {"model_type": "minimax_h3_image_pruned", "prompt": prompt or "A watercolour fox sleeping under a pear tree at dusk, soft light.",
             "resolution": "2048x2048", "num_inference_steps": 6, "guidance_scale": 1.0, "guidance_phases": 1, "sample_solver": "euler",
             "batch_size": 1, "seed": 42, "image_mode": 1, "config": "distilled_4b"}
        if kind == "h3edit":
            s.update(image_refs=[str(Path(pic).resolve())], video_prompt_type="KI",
                     prompt=prompt or "<Picture 1> with the person wearing a red knitted scarf, everything else unchanged.")
        return {"kind": "image", "family": "h3_image", "settings": s}
    if kind == "sdxl":
        return {"kind": "image", "family": "sdxl", "count": 1, "params": {
            "checkpoint": JANKU, "prompt": "masterpiece, best quality, absurdres, " + (prompt or "1girl, silver hair, rain, neon city, night"),
            "negative": "nsfw, worst quality, low quality, lowres, blurry", "width": 832, "height": 1216, "seed": 42, "steps": 30,
            "cfg": 5.0, "sampler": "euler_ancestral", "scheduler": "normal", "denoise": 1.0, "loras": []}}
    if kind == "qwen":
        return {"kind": "image", "family": "qwen_edit", "count": 1, "params": {
            "prompt": prompt or "make it night, with warm window lights", "negative": "", "width": 1216, "height": 832, "seed": 42, "steps": 4,
            "cfg": 1.0, "sampler": "euler", "scheduler": "simple", "denoise": 1.0, "images": [str(Path(pic).resolve())],
            "loras": [{"path": LIGHTNING, "strength": 1.0}]}}
    raise SystemExit(__doc__)


def main():
    kind, args = sys.argv[1], sys.argv[2:]
    j = job(kind, args)
    t0 = time.time()
    wsl_runner.wait_for_ram()
    try:
        out = wsl_runner.run(j, lambda ev: print("  ", json.dumps(ev)[:200], flush=True), lambda: False, keep_warm=60)
        print("DONE", json.dumps(out), f"{time.time() - t0:.0f}s total")
    finally:
        wsl_runner.release()


if __name__ == "__main__":
    main()
