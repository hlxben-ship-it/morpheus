"""From Windows: an H3-Image edit through Morpheus' own path (engine/wsl_runner -> warm worker), seed h3_image_edit_speed.
Job 1 = reference size "full" (cold), job 2 = "1024" (warm). Same settings as bench variant img_pre1, so job 1 must match its picture.
    .venv/Scripts/python.exe bench/h3tune/e2e_image.py"""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from engine import wsl_runner
HERE = Path(__file__).parent
OUT = ROOT / "output" / "bench" / "h3tune" / "e2e_image"; OUT.mkdir(parents=True, exist_ok=True)
pic = str(HERE / "in" / "yukomo_p.png").replace("\\", "/")
base = {"model_type": "minimax_h3_image_pruned", "prompt": "Keep everything in <Picture 1> as it is, but she now wears a red knitted scarf and holds a steaming cup of tea.",
        "resolution": "1664x2496", "num_inference_steps": 6, "guidance_scale": 1.0, "guidance_phases": 1, "sample_solver": "euler",
        "batch_size": 1, "seed": 20261117, "image_mode": 1, "config": "distilled_4b", "video_prompt_type": "I", "image_refs": [pic]}
for name, extra in (("full", {}), ("1024", {"_h3_ref_size": "1024"})):
    t = time.time()
    ev = wsl_runner.run({"kind": "image", "family": "h3_image", "settings": {**base, **extra}}, lambda e: None, lambda: False, keep_warm=30)
    print(name, "wall", round(time.time() - t, 1), {k: ev.get(k) for k in ("event", "message", "seconds", "vram_peak_gb", "ram_peak_gb")}, flush=True)
    if ev.get("file"):
        shutil.copy(ev["file"], OUT / f"{name}{Path(ev['file']).suffix}")
wsl_runner.release()
