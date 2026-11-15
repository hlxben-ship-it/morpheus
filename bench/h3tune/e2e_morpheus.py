"""From Windows: the bench clip through Morpheus' own path (engine/wsl_runner -> warm worker), one cold + one warm job per model.
    .venv/Scripts/python.exe bench/h3tune/e2e_morpheus.py [int8 q4] [480s]"""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from engine import wsl_runner
HERE = Path(__file__).parent
clips = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
MODELS = {"int8": "minimax_h3_ref2va_hyperflow", "q4": "minimax_h3_ref2va_hyperflow_gguf_q4_k_m", "q6": "minimax_h3_ref2va_hyperflow_gguf_q6_k"}
args = sys.argv[1:] or ["int8", "q4"]
clip = next((a for a in args if a in clips), "480s"); models = [a for a in args if a in MODELS]
pic = str(HERE / "in" / "yukomo.jpg").replace("\\", "/")
for m in models:
    s = {"model_type": MODELS[m], "prompt": clips["_prompt"], "image_prompt_type": "S", "image_start": pic, "image_refs": [pic],
         "video_prompt_type": "I", "seed": 20261112, "num_inference_steps": 4, "sample_solver": "euler", "guidance_scale": 1.0,
         "guidance_phases": 1, "force_fps": 24, "multi_prompts_gen_type": "FG", "config": "distilled_4b,lightvae_int8", **clips[clip]}
    if s["video_length"] < 107: s["_frames_minimum"] = s["video_length"]
    for i in ("cold", "warm"):
        t = time.time()
        ev = wsl_runner.run({"kind": "render", "settings": s}, lambda e: None, lambda: False, keep_warm=30)
        print(m, clip, i, "wall", round(time.time() - t, 1), {k: ev.get(k) for k in ("seconds", "vram_peak_gb", "ram_peak_gb")}, flush=True)
        d = Path(ev["file"]).parent
        if d.parent.name == "jobs": shutil.rmtree(d, ignore_errors=True); Path(str(d) + ".result.json").unlink(missing_ok=True)
    wsl_runner.release()
