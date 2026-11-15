"""From Windows: stack switching through Morpheus' own path: LTX (cold, warm) on /root/venv, then H3 Q4 on /root/venv10, then LTX again.
    .venv/Scripts/python.exe bench/h3tune/e2e_stack.py"""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from engine import wsl_runner
HERE = Path(__file__).parent
clips = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
pic = str(HERE / "in" / "yukomo.jpg").replace("\\", "/")
ltx = {"model_type": "ltx2_22B_dev_q6", "prompt": "The camera slowly pushes in, soft wind moves the hair.", "image_prompt_type": "S", "image_start": pic,
       "seed": 20261112, "num_inference_steps": 8, "sample_solver": "distilled_8_steps", "guidance_scale": 1.0, "guidance_phases": 2, "force_fps": 24,
       "multi_prompts_gen_type": "FG", **clips["ltx2s"]}
h3 = {"model_type": "minimax_h3_ref2va_hyperflow_gguf_q4_k_m", "prompt": clips["_prompt"], "image_prompt_type": "S", "image_start": pic, "image_refs": [pic],
      "video_prompt_type": "I", "seed": 20261112, "num_inference_steps": 4, "sample_solver": "euler", "guidance_scale": 1.0, "guidance_phases": 1,
      "force_fps": 24, "multi_prompts_gen_type": "FG", "config": "distilled_4b,lightvae_int8", "_frames_minimum": 73, **clips["480s"]}
for tag, s in (("ltx cold", ltx), ("ltx warm", ltx), ("h3 switch", h3), ("ltx switch", ltx)):
    t = time.time()
    ev = wsl_runner.run({"kind": "render", "settings": s}, lambda e: None, lambda: False, keep_warm=30)
    print(tag, "stack", wsl_runner.WARM.stack, "wall", round(time.time() - t, 1), {k: ev.get(k) for k in ("seconds", "vram_peak_gb", "ram_peak_gb")}, flush=True)
    d = Path(ev["file"]).parent
    if d.parent.name == "jobs": shutil.rmtree(d, ignore_errors=True); Path(str(d) + ".result.json").unlink(missing_ok=True)
wsl_runner.release()
