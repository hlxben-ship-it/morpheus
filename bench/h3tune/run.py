"""H3 tuning bench (Glia seed challenge_spike_h3_tuning). Runs inside WSL, one variant per process:

    python run.py variants/<name>.json

Variant keys (all optional but name): name, clip (clips.json), model (int8|q4|q6|fl2va_q4), settings (merged over the
clip), cli (WanGP flags: attention, profile, preload, ...), env, wangp (root of the WanGP copy), sol (false = no Sol),
pre_hooks / hooks (python files run before the WanGP import / after init, namespace: var, session, wgp),
repeat (jobs in the same session: warm numbers), ref (run to compare frames with), note.
Out: output/bench/h3tune/<name>/{result.json, <name>.mp4, strip.png} + a row in results.csv.
"""
import csv, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTROOT = Path("/mnt/c/dev/morpheus/output/bench/h3tune")
var = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
name = var["name"]; sys.argv = sys.argv[:1]  # WanGP parses sys.argv itself
for k, v in var.get("env", {}).items(): os.environ[k] = str(v)
sys.path.insert(0, "/mnt/c/dev/morpheus/engine/wsl")
if not var.get("no_spoof"): import amdspoof  # noqa: F401
ns = {"var": var}
for h in var.get("pre_hooks", []): exec(compile((HERE / h).read_text(), h, "exec"), ns)

MODELS = {"int8": "minimax_h3_ref2va_hyperflow", "q4": "minimax_h3_ref2va_hyperflow_gguf_q4_k_m",
          "q6": "minimax_h3_ref2va_hyperflow_gguf_q6_k", "fl2va_q4": "minimax_h3_fl2va_hyperflow_gguf_q4_k_m", "ltx": "ltx2_22B_dev_q6"}
clips = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
clip, model = var.get("clip", "480s"), var.get("model", "int8")
pic = str(HERE / "in" / "yukomo.jpg")
settings = {"model_type": MODELS[model], "prompt": clips["_prompt"], "image_prompt_type": "S", "image_start": pic,
            "image_refs": [pic], "video_prompt_type": "I", "seed": 20261112, "num_inference_steps": 4,
            "sample_solver": "euler", "guidance_scale": 1.0, "guidance_phases": 1, "force_fps": 24,
            "multi_prompts_gen_type": "FG", "config": "distilled_4b,lightvae_int8", **clips[clip]}
if model == "ltx":  # LTX 2.3 dev Q6 + distill LoRA, what morpheus.video_clip() and the Playground send (engine/jobs.py ltx_steps(8))
    settings = {"model_type": "ltx2_22B_dev_q6", "prompt": "The camera slowly pushes in, soft wind moves the hair.", "image_prompt_type": "S",
                "image_start": pic, "seed": 20261112, "num_inference_steps": 8, "sample_solver": "distilled_8_steps", "guidance_scale": 1.0,
                "guidance_phases": 2, "force_fps": 24, "multi_prompts_gen_type": "FG", **clips[clip]}
    if var.get("end"): settings.update(image_prompt_type="SE", image_end=pic)
    var["sol"] = False
if var.get("sol", True): settings.update({"overrides": "attention_mode='sol'", "attention_sparsity": 1.3})
settings.update(var.get("settings", {}))
settings = {k: v for k, v in settings.items() if v is not None}
cli = {"attention": "sdpa", "profile": 4, "preload": 14000, **var.get("cli", {})}
cli_args = [x for k, v in cli.items() if v is not None for x in ([f"--{k}"] if v is True else [f"--{k}", str(v)])]

import psutil
ROOT = Path(var.get("wangp", "/root/wangp")); os.chdir(ROOT); sys.path.insert(0, str(ROOT))
out_dir = OUTROOT / name; out_dir.mkdir(parents=True, exist_ok=True)
peak = {"avail_min": 1e9, "rss_max": 0.0, "swap_max": 0.0}; stop = threading.Event(); me = psutil.Process()
swap0 = psutil.swap_memory().used
def watch():
    while not stop.is_set():
        peak["avail_min"] = min(peak["avail_min"], psutil.virtual_memory().available / 2**30)
        peak["rss_max"] = max(peak["rss_max"], me.memory_info().rss / 2**30)
        peak["swap_max"] = max(peak["swap_max"], (psutil.swap_memory().used - swap0) / 2**30); time.sleep(0.25)
threading.Thread(target=watch, daemon=True).start()

T0 = time.time()
from shared.api import init
if var.get("early_wgp"): import wgp  # noqa: F401  what Morpheus' warm worker did until 2026-11-15: WanGP imported before init, so the flags never arrive
session = init(root=ROOT, console_output=True, cli_args=cli_args)
session._ensure_runtime()  # imports wgp with the cli flags as its argv
import torch, wgp
t_init = time.time() - T0
ns.update(session=session, wgp=wgp, torch=torch)
for h in var.get("hooks", []): exec(compile((HERE / h).read_text(), h, "exec"), ns)
if model != "ltx" and settings["video_length"] < 107: wgp.get_model_def(settings["model_type"])["frames_minimum"] = settings["video_length"]
for k in ("save_path", "image_save_path", "audio_save_path"): wgp.server_config[k] = str(ROOT / "outputs")


def one_job(i):
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); marks = []; last = None
    job = session.submit_task(dict(settings))
    for ev in job.events.iter(timeout=0.25):
        if ev.kind != "progress": continue
        p = ev.data; now = (p.phase, p.current_step)
        if now != last:
            last = now; marks.append((p.phase, p.current_step, round(time.time() - t0, 2)))
    res = job.result(); total = time.time() - t0
    def first(ph): return next((t for p, s, t in marks if p == ph), None)
    inf = [(s, t) for p, s, t in marks if p == "inference" and s is not None]
    steps = [round(b[1] - a[1], 1) for a, b in zip(inf, inf[1:]) if b[0] == a[0] + 1]
    dec = first("decoding"); t_inf = first("inference")
    if dec and inf: steps.append(round(dec - inf[-1][1], 1))  # the last step ends where decoding starts
    rest = sorted(steps[1:] or steps)
    r = {"job": i, "success": bool(res.success and res.generated_files), "errors": [e.message for e in res.errors],
         "total": round(total, 1), "load": round(first("encoding_text") or 0, 1),
         "text": round((t_inf or 0) - (first("encoding_text") or 0), 1), "denoise": round((dec or total) - (t_inf or 0), 1),
         "decode": round(total - (dec or total), 1), "steps": steps, "marks": marks,
         "s_step": rest[len(rest) // 2] if rest else None,
         "vram_alloc": round(torch.cuda.max_memory_allocated() / 2**30, 1), "vram_resv": round(torch.cuda.max_memory_reserved() / 2**30, 1),
         "file": str(res.generated_files[-1]) if res.generated_files else None}
    print("@@", json.dumps({k: v for k, v in r.items() if k != "marks"}), flush=True); return r


jobs = []; err = None
try:
    for i in range(int(var.get("repeat", 1))): jobs.append(one_job(i))
except Exception as e:  # noqa: BLE001
    import traceback; traceback.print_exc(); err = f"{type(e).__name__}: {e}"
for fn in ns.get("after", []): fn()
stop.set()
out = {"name": name, "model": model, "clip": clip, "note": var.get("note", ""), "cli": cli, "settings": settings, "error": err,
       "init": round(t_init, 1), "jobs": jobs, "rss_max": round(peak["rss_max"], 1), "avail_min": round(peak["avail_min"], 1),
       "swap_max": round(peak["swap_max"], 1), "wall": round(time.time() - T0, 1)}
ok = bool(jobs) and all(j["success"] for j in jobs) and not err
q = {}
if ok:
    import shutil
    mp4 = out_dir / f"{name}.mp4"; shutil.copy(jobs[-1]["file"], mp4)
    try:
        sys.path.insert(0, str(HERE)); import quality
        ref = OUTROOT / var["ref"] / f"{var['ref']}.mp4" if var.get("ref") else None
        q = quality.check(mp4, out_dir / "strip.png", ref if ref and ref.exists() else None)
    except Exception as e:  # noqa: BLE001
        q = {"quality_error": str(e)}
out["quality"] = q
(out_dir / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
j0, jl = (jobs[0], jobs[-1]) if jobs else ({}, {})
fail = "" if ok else " ERR " + (err or "; ".join(sum((j["errors"] for j in jobs), [])))[:200]
row = {"time": time.strftime("%m-%d %H:%M"), "name": name, "model": model, "clip": clip, "ok": ok, "total": j0.get("total"),
       "load": j0.get("load"), "text": j0.get("text"), "denoise": j0.get("denoise"), "s_step": j0.get("s_step"), "decode": j0.get("decode"),
       "warm_total": jl.get("total") if len(jobs) > 1 else "", "rss": out["rss_max"], "avail_min": out["avail_min"], "swap": out["swap_max"],
       "vram": j0.get("vram_resv"), "ssim": q.get("ssim", ""), "sharp": q.get("sharp_ratio", q.get("sharp", "")), "motion": q.get("motion_ratio", q.get("motion", "")),
       "ref": var.get("ref", ""), "note": var.get("note", "") + fail}
f = HERE / "results.csv"; new = not f.exists()
with f.open("a", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    if new: w.writeheader()
    w.writerow(row)
print("## ROW", json.dumps(row), flush=True)
os._exit(0 if ok else 1)
