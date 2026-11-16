"""Warm render worker for WSL: one WanGP session that takes job after job and keeps the last model loaded in RAM/VRAM.

    python warm_worker.py [--keep-warm SECONDS]        (inside WSL, after env.sh)

stdin     one JSON job per line, the Morpheus engine job shape:
            {"kind": "render", "settings": {...}, "out_dir": "...", "result": "...json", "id": "..."}
            {"kind": "upscale", "video": "...", "upsampling": "relayvsr*1.5", "out_dir": "...", "result": "...", "id": "..."}
            {"kind": "image", "family": "sdxl"|"qwen_edit", "params": {...}, "count": N, "out_dir": "...", "id": "..."}   own pipelines
            {"kind": "image", "family": "h3_image", "settings": {...WanGP...}, ...}                                         H3-Image via WanGP
            {"kind": "unload"}   frees the model, stays alive        {"kind": "quit"}
stdout    "@@ {json}" lines: {"event": "ready"}, progress, done, error (each with the job "id"), {"event": "idle_exit"}
Exits by itself after --keep-warm seconds without a job (default 600), so RAM goes back to Windows when nobody renders.
Windows paths (C:/...) in a job are mapped to /mnt/c/...; the output file is reported as a Windows path again.
"""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__))
import amdspoof  # noqa: F401  (RDNA3 -> Ampere code paths in WanGP's Triton kernels; before any WanGP import)
import json, os, queue, threading, time
from pathlib import Path

WANGP = Path(os.environ.get("WARM_WANGP", "/root/wangp"))
KEEP_WARM = float(sys.argv[sys.argv.index("--keep-warm") + 1]) if "--keep-warm" in sys.argv else 600.0
# Big models (Qwen-Edit ~17 GB, H3 ~20 GB) leave Windows almost no RAM in the 24 GB VM: whatever keep-warm says, they go
# after HEAVY_IDLE without a job; a light model (SDXL) left behind keeps the worker for the rest of KEEP_WARM (2026-11-16)
HEAVY_IDLE = min(float(os.environ.get("WARM_HEAVY_IDLE", 600)), KEEP_WARM)
sys.argv = sys.argv[:1]  # WanGP parses sys.argv itself and rejects unknown flags


def say(obj):
    sys.__stdout__.write("@@ " + json.dumps(obj) + "\n"); sys.__stdout__.flush()


def to_linux(v):
    if isinstance(v, str) and len(v) > 2 and v[1] == ":" and v[2] in "/\\":
        return "/mnt/" + v[0].lower() + v[2:].replace("\\", "/")
    if isinstance(v, list): return [to_linux(x) for x in v]
    if isinstance(v, dict): return {k: to_linux(x) for k, x in v.items()}
    return v


def to_windows(p):
    p = str(p)
    if p.startswith("/mnt/") and len(p) > 6: return p[5].upper() + ":" + p[6:]
    return p


def mem():
    """Peak VRAM of this job and peak process RAM (GB), logged with each done event."""
    try:
        import resource, torch
        return {"vram_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
                "ram_peak_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)}
    except Exception:
        return {}


def main():
    os.chdir(WANGP); sys.path.insert(0, str(WANGP))
    from shared.api import init
    # --perc-reserved-mem-max: the share of RAM WanGP pins for transfers. Its Linux default (0.6) held ~12 GB for nothing here: the
    # step is compute-bound, 0.05 renders the same clip at the same speed with ~11 GB more RAM free (Glia seed challenge_spike_h3_tuning).
    session = init(root=WANGP, console_output=False,
                   cli_args=["--attention", "sdpa", "--profile", "4", "--preload", os.environ.get("WARM_PRELOAD", "14000"),
                             "--perc-reserved-mem-max", os.environ.get("WARM_PINNED", "0.05")])
    session._ensure_runtime()  # WanGP reads its flags when wgp is imported: importing it before init() dropped them all (until 2026-11-15)
    import wgp
    import gguf_triton  # noqa: F401  one Triton kernel per GGUF linear instead of PyTorch's dequantisation passes (Q4_K, Q6_K)
    jobs = queue.Queue()
    threading.Thread(target=lambda: [jobs.put(l) for l in sys.stdin], daemon=True).start()
    say({"event": "ready", "keep_warm": KEEP_WARM})
    heavy = False   # a big model is loaded (set after each job)
    idle_since = time.time()
    while True:
        if idle_since is None:  # a job just ended (every path below ends in continue): the idle clock starts now
            idle_since = time.time()
        wait = (HEAVY_IDLE if heavy else KEEP_WARM) - (time.time() - idle_since)
        try:
            line = jobs.get(timeout=max(1.0, wait))
        except queue.Empty:
            if heavy:
                heavy = False
                left = False
                try: wgp.release_model()
                except Exception: pass
                if "image.runner" in sys.modules:
                    left = sys.modules["image.runner"].release_heavy()
                if left and KEEP_WARM > HEAVY_IDLE:  # SDXL stays warm for the rest of keep-warm
                    say({"event": "heavy_released"}); continue
            say({"event": "idle_exit"}); break
        if not line.strip(): continue
        job = to_linux(json.loads(line)); jid = job.get("id")
        if job["kind"] == "quit": break
        idle_since = None
        if job["kind"] in ("image", "render"):  # what this job leaves loaded: Qwen-Edit and H3 (image or video) count as big
            mt = str((job.get("settings") or {}).get("model_type", ""))
            heavy = job.get("family") in ("qwen_edit", "h3_image", "krea2") or "h3" in mt
        if job["kind"] == "unload":
            try: wgp.release_model()
            except Exception as e: say({"event": "error", "id": jid, "message": f"unload: {e}"})
            say({"event": "unloaded", "id": jid}); continue
        try:  # page cache from the last job (or a big file copy) pushes the next one into swap: 4x slower (2026-11-14 bridge bench)
            os.sync(); open("/proc/sys/vm/drop_caches", "w").write("1")
        except OSError: pass
        t0 = time.time()
        try:
            import torch; torch.cuda.reset_peak_memory_stats()
        except Exception: pass
        try:
            if job.get("out_dir"):
                Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
                for k in ("save_path", "image_save_path", "audio_save_path", "mesh_save_path"):
                    wgp.server_config[k] = job["out_dir"]  # what shared.api does for output_dir, per job here
            if job["kind"] == "image" and job.get("family") != "h3_image":  # own pipelines (engine/wsl/image): SDXL, Qwen-Edit, Krea 2
                from image import runner as image_runner
                wgp.release_model()  # never a video model and an image model in RAM together (seed image_playground G2)
                res = image_runner.run(job, say)
                out = {"event": "done", "id": jid, "file": to_windows(res["files"][-1]), "files": [to_windows(f) for f in res["files"]],
                       "seeds": res["seeds"], "load_seconds": res["load_seconds"], "resident": res["resident"],
                       "seconds": round(time.time() - t0, 1), **mem()}
                if job.get("result"): Path(job["result"]).write_text(json.dumps(out), encoding="utf-8")
                say(out); continue
            if job["kind"] in ("render", "image"):  # an "image" job here is H3-Image: a WanGP task like a video render
                if "image.runner" in sys.modules:  # an own image model is loaded: free it before WanGP loads one
                    sys.modules["image.runner"].release_all()
                settings = dict(job["settings"])
                floor = settings.pop("_frames_minimum", None)
                if floor: wgp.get_model_def(settings["model_type"])["frames_minimum"] = int(floor)
                handle = session.submit_task(settings)
            elif job["kind"] == "upscale":
                handle = session.submit_media_postprocessing(job["video"], spatial_upsampling=job["upsampling"])
            else:
                raise ValueError(f"Unknown job kind {job['kind']!r}.")
            last = None
            for ev in handle.events.iter(timeout=0.5):
                if ev.kind != "progress": continue
                p = ev.data; now = (p.phase, p.current_step, p.total_steps)
                if now != last:
                    last = now; say({"event": "progress", "id": jid, "phase": p.phase, "step": p.current_step, "total": p.total_steps})
            result = handle.result()
            if not result.success or not result.generated_files:
                out = {"event": "error", "id": jid, "message": "; ".join(e.message for e in result.errors) or "WanGP made no file."}
            else:
                files = [to_windows(Path(f).resolve()) for f in result.generated_files]
                out = {"event": "done", "id": jid, "file": files[-1], "files": files, "seconds": round(time.time() - t0, 1), **mem()}
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(file=sys.__stderr__)
            out = {"event": "error", "id": jid, "message": f"{type(e).__name__}: {e}"}
        if job.get("result"): Path(job["result"]).write_text(json.dumps(out), encoding="utf-8")
        say(out)
    sys.__stdout__.flush(); os._exit(0)


if __name__ == "__main__":
    main()
