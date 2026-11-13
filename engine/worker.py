"""The Morpheus render worker: one job per process, then it exits (G5: RAM fully free afterwards).

    engine\\.venv\\Scripts\\python.exe -m engine.worker <job.json>      (from the Morpheus root)

job.json  {"kind": "render", "settings": {...WanGP settings, see engine/jobs.py}, "out_dir": "...", "result": "...json"}
          {"kind": "upscale", "video": "...mp4", "upsampling": "relayvsr*1.5", "out_dir": "...", "result": "...json"}

stdout    lines starting with "@@ " are JSON for the caller:
            {"event": "progress", "phase": "inference", "step": 3, "total": 8}
            {"event": "done", "file": "...mp4", "seconds": 212.4}   or   {"event": "error", "message": "..."}
          everything else is WanGP's own log.
result    the same done/error object, written to the "result" path before the process ends.

No Windows-only code in here (the Windows environment variables come from the caller), so the same worker can run under
WSL2 later. WanGP does not end the process after a failed job (it hangs after writing the result): this worker always
ends itself with os._exit.
"""
import json
import os
import sys
import time
from pathlib import Path

WANGP = Path(__file__).with_name("wangp")


def say(obj: dict) -> None:
    sys.__stdout__.write("@@ " + json.dumps(obj) + "\n")  # the real stdout: WanGP captures sys.stdout as "stream" events
    sys.__stdout__.flush()


def run(job: dict) -> dict:
    sys.path.insert(0, str(WANGP))
    from shared.api import init  # noqa: E402  (WanGP's headless API, after the path is set)

    t0 = time.time()
    session = init(root=WANGP, output_dir=job["out_dir"], console_output=False,
                   cli_args=["--attention", "sdpa", "--profile", "4", "--preload", "0"])
    if job["kind"] == "render":
        settings = dict(job["settings"])
        floor = settings.pop("_frames_minimum", None)
        if floor:  # below the model's frames_minimum (H3 under 4.5 s); WanGP reads it from its model def
            import wgp
            wgp.get_model_def(settings["model_type"])["frames_minimum"] = int(floor)
        handle = session.submit_task(settings)
    elif job["kind"] == "upscale":
        handle = session.submit_media_postprocessing(job["video"], spatial_upsampling=job["upsampling"])
    else:
        raise ValueError(f"Unknown job kind {job['kind']!r}.")
    last = None
    for ev in handle.events.iter(timeout=0.5):
        # never echo "stream" events: WanGP captures this process' stdout as those, printing them feeds a loop
        if ev.kind != "progress":
            continue
        p = ev.data
        now = (p.phase, p.current_step, p.total_steps)
        if now != last:
            last = now
            say({"event": "progress", "phase": p.phase, "step": p.current_step, "total": p.total_steps})
    result = handle.result()
    if not result.success or not result.generated_files:
        msg = "; ".join(e.message for e in result.errors) or "WanGP made no file."
        return {"event": "error", "message": msg, "cancelled": result.cancelled}
    return {"event": "done", "file": str(Path(result.generated_files[-1]).resolve()), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        out = run(job)
    except Exception as e:  # noqa: BLE001  (anything: the caller gets it as text)
        import traceback
        traceback.print_exc(file=sys.__stderr__)
        out = {"event": "error", "message": f"{type(e).__name__}: {e}"}
    Path(job["result"]).write_text(json.dumps(out), encoding="utf-8")
    say(out)
    sys.__stdout__.flush()
    sys.__stderr__.flush()
    os._exit(0 if out["event"] == "done" else 1)


if __name__ == "__main__":
    main()
