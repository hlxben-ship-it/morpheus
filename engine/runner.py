"""Server side of the render worker: start engine/worker.py in the engine venv, relay its progress lines, kill it on Stop.

Runs in the light Morpheus venv (stdlib + psutil). The Windows specifics live here, not in the worker: the venv path and
the environment variables WanGP needs on this PC (see the Know-How page wangp_on_amd_windows).
"""
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

ENGINE = Path(__file__).parent
ROOT = ENGINE.parent
PYTHON = ENGINE / ".venv" / "Scripts" / "python.exe"
JOBS_DIR = ROOT / "output" / "engine" / "jobs"
ENV = {"MIOPEN_FIND_MODE": "FAST",       # else MIOpen tunes for ages
       "HF_HUB_DISABLE_XET": "1",         # Xet downloads die ("Background writer channel closed")
       "NUMBA_DISABLE_JIT": "1",          # Smart App Control blocks numba's DLLs; WanGP imports numba via rembg
       "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
MIN_FREE_GB = 14  # the 22B models take 20-25 GB of RAM; the low-memory kills hit while loading right after a previous run
RAM_WAIT = 90     # seconds to wait for that much to be free (the last worker's memory takes a moment to come back)


def free_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1024**3


def wait_for_ram(stopped=lambda: False, need: float = MIN_FREE_GB, wait: float = RAM_WAIT) -> None:
    deadline = time.time() + wait
    while (free := free_gb()) < need:
        if stopped():
            raise ValueError("Stopped.")
        if time.time() > deadline:
            raise ValueError(f"Only {free:.1f} GB RAM free, the video models need about {need} GB to load. Close something (a game, "
                             "browser tabs, ComfyUI) and render again.")
        time.sleep(1)


def kill_tree(proc: subprocess.Popen) -> None:
    import psutil
    try:
        parent = psutil.Process(proc.pid)
        for p in parent.children(recursive=True) + [parent]:
            p.kill()
    except psutil.NoSuchProcess:
        pass


def run(job: dict, on_event, stopped, idle_timeout: float = 900) -> dict:
    """Run one job (see engine/worker.py for the shape) and return its done/error object. on_event gets every progress
    line. stopped() is polled; when it turns true the worker is killed, which frees its VRAM and RAM at once.
    idle_timeout: no line for this long = the worker hangs (WanGP can after a failure) and is killed."""
    if not PYTHON.is_file():
        raise RuntimeError(f"The engine venv is missing: {PYTHON}")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:12]
    job = {**job, "out_dir": str(JOBS_DIR / name), "result": str(JOBS_DIR / f"{name}.result.json")}
    job_file = JOBS_DIR / f"{name}.job.json"
    job_file.write_text(json.dumps(job, indent=1), encoding="utf-8")
    log = open(JOBS_DIR / f"{name}.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([str(PYTHON), "-m", "engine.worker", str(job_file)], cwd=ROOT, env={**os.environ, **ENV},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    last = {"t": time.time(), "out": None}

    def read():
        for line in proc.stdout:
            last["t"] = time.time()
            log.write(line)
            log.flush()
            if line.startswith("@@ "):
                try:
                    ev = json.loads(line[3:])
                except ValueError:
                    continue
                if ev.get("event") == "progress":
                    on_event(ev)
                else:
                    last["out"] = ev

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        while proc.poll() is None:
            if stopped():
                kill_tree(proc)
                raise ValueError("Stopped.")
            if time.time() - last["t"] > idle_timeout:
                kill_tree(proc)
                raise RuntimeError(f"The render worker went silent for {idle_timeout // 60} minutes and was stopped. Log: {log.name}")
            time.sleep(0.5)
        reader.join(5)
    finally:
        if proc.poll() is None:
            kill_tree(proc)
        log.close()
    out = last["out"]
    if out is None:
        try:
            out = json.loads(Path(job["result"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError(f"The render worker ended without a result (exit {proc.returncode}). Log: {log.name}")
    if out.get("event") != "done":
        raise RuntimeError(f"Render failed: {out.get('message')} (log: {log.name})")
    return out
