"""Render video (H3 HyperFlow, LTX 2.3) in WSL2 (ROCm + Triton: Sol sparse attention, ~2x faster per step) with a warm worker that keeps the model loaded.

The only video engine since 2026-11-14 (the Windows worker engine/runner.py + worker.py is gone). One worker process lives inside
WSL (engine/wsl/warm_worker.py) and takes job after job over stdin; it keeps the last model in RAM/VRAM and ends itself after
keep_warm seconds without a job (so the ~23 GB go back to Windows). release() ends it at once: Stop, and before ComfyUI starts.

Setup inside WSL (not code): Ubuntu-24.04, ROCm 7.2.4 + librocdxg, /root/venv (torch 2.10 + triton 3.6), WanGP copy at
/root/wangp with the H3 + LTX checkpoints on ext4 (finetunes/ltx2_22B_dev_q6.json, loras/ltx2/), two AMD patches. See the Glia seed spike_wsl2_rocm_triton.
"""
import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path


ENGINE = Path(__file__).parent
DISTRO = "Ubuntu-24.04"
START = ENGINE / "wsl" / "start_warm.sh"
JOBS_DIR = ENGINE.parent / "output" / "engine" / "jobs"
RAM_WAIT = 90  # seconds to wait for the RAM (the last worker's memory takes a moment to come back)
WSL_MODELS = {"minimax_h3_ref2va_hyperflow", "ltx2_22B_dev_q6"}  # checkpoints on the WSL disk: all video renders here (2026-11-14)
SOL = {"overrides": "attention_mode='sol'", "attention_sparsity": 1.3}  # tau 1.3: 2x per step; 1.6/2.0 bought ~0-6 % more
SOL_MODELS = {"minimax_h3_ref2va_hyperflow", "minimax_h3_ref2va_hyperflow_gguf_q4_k_m", "minimax_h3_ref2va_hyperflow_gguf_q6_k"}  # Sol sparse attention is an H3 path in WanGP
NEED_GB = 20       # Windows RAM the VM needs to grow into for H3 (process ~22.7 GB inside a 24 GB cap)
READY_WAIT = 240   # VM boot + WanGP import


def linux_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    return f"/mnt/{s[0].lower()}{s[2:]}"


def free_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1024**3


def _wait_free(stopped, need: float, wait: float = RAM_WAIT) -> None:
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


def distro_running() -> bool:
    try:
        out = subprocess.run(["wsl.exe", "-l", "--running", "-q"], capture_output=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return DISTRO in out.decode("utf-16-le", errors="ignore")
    except (OSError, subprocess.SubprocessError):
        return True  # unsure: try the kill


class _Warm:
    def __init__(self):
        self.proc = None
        self.events = queue.Queue()
        self.lock = threading.Lock()   # one job at a time (server also holds the gpu lock)
        self.log = None
        self.keep_warm = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, keep_warm: int, preload_mb: int, stopped) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self.log = open(JOBS_DIR / "wsl_warm.log", "a", encoding="utf-8", errors="replace")
        self.events = queue.Queue()
        self.proc = subprocess.Popen(
            ["wsl.exe", "-d", DISTRO, "-u", "root", "--", "bash", linux_path(START), str(int(keep_warm)), str(int(preload_mb))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.keep_warm = keep_warm
        proc, log, events = self.proc, self.log, self.events

        def read():
            for line in proc.stdout:
                log.write(line); log.flush()
                if line.startswith("@@ "):
                    try:
                        events.put(json.loads(line[3:]))
                    except ValueError:
                        pass
            events.put({"event": "exit", "code": proc.wait()})

        threading.Thread(target=read, daemon=True).start()
        deadline = time.time() + READY_WAIT
        while True:
            if stopped():
                self.release(); raise ValueError("Stopped.")
            try:
                ev = self.events.get(timeout=1)
            except queue.Empty:
                if time.time() > deadline:
                    self.release(); raise RuntimeError(f"The WSL render worker did not start. Log: {log.name}")
                continue
            if ev.get("event") == "ready":
                return
            if ev.get("event") == "exit":
                raise RuntimeError(f"The WSL render worker ended at start (exit {ev.get('code')}). Log: {log.name}")

    def release(self) -> None:
        """Also kills a worker another process left behind (a restarted server, a bench script): it would hold ~23 GB.
        Only when the distro runs: a picture render must not boot the VM."""
        if self.proc is None and not distro_running():
            return
        try:
            subprocess.run(["wsl.exe", "-d", DISTRO, "-u", "root", "--", "bash", "-c", "pkill -f [w]arm_worker.py; sleep 2; sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"], timeout=20,
                           capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
        if self.proc is not None:
            kill_tree(self.proc)
        self.proc = None


WARM = _Warm()


def release() -> None:
    """End the warm worker now (frees its RAM and VRAM). Safe to call when none runs."""
    WARM.release()


def status() -> dict:
    return {"alive": WARM.alive(), "keep_warm": WARM.keep_warm}


def handles(work: dict) -> bool:
    return work.get("kind") == "render" and work.get("settings", {}).get("model_type") in WSL_MODELS


def wait_for_ram(stopped=lambda: False) -> None:
    if not WARM.alive():  # a warm worker of ours already holds its memory
        WARM.release()  # one another process left behind would hold ~23 GB we are about to wait for
        _wait_free(stopped, NEED_GB)


def run(job: dict, on_event, stopped, idle_timeout: float = 900, keep_warm: int = 600, preload_mb: int = 14000) -> dict:
    """Run one render in the warm WSL worker (started if needed) and return its done/error object, like runner.run."""
    with WARM.lock:
        if WARM.alive() and WARM.keep_warm != keep_warm:
            WARM.release()  # the idle timer is fixed at start
        if not WARM.alive():
            on_event({"phase": "starting_wsl"})
            WARM.start(keep_warm, preload_mb, stopped)
        name = uuid.uuid4().hex[:12]
        work = {**job, "id": name, "out_dir": str(JOBS_DIR / name), "result": str(JOBS_DIR / f"{name}.result.json")}
        if work.get("settings", {}).get("model_type") in SOL_MODELS:  # LTX rejects the sol override; an upscale has no settings
            work["settings"] = {**work["settings"], **SOL}
        WARM.proc.stdin.write(json.dumps(work) + "\n"); WARM.proc.stdin.flush()
        last = time.time()
        while True:
            if stopped():
                WARM.release(); raise ValueError("Stopped.")
            try:
                ev = WARM.events.get(timeout=0.5)
            except queue.Empty:
                if time.time() - last > idle_timeout:
                    WARM.release()
                    raise RuntimeError(f"The WSL render worker went silent for {idle_timeout // 60} minutes and was stopped.")
                continue
            last = time.time()
            kind = ev.get("event")
            if kind == "exit":
                WARM.proc = None
                raise RuntimeError(f"The WSL render worker ended mid-job (exit {ev.get('code')}). Log: {WARM.log.name}")
            if ev.get("id") != name:
                continue
            if kind == "progress":
                on_event(ev)
            elif kind == "done":
                return ev
            elif kind == "error":
                raise RuntimeError(f"Render failed: {ev.get('message')} (log: {WARM.log.name})")
