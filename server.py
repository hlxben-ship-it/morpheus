"""Morpheus web UI — http://127.0.0.1:5196

Serves the page in web/, generated images from output/, and a small JSON API around
morpheus.run() and morpheus.run_edit(). Starts ComfyUI on demand if it is not already running.
"""
import base64
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

import morpheus

PORT = 5196
WEB = morpheus.ROOT / "web"
RESOLUTIONS = [
    {"label": "Square", "width": 1024, "height": 1024},
    {"label": "Landscape 3:2", "width": 1216, "height": 832},
    {"label": "Wide 16:9", "width": 1344, "height": 768},
    {"label": "Portrait 2:3", "width": 832, "height": 1216},
    {"label": "Tall 9:16", "width": 768, "height": 1344},
    {"label": "Small", "width": 768, "height": 768},
]

gpu = threading.Lock()  # ComfyUI renders one image at a time anyway; this keeps requests in order
comfy_lock = threading.Lock()


def other_vram_gb() -> float:
    """Dedicated VRAM in use by every process on the card; called while ComfyUI is down, so it is everyone else."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "((Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | Measure CookedValue -Sum).Sum"],
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        return float(r.stdout.strip().replace(",", ".")) / 1024 ** 3
    except ValueError:
        return 2.0


def ensure_comfy(timeout: float = 300) -> None:
    """Start ComfyUI in the background if it is not answering, then wait until it is."""
    with comfy_lock:
        if morpheus.comfy_up():
            return
        log = open(morpheus.ROOT / "comfyui.log", "ab")
        # ComfyUI can't see what the desktop and browser hold on the card (measured 1.5-7 GB); without this reserve
        # it plans with ~18 GB, Windows spills the overflow into shared memory and a step takes 81 s instead of 1.2 s
        reserve = other_vram_gb() + 1.0
        # our route for the loaded models, refreshed every start so an update lands
        shutil.copy2(morpheus.ROOT / "comfy_ext" / "morpheus_status.py", morpheus.ROOT / "ComfyUI" / "custom_nodes")
        subprocess.Popen(
            [sys.executable, str(morpheus.ROOT / "ComfyUI" / "main.py"), "--listen", "127.0.0.1", "--port", "8188",
             "--disable-auto-launch", "--preview-method", "latent2rgb", "--reserve-vram", f"{reserve:.1f}"],
            stdout=log, stderr=subprocess.STDOUT, cwd=morpheus.ROOT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.time() + timeout
        while time.time() < deadline:
            if morpheus.comfy_up():
                return
            time.sleep(1)
        raise RuntimeError("ComfyUI did not start, see comfyui.log.")


# Claude runs from its own empty folder: inside C:\dev it would pick up the projects' CLAUDE.md files,
# and --resume only finds a session from the folder it was started in.
# still named after Imageer: sessions started there before the rename only resume from the same folder
CHAT_DIR = Path(tempfile.gettempdir()) / "imageer-chat"

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": ["generate", "edit", "none"]},
        "prompt": {"type": "string"},
        "model": {"type": "string"},
        "style": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
    "required": ["reply", "action", "prompt", "model", "style", "width", "height"],
}


def chat_system_prompt() -> str:
    models = "\n".join(f"- {k}: {m['label']}, {m['note']}" for k, m in morpheus.MODELS.items() if morpheus.installed(k))
    styles = "\n".join(f"- {k}: {s['label']}" for k, s in morpheus.STYLES.items())
    sizes = ", ".join(f"{r['width']}x{r['height']} ({r['label']})" for r in RESOLUTIONS)
    return f"""You are the picture assistant in Morpheus, a local image generator on the user's GPU.
The user describes what they want to see.

Decide for every message:
- action "generate": a new picture. prompt is a detailed English image prompt: subject, setting, light, art style, composition.
  For follow-ups ("brighter", "with a hat") rewrite the previous prompt accordingly.
- action "edit": the user wants an existing picture changed ("make it night", "give her a hat", "full body"): the
  attached picture if there is one, otherwise the latest picture in this session. prompt is then a short English
  change instruction ("make it night, keep everything else"). model and style empty; width and height 0.
  Prefer "edit" over "generate" whenever the user changes something in the picture they just got; "generate" is for
  a different picture or when the change is so big that little of the old one would stay.
- With an attached picture and a wish for a new picture in its style or with its subject: action "generate", and
  describe the attached picture's style, colours, subject and composition in detail in the prompt.
- Always look at an attached picture first with the Read tool.
- action "none": only answer or ask, no picture. Leave prompt, model and style empty.

Models (only these are installed):
{models}
The default for new pictures is z-image-turbo (about 20 s, photoreal and good at painting styles). flux2-klein-4b is
just as fast, an alternative with a different look. noobai-xl only for anime and illustration (then write the prompt
as comma-separated Danbooru tags, e.g. "1girl, elf, silver hair, forest, night"). qwen-image-2512 takes 2 to 4
minutes: use it only when the picture needs lettering or the user explicitly wants the highest quality, and say that
it takes longer. realvis-xl and chroma only when the user names them.
If the user names a specific model, use exactly that one.

Styles:
{styles}
"pixel" and "gameboy" only when the user wants a retro or pixel look; the pixel post-processing then runs automatically.
Otherwise "none", and put the wanted style into the prompt.

Formats: {sizes}. Use 1024x1024 unless something speaks against it.

reply: one or two sentences about what you are doing or what you need to know. No Markdown.
Always reply in English, whatever language the user writes in."""


REFINE_MAX = 5
REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["good", "retry"]},
        "critique": {"type": "string"},
        "prompt": {"type": "string"},
        "model": {"type": "string"},
        "style": {"type": "string"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
    "required": ["verdict", "critique", "prompt", "model", "style", "width", "height"],
}


def refine_prompt() -> str:
    return chat_system_prompt().split("Decide for every message:")[0] + f"""You check a generated picture against a target picture.
Compare the art style (lines, shading, pixel size, palette), the subject and its details (hairstyle, hair colour,
clothing, accessories, pose, background) and the composition.

verdict "good": the result hits the style and the essential details. Be honest, but not pedantic.
verdict "retry": say in critique what is furthest off, and write an improved English prompt that corrects exactly
that. You may change model, style, width, height if it helps (e.g. another model for the style, or style "none"
instead of "pixel" when the pixel post-processing destroys the look).
With "good", repeat prompt, model, style, width, height unchanged.

critique: one or two sentences, always in English, no Markdown.

Models:
""" + "\n".join(f"- {k}: {m['label']}, {m['note']}" for k, m in morpheus.MODELS.items() if morpheus.installed(k)) + \
        "\n\nStyles:\n" + "\n".join(f"- {k}: {s['label']}" for k, s in morpheus.STYLES.items()) + \
        "\n\nFormats: " + ", ".join(f"{r['width']}x{r['height']}" for r in RESOLUTIONS)


def ask_claude(message: str, session: str | None, ghost: bool = False, see: bool = False,
               system: str | None = None, schema: dict | None = None) -> tuple[dict, str]:
    """One chat turn through the Claude Code CLI; returns the structured answer and the session id.

    Ghost turns are not saved by the CLI and cannot be resumed; the caller passes the conversation in the message.
    """
    CHAT_DIR.mkdir(exist_ok=True)
    cmd = [shutil.which("claude") or "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema or CHAT_SCHEMA),
           "--tools", "Read" if see else "",  # Read only to look at a picture copied into CHAT_DIR
           "--model", "sonnet", "--system-prompt", system or chat_system_prompt()]
    if ghost:
        cmd.append("--no-session-persistence")
    elif session:
        cmd += ["--resume", session]
    p = subprocess.run(cmd, input=message, capture_output=True, text=True, encoding="utf-8", cwd=CHAT_DIR, timeout=180,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Claude did not answer: {(p.stderr or p.stdout).strip()[:300]}")
    if out.get("is_error") or not out.get("structured_output"):
        raise RuntimeError(f"Claude error: {str(out.get('result'))[:300]}")
    return out["structured_output"], out["session_id"]


# Ghost mode: results live only here, in memory, served under /output/ghost/. Nothing reaches the disk,
# and a restart of the server forgets them. Oldest go first once the store is full.
GHOST: dict[str, bytes] = {}
GHOST_MAX = 40
ghost_lock = threading.Lock()


def ghost_keep(meta: dict) -> dict:
    """Move a ghost run's images into the store; the metadata then points at them like at files."""
    pictures = meta.pop("pictures")
    with ghost_lock:
        for key in ("raw", "image"):
            buf = BytesIO()
            pictures[key].save(buf, "PNG")
            GHOST[meta[key]] = buf.getvalue()
            meta[key] = "ghost/" + meta[key]
        while len(GHOST) > GHOST_MAX:
            GHOST.pop(next(iter(GHOST)))
    return {**meta, "ghost": True}


def result_picture(meta: dict) -> Image.Image:
    """The unprocessed picture of a result, from the ghost store or from output/."""
    ref = meta.get("raw") or meta["image"]
    return ghost_image(ref) if ref.startswith("ghost/") else Image.open(morpheus.OUTPUT / ref)


def ghost_image(ref: str) -> Image.Image:
    with ghost_lock:
        data = GHOST.get(ref.removeprefix("ghost/"))
    if data is None:
        raise ValueError("The ghost picture is gone.")
    return Image.open(BytesIO(data))


# Face swap runs in its own venv (insightface + DirectML), as a subprocess that prints "progress i total" per frame.
# Jobs live here so the page can poll them; a swap takes minutes, longer than one request should stay open.
FACESWAP_PY = morpheus.ROOT / ".venv-faceswap" / "Scripts" / "python.exe"
SWAPS: dict[str, dict] = {}
# Videos arrive raw through /api/upload (base64 in JSON chokes on long clips) and wait here for their job.
UPLOADS = Path(tempfile.gettempdir()) / "morpheus-uploads"


def faceswap_job(job: dict, photo: bytes, video: Path, quality: str, enhance: bool, hair: bool, ghost: bool,
                 span: tuple[float, float] | None = None, upscale: int = 1) -> None:
    t0 = time.time()
    try:
        with gpu, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "photo.png").write_bytes(photo)
            shutil.move(video, tmp / "video.mp4")
            name = f"{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            out = tmp / name if ghost else morpheus.OUTPUT / "faceswap" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            job["state"] = "running"
            cmd = [str(FACESWAP_PY), "-I", str(morpheus.ROOT / "faceswap.py"), str(tmp / "photo.png"), str(tmp / "video.mp4"),
                   str(out), "--quality", quality]
            if not enhance:
                cmd.append("--no-enhance")
            if hair:
                cmd.append("--hair")
            if upscale > 1:
                cmd += ["--upscale", str(upscale)]
            if span:
                cmd += ["--start", f"{span[0]:.3f}", "--end", f"{span[1]:.3f}"]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                 errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in p.stdout:
                parts = line.split()
                if parts[:1] == ["progress"]:
                    job["frame"], job["total"] = int(parts[1]), int(parts[2])
                elif parts[:1] == ["done"]:
                    job["missed"] = int(parts[2])
            err = p.stderr.read()
            code = p.wait()
            if code != 0:
                msg = next((l.removeprefix("error: ") for l in err.splitlines() if l.startswith("error: ")), None)
                raise RuntimeError(msg or "The face swap stopped: " + err.strip().splitlines()[-1][:300])
            if ghost:
                with ghost_lock:
                    GHOST[name] = out.read_bytes()
                job["video"] = "ghost/" + name
            else:
                job["video"] = "faceswap/" + name
        job["seconds"] = round(time.time() - t0)
        job["state"] = "done"
    except Exception as e:
        traceback.print_exc()
        job["error"] = str(e)
        job["state"] = "error"


def history(limit: int = 60) -> list[dict]:
    items = []
    for f in sorted(morpheus.OUTPUT.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            items.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return items


# Chat, generate and edit started with "background": the result waits here for the page, also across a reload.
JOBS: dict[str, dict] = {}


def progress_for(req: dict):
    """The callback a background job hands to morpheus.run: sampler steps and the newest live preview frame."""
    job = req.get("_job")
    if job is None:
        return None

    def update(step=None, total=None, preview=None):
        if step is not None:
            job["step"], job["total"] = step, total
        if preview is not None:
            job["_preview"] = preview
            job["previews"] = job.get("previews", 0) + 1  # the page reloads the frame when this changes
    return update


def background(job: dict, handler, req: dict) -> None:
    req["_job"] = job
    try:
        job["result"] = handler(req)
        job["state"] = "done"
    except Exception as e:
        if not isinstance(e, ValueError):
            traceback.print_exc()
        job["error"] = str(e)
        job["state"] = "error"


# GPU load and VRAM from the Windows performance counters, which see every process (ROCm, the desktop, games),
# not just ComfyUI's own torch. One long-lived PowerShell samples them; spawning one per poll takes seconds.
GPU_STATS = {"load": None, "vram": None, "procs": []}
GPU_SAMPLER = r"""
$ErrorActionPreference = 'SilentlyContinue'
while (Get-Process -Id SERVER_PID) {  # ends with the server, also when it is killed
  $c = (Get-Counter '\GPU Engine(*)\Utilization Percentage', '\GPU Adapter Memory(*)\Dedicated Usage',
                    '\GPU Process Memory(*)\Local Usage').CounterSamples
  $eng = @{}; $vram = [double]0; $mem = @{}
  foreach ($s in $c) {
    if ($s.Path -match 'luid_(\w+?_\w+?)_phys_\d+_eng_(\d+)') { $eng[$Matches[1] + $Matches[2]] += $s.CookedValue }
    elseif ($s.Path -match 'process memory\(pid_(\d+)_') { $mem[[int]$Matches[1]] += $s.CookedValue }
    elseif ($s.Path -like '*dedicated usage') { $vram = [Math]::Max($vram, $s.CookedValue) }
  }
  # per process "Local Usage" is what sits in VRAM; its "Dedicated Usage" counts what the desktop maps from
  # other programs again (dwm: 7 GB against 0.7 GB), so those never added up to the adapter's total
  $load = ($eng.Values | Measure-Object -Maximum).Maximum
  $top = $mem.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 3 | ForEach-Object {
    $p = Get-Process -Id $_.Key
    $name = if ($p.Description) { $p.Description } else { $p.ProcessName }
    if ($p.ProcessName -eq 'python' -and (Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Key)").CommandLine -match 'ComfyUI.main\.py') { $name = 'ComfyUI' }
    @{ pid = $_.Key; name = $name; exe = $p.ProcessName; bytes = [long]$_.Value } }
  @{ load = [int][Math]::Min(100, [double]$load); vram = [long]$vram; procs = @($top) } | ConvertTo-Json -Compress -Depth 3
  Start-Sleep -Milliseconds 600
}
"""


def sample_gpu() -> None:
    """Task Manager's view: an adapter's load is its busiest engine, summed over processes; VRAM per process too."""
    p = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", GPU_SAMPLER.replace("SERVER_PID", str(os.getpid()))],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in p.stdout:
        try:
            GPU_STATS.update(json.loads(line))
        except ValueError:
            pass


class GpuTemp:
    """Temperatures from AMD's driver library (ADL's PMLog, as the Adrenalin overlay reads them). Windows has no
    counter for them. None where the library or the sensor is missing, e.g. on another vendor's card."""
    SENSORS = {"edge": 8, "mem": 9, "hotspot": 27, "fan": 15}  # ADL_PMLOG_* indices; fan in percent

    def __init__(self):
        self.lock, self.ctx, self.adapter = threading.Lock(), None, None
        try:
            import ctypes as C
            self.C, self.adl = C, C.WinDLL("atiadlxx.dll")
            self._buffers = []
            self._alloc = C.WINFUNCTYPE(C.c_void_p, C.c_int)(self._malloc)
            ctx = C.c_void_p()
            if self.adl.ADL2_Main_Control_Create(self._alloc, 1, C.byref(ctx)) != 0:
                return
            n = C.c_int()
            self.adl.ADL2_Adapter_NumberOfAdapters_Get(ctx, C.byref(n))
            for i in range(n.value):
                active = C.c_int()
                self.adl.ADL2_Adapter_Active_Get(ctx, i, C.byref(active))
                if active.value:
                    self.ctx, self.adapter = ctx, i
                    break

            class Sensor(C.Structure):
                _fields_ = [("supported", C.c_int), ("value", C.c_int)]

            class PMLog(C.Structure):
                _fields_ = [("size", C.c_int), ("sensors", Sensor * 256)]
            self.PMLog = PMLog
        except (OSError, AttributeError):
            self.ctx = None

    def _malloc(self, size):
        buf = self.C.create_string_buffer(size)
        self._buffers.append(buf)  # ADL keeps what it allocates for the life of the context
        return self.C.addressof(buf)

    def read(self) -> dict:
        if self.ctx is None:
            return {}
        with self.lock:
            out = self.PMLog()
            if self.adl.ADL2_New_QueryPMLogData_Get(self.ctx, self.adapter, self.C.byref(out)) != 0:
                return {}
        return {k: out.sensors[i].value for k, i in self.SENSORS.items() if out.sensors[i].supported}


GPU_TEMP = GpuTemp()


VRAM_TOTAL = None  # from ComfyUI once, it doesn't change


def status() -> dict:
    """ComfyUI's state and the GPU's, for the header."""
    global VRAM_TOTAL
    comfy, queue, models = False, 0, []
    try:
        with urllib.request.urlopen(morpheus.SERVER + "/queue", timeout=1) as r:
            q = json.loads(r.read())
        comfy, queue = True, len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        try:
            with urllib.request.urlopen(morpheus.SERVER + "/morpheus/loaded", timeout=1) as r:
                models = json.loads(r.read())
        except OSError:
            models = None  # ComfyUI started before comfy_ext was there
        if VRAM_TOTAL is None:
            with urllib.request.urlopen(morpheus.SERVER + "/system_stats", timeout=1) as r:
                VRAM_TOTAL = json.loads(r.read())["devices"][0]["vram_total"]
    except (OSError, KeyError, IndexError, ValueError):
        pass
    running = sum(1 for j in JOBS.values() if j.get("state") in ("running", "waiting")) + \
        sum(1 for j in SWAPS.values() if j.get("state") in ("running", "waiting"))
    return {"comfy": comfy, "busy": gpu.locked() or queue > 0, "queue": queue, "jobs": running,
            "load": GPU_STATS["load"], "vram": GPU_STATS["vram"], "vram_total": VRAM_TOTAL, "procs": GPU_STATS["procs"],
            "temp": GPU_TEMP.read(), "models": models}


def copy_picture(source: str, dest: Path) -> None:
    """A gallery or ghost picture copied to dest, for Claude to read."""
    if source.startswith("ghost/"):
        ghost_image(source).save(dest)
        return
    src = (morpheus.OUTPUT / source).resolve()
    if morpheus.OUTPUT.resolve() not in src.parents or not src.is_file():
        raise ValueError("The attached picture no longer exists.")
    dest.write_bytes(src.read_bytes())


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the console quiet apart from errors
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, base: Path):
        path = path.resolve()
        if base.resolve() not in path.parents or not path.is_file():
            return self.send_error(404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/":
            return self.send_file(WEB / "index.html", WEB)
        if route in ("/logo.png", "/favicon.png", "/city.png"):
            return self.send_file(WEB / route[1:], WEB)
        if route == "/api/options":
            models = [{"key": k, "label": m["label"], "note": m["note"], "installed": morpheus.installed(k)}
                      for k, m in morpheus.MODELS.items()]
            edit_models = [{"key": k, "label": m["label"], "note": m["note"], "installed": morpheus.installed(k)}
                           for k, m in morpheus.EDIT_MODELS.items()]
            styles = [{"key": k, "label": s["label"]} for k, s in morpheus.STYLES.items()]
            return self.send_json({"models": models, "edit_models": edit_models, "styles": styles, "resolutions": RESOLUTIONS})
        if route == "/api/status":
            return self.send_json(status())
        if route == "/api/history":
            return self.send_json(history())
        if route.startswith("/api/job/"):
            job_id, _, part = route.removeprefix("/api/job/").partition("/")
            job = JOBS.get(job_id)
            if not job:
                return self.send_error(404)
            if part == "preview":  # the live frame of the picture forming
                data = job.get("_preview")
                if not data:
                    return self.send_error(404)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(data)
            return self.send_json({k: v for k, v in job.items() if not k.startswith("_")})
        if route.startswith("/api/faceswap/"):
            job = SWAPS.get(route.removeprefix("/api/faceswap/"))
            return self.send_json(job) if job else self.send_error(404)
        if route.startswith("/output/ghost/"):
            with ghost_lock:
                data = GHOST.get(urllib.parse.unquote(route.removeprefix("/output/ghost/")))
            if data is None:
                return self.send_error(404)
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(route)[0] or "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")  # keep it out of the browser's disk cache too
            self.end_headers()
            return self.wfile.write(data)
        if route.startswith("/output/"):
            return self.send_file(morpheus.OUTPUT / route.removeprefix("/output/"), morpheus.OUTPUT)
        self.send_error(404)

    def do_POST(self):
        if self.path == "/api/upload":  # raw bytes, not JSON
            return self.upload()
        handlers = {"/api/generate": self.generate, "/api/chat": self.chat, "/api/edit": self.edit,
                    "/api/ghost/clear": self.ghost_clear, "/api/faceswap": self.faceswap, "/api/delete": self.delete}
        if self.path not in handlers:
            return self.send_error(404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            handler = handlers[self.path]
            if req.get("background") and self.path in ("/api/generate", "/api/chat", "/api/edit"):
                # runs on after the page is gone (a reload mid-render); the page polls /api/job/<id> and finds it again
                job_id = f"{time.time_ns():x}"
                JOBS[job_id] = {"state": "running"}
                threading.Thread(target=background, args=(JOBS[job_id], handler, req), daemon=True).start()
                return self.send_json({"job": job_id})
            self.send_json(handler(req))
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:  # report anything else to the page instead of dropping the connection
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def chat(self, req):
        message = str(req.get("message", "")).strip()
        if not message:
            raise ValueError("Please write something.")
        ghost = bool(req.get("ghost"))
        if ghost and req.get("context"):
            # No saved session to resume, so the page sends the ghost conversation so far.
            lines = [f"{'User' if m.get('role') == 'user' else 'You'}: {m.get('text', '')}" for m in req["context"][-20:]]
            message = "Conversation so far:\n" + "\n".join(lines) + "\n\nNew message:\n" + message
        last = req.get("last")
        if isinstance(last, dict) and last.get("prompt"):
            # the picture the user sees last, possibly made outside the chat (Manual, Edit, Again)
            message = (f"[Latest picture in this session (action \"edit\" changes it): model {last.get('model')}, "
                       f"{last.get('width')}x{last.get('height')}, Prompt: {last['prompt']}]\n\n{message}")
        # An attached picture: an upload (data URL) or a gallery file. Claude reads a copy in its working folder;
        # the copy goes right after the turn, so a ghost picture is never left behind.
        attach = {k: req[k] for k in ("upload", "source") if req.get(k)}
        seen = None
        if attach:
            CHAT_DIR.mkdir(exist_ok=True)
            seen = CHAT_DIR / f"attached-{uuid.uuid4().hex[:8]}.png"
            if "upload" in attach:
                seen.write_bytes(base64.b64decode(str(attach["upload"]).split(",", 1)[-1]))
            else:
                copy_picture(attach["source"], seen)
            # spelled out here, because a resumed conversation may come from before "edit" existed and assume it can't
            message = (f"[Attached picture: {seen.name} — look at it with the Read tool. You can edit this picture "
                       f"directly: action \"edit\" changes exactly this picture and keeps the rest. Use \"edit\" when the "
                       f"user wants this picture changed; \"generate\" only for a new picture in its style or with its subject.]"
                       f"\n\n{message}")
        # Nothing attached: Claude may still look at the latest picture, e.g. "is her eye broken?"
        latest = None
        if not seen and isinstance(last, dict) and last.get("image"):
            latest = CHAT_DIR / f"latest-{uuid.uuid4().hex[:8]}.png"
            try:
                CHAT_DIR.mkdir(exist_ok=True)
                copy_picture(str(last["image"]), latest)
                message = (f"[The latest picture is {latest.name}. Look at it with the Read tool when the user asks "
                           f"about it or a change needs you to see it; otherwise don't.]\n\n{message}")
            except Exception:
                latest = None  # deleted meanwhile: chat on without it
        try:
            # per turn, because a resumed conversation that began in German outweighs the system prompt
            answer, session = ask_claude("[Reply in English.]\n\n" + message, req.get("session") or None, ghost,
                                         see=bool(seen or latest))
            result = {"session": None if ghost else session, "reply": answer["reply"], "action": answer["action"], "image": None}
            if answer["action"] == "edit" and not attach and isinstance(last, dict) and last.get("image"):
                attach = {"source": str(last["image"])}  # no picture attached: the edit is of the latest one
            if answer["action"] == "edit" and attach:
                # the normal edit route, so the result links back to its source like any edit
                result["image"] = self.edit({**attach, "prompt": answer["prompt"], "ghost": ghost, "_job": req.get("_job")})
                result["image"]["edit"] = True
            elif answer["action"] == "generate":
                result["image"] = self.render(answer, ghost, req)
                if seen:
                    # a reference picture is the target: look at each result and re-prompt until it matches
                    result.update(self.refine(seen, answer, result["image"], ghost, req))
        finally:
            for f in (seen, latest):
                if f:
                    f.unlink(missing_ok=True)
        return result

    def render(self, answer: dict, ghost: bool, req: dict) -> dict:
        model = answer["model"] if answer["model"] in morpheus.MODELS and morpheus.installed(answer["model"]) else "z-image-turbo"
        style = answer["style"] if answer["style"] in morpheus.STYLES else "none"
        size = next((r for r in RESOLUTIONS if r["width"] == answer["width"] and r["height"] == answer["height"]), RESOLUTIONS[0])
        ensure_comfy()
        with gpu:
            meta = morpheus.run(answer["prompt"], style, model, size["width"], size["height"], ghost=ghost,
                                progress=progress_for(req))
        return ghost_keep(meta) if ghost else meta

    def refine(self, target: Path, answer: dict, first: dict, ghost: bool, req: dict) -> dict:
        """Compare each result with the target picture and render again until Claude is happy, at most REFINE_MAX rounds.

        Every round's picture stays in the trail, so the user sees how it converged. Each judgement is a fresh,
        unsaved Claude call: it sees only target, result and prompt, not the chat it came from.
        """
        job = req.get("_job") or {}
        trail, current, note = [], first, ""
        for round_ in range(1, REFINE_MAX + 1):
            job["status"] = f"Round {round_} of {REFINE_MAX}: comparing the result with the target …"
            shot = CHAT_DIR / f"result-{uuid.uuid4().hex[:8]}.png"
            try:
                result_picture(current).save(shot)
                verdict, _ = ask_claude(
                    f"Target: {target.name}\nResult: {shot.name}\nPrompt of the result: {current['prompt']}\n"
                    f"Model: {current['model']}, style: {current.get('style', 'none')}, {current['width']}x{current['height']}\n\n"
                    "Look at both pictures with the Read tool and compare them.", None, ghost=True, see=True,
                    system=refine_prompt(), schema=REFINE_SCHEMA)
            finally:
                shot.unlink(missing_ok=True)
            note = verdict["critique"]
            if verdict["verdict"] == "good":
                return {"trail": trail, "image": current, "reply": f"{answer['reply']}\n\nRound {round_}: {note}"}
            if round_ == REFINE_MAX:
                break
            trail.append({"image": current, "note": f"Round {round_}: {note}"})
            job["status"] = f"Round {round_ + 1} of {REFINE_MAX}: {note[:140]}"
            job.update(step=0, total=0)
            current = self.render({**answer, **{k: verdict[k] for k in ("prompt", "model", "style", "width", "height")}},
                                  ghost, req)
        return {"trail": trail, "image": current,
                "reply": f"{answer['reply']}\n\nReached {REFINE_MAX} rounds. Last: {note}\n"
                         "Shall I keep tuning the prompt?"}

    def generate(self, req):
        """The manual form: exact prompt and settings, no Claude in between."""
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Please enter a prompt.")
        ghost = bool(req.get("ghost"))
        ensure_comfy()
        with gpu:
            meta = morpheus.run(
                prompt, req.get("style", "none"), req.get("model", "z-image-turbo"),
                int(req.get("width", 1024)), int(req.get("height", 1024)),
                int(req["seed"]) if str(req.get("seed", "")).strip() else None, ghost=ghost, progress=progress_for(req))
        return ghost_keep(meta) if ghost else meta

    def edit(self, req):
        """Edit by instruction. The source is a gallery image (path under output/) or an upload as a data URL."""
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Please describe what should change.")
        model = req.get("model", "flux2-klein-4b-edit")
        if model not in morpheus.EDIT_MODELS:
            raise ValueError(f"Unknown model '{model}'.")
        ghost = bool(req.get("ghost"))
        ref = str(req.get("source", ""))
        source_ref = None
        if req.get("upload"):
            data = base64.b64decode(str(req["upload"]).split(",", 1)[-1])
            if ghost:
                source = Image.open(BytesIO(data))  # a ghost upload never touches the disk
            else:
                # Uploads live in output/uploads, so results can link back to them; history() only reads output/*.json.
                source = morpheus.OUTPUT / "uploads" / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(data)
        elif ref.startswith("ghost/"):
            source = ghost_image(ref)
            source_ref = ref if ghost else ""  # a kept result must not link to a ghost that will vanish
        else:
            source = (morpheus.OUTPUT / ref).resolve()
            if morpheus.OUTPUT.resolve() not in source.parents or not source.is_file():
                raise ValueError("Please pick a source picture.")
        ensure_comfy()
        with gpu:
            meta = morpheus.run_edit(prompt, source, model,
                                    int(req["seed"]) if str(req.get("seed", "")).strip() else None,
                                    ghost=ghost, source_ref=source_ref, progress=progress_for(req))
        return ghost_keep(meta) if ghost else meta

    def upload(self):
        """Store a raw upload (the video) in chunks and hand back its id."""
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return self.send_json({"error": "Empty upload."}, 400)
        UPLOADS.mkdir(exist_ok=True)
        upload_id = f"{time.time_ns():x}"
        with open(UPLOADS / upload_id, "wb") as f:
            while length:
                chunk = self.rfile.read(min(length, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                length -= len(chunk)
        self.send_json({"id": upload_id})

    def faceswap(self, req):
        """Start a face swap job: the photo as a data URL, the video as an id from /api/upload; the page polls
        /api/faceswap/<id>."""
        if not FACESWAP_PY.is_file():
            raise ValueError("Face swap is not installed (.venv-faceswap is missing, see README).")
        video = UPLOADS / str(req.get("video_id", "")).strip("./\\")
        if not req.get("photo") or not req.get("video_id") or not video.is_file():
            raise ValueError("Please pick a photo and a video.")
        photo = base64.b64decode(str(req["photo"]).split(",", 1)[-1])
        job_id = f"{time.time_ns():x}"
        quality = req.get("quality") if req.get("quality") in ("fast", "premium") else "high"
        span = None  # only a section of the video, in seconds
        if req.get("start") is not None and req.get("end") is not None:
            start, end = float(req["start"]), float(req["end"])
            if end - start < 0.1:
                raise ValueError("The chosen section is too short.")
            span = (max(0.0, start), end)
        upscale = req.get("upscale") if req.get("upscale") in (2, 4) else 1
        SWAPS[job_id] = job = {"state": "waiting", "frame": 0, "total": 0, "quality": quality, "hair": bool(req.get("hair")),
                               "ghost": bool(req.get("ghost")), "upscale": upscale}
        threading.Thread(target=faceswap_job, args=(job, photo, video, quality, req.get("enhance", True) is not False,
                                                    job["hair"], job["ghost"], span, upscale),
                         daemon=True).start()
        return {"job": job_id}

    def ghost_clear(self, req):
        with ghost_lock:
            GHOST.clear()
        return {"ok": True}

    def delete(self, req):
        """Delete one gallery picture from disk: the picture, its raw version and the sidecar. Edit sources in
        output/uploads stay, other results may still link to them."""
        ref = str(req.get("image", ""))
        if ref.startswith("ghost/"):
            with ghost_lock:
                GHOST.pop(ref.removeprefix("ghost/"), None)
            return {"ok": True}
        out = morpheus.OUTPUT.resolve()
        for sidecar in out.glob("*.json"):
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if ref not in (meta.get("image"), meta.get("raw")):
                continue
            for name in {meta.get("image"), meta.get("raw")} - {None}:
                f = (out / name).resolve()
                if f.parent == out:  # never outside output/
                    f.unlink(missing_ok=True)
            sidecar.unlink()
            print(f"Morpheus: deleted {sidecar.stem} ({time.strftime('%H:%M:%S')})", flush=True)
            return {"ok": True}
        raise ValueError("This picture is not in the gallery.")


def warm_up() -> None:
    """Start ComfyUI and render one tiny throwaway image with the default model, so the first real prompt
    doesn't pay for reading ~20 GB of weights from disk (123 s cold vs ~20 s warm)."""
    try:
        ensure_comfy()
        with gpu:
            morpheus.generate(morpheus.MODELS["z-image-turbo"]["workflow"]("warm up", 256, 256, 0, 1), ghost=True)
        print("Morpheus: ComfyUI is warm.")
    except Exception as e:  # the first real render just pays the cold start then
        print(f"Morpheus: warm-up failed ({e}).")


if __name__ == "__main__":
    print(f"Morpheus is running on http://127.0.0.1:{PORT}")
    threading.Thread(target=warm_up, daemon=True).start()
    threading.Thread(target=sample_gpu, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
