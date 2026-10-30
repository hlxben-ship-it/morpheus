"""Morpheus web UI — http://127.0.0.1:5196

Serves the page in web/, generated images from output/, and a small JSON API around
morpheus.run() and morpheus.run_edit(). Starts ComfyUI on demand if it is not already running.
"""
import base64
import json
import mimetypes
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

import morpheus

PORT = 5196
WEB = morpheus.ROOT / "web"
RESOLUTIONS = [
    {"label": "Quadrat", "width": 1024, "height": 1024},
    {"label": "Quer 3:2", "width": 1216, "height": 832},
    {"label": "Breit 16:9", "width": 1344, "height": 768},
    {"label": "Hoch 2:3", "width": 832, "height": 1216},
    {"label": "Hoch 9:16", "width": 768, "height": 1344},
    {"label": "Klein", "width": 768, "height": 768},
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
        raise RuntimeError("ComfyUI ist nicht gestartet, siehe comfyui.log.")


# Claude runs from its own empty folder: inside C:\dev it would pick up the projects' CLAUDE.md files,
# and --resume only finds a session from the folder it was started in.
# still named after Imageer: sessions started there before the rename only resume from the same folder
CHAT_DIR = Path(tempfile.gettempdir()) / "imageer-chat"

CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "action": {"type": "string", "enum": ["generate", "none"]},
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
    return f"""Du bist der Bild-Assistent in Morpheus, einer lokalen Bildgenerierung auf der Grafikkarte des Nutzers.
Der Nutzer beschreibt auf Deutsch, was er sehen will.

Entscheide pro Nachricht:
- action "generate": ein neues Bild. prompt ist ein ausführlicher englischer Bildprompt: Motiv, Umgebung, Licht, Bildstil, Bildaufbau.
  Bei Folgewünschen ("heller", "mit Hut") schreibst du den vorigen Prompt entsprechend angepasst neu.
- action "none": nur antworten oder nachfragen, kein Bild. prompt, model und style dann leer lassen.

Modelle (nur diese sind installiert):
{models}
Standard für neue Bilder ist z-image-turbo (rund 20 s, fotorealistisch und gut für Malstile). flux2-klein-4b ist
gleich schnell und eine Alternative mit anderem Look. noobai-xl nur für Anime und Illustration (dann den Prompt als
kommagetrennte Danbooru-Tags schreiben, z. B. "1girl, elf, silver hair, forest, night"). qwen-image-2512 dauert
2 bis 4 Minuten: nimm es nur, wenn Schrift im Bild vorkommen soll oder der Nutzer ausdrücklich höchste Qualität
will, und sag dann, dass es länger dauert. realvis-xl und chroma nur, wenn der Nutzer sie nennt.
Wenn der Nutzer ein bestimmtes Modell nennt, nimm genau das.

Stile:
{styles}
"pixel" und "gameboy" nur, wenn der Nutzer Retro- oder Pixel-Look will; die Pixel-Nachbearbeitung passiert danach automatisch.
Sonst "none" und den gewünschten Stil in den Prompt schreiben.

Formate: {sizes}. Nimm 1024x1024, wenn nichts dagegen spricht.

reply: ein bis zwei Sätze auf Deutsch, was du machst oder was du wissen musst. Kein Markdown."""


def ask_claude(message: str, session: str | None, ghost: bool = False) -> tuple[dict, str]:
    """One chat turn through the Claude Code CLI; returns the structured answer and the session id.

    Ghost turns are not saved by the CLI and cannot be resumed; the caller passes the conversation in the message.
    """
    CHAT_DIR.mkdir(exist_ok=True)
    cmd = [shutil.which("claude") or "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(CHAT_SCHEMA),
           "--tools", "",
           "--model", "sonnet", "--system-prompt", chat_system_prompt()]
    if ghost:
        cmd.append("--no-session-persistence")
    elif session:
        cmd += ["--resume", session]
    p = subprocess.run(cmd, input=message, capture_output=True, text=True, encoding="utf-8", cwd=CHAT_DIR, timeout=180,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Claude hat nicht geantwortet: {(p.stderr or p.stdout).strip()[:300]}")
    if out.get("is_error") or not out.get("structured_output"):
        raise RuntimeError(f"Claude-Fehler: {str(out.get('result'))[:300]}")
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


def ghost_image(ref: str) -> Image.Image:
    with ghost_lock:
        data = GHOST.get(ref.removeprefix("ghost/"))
    if data is None:
        raise ValueError("Das Geisterbild ist nicht mehr da.")
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
                raise RuntimeError(msg or "Der Face Swap ist abgebrochen: " + err.strip().splitlines()[-1][:300])
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
        if route in ("/logo.png", "/favicon.png"):
            return self.send_file(WEB / route[1:], WEB)
        if route == "/api/options":
            models = [{"key": k, "label": m["label"], "note": m["note"], "installed": morpheus.installed(k)}
                      for k, m in morpheus.MODELS.items()]
            edit_models = [{"key": k, "label": m["label"], "note": m["note"], "installed": morpheus.installed(k)}
                           for k, m in morpheus.EDIT_MODELS.items()]
            styles = [{"key": k, "label": s["label"]} for k, s in morpheus.STYLES.items()]
            return self.send_json({"models": models, "edit_models": edit_models, "styles": styles, "resolutions": RESOLUTIONS})
        if route == "/api/status":
            return self.send_json({"comfy": morpheus.comfy_up(), "busy": gpu.locked()})
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
                    "/api/ghost/clear": self.ghost_clear, "/api/faceswap": self.faceswap}
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
            raise ValueError("Bitte etwas schreiben.")
        ghost = bool(req.get("ghost"))
        if ghost and req.get("context"):
            # No saved session to resume, so the page sends the ghost conversation so far.
            lines = [f"{'Nutzer' if m.get('role') == 'user' else 'Du'}: {m.get('text', '')}" for m in req["context"][-20:]]
            message = "Bisheriges Gespräch:\n" + "\n".join(lines) + "\n\nNeue Nachricht:\n" + message
        last = req.get("last")
        if isinstance(last, dict) and last.get("prompt"):
            # the picture the user sees last, possibly made outside the chat (Manuell, Bearbeiten, Nochmal)
            message = (f"[Letztes Bild in dieser Session: Modell {last.get('model')}, {last.get('width')}x{last.get('height')}, "
                       f"Prompt: {last['prompt']}]\n\n{message}")
        answer, session = ask_claude(message, req.get("session") or None, ghost)

        result = {"session": None if ghost else session, "reply": answer["reply"], "action": answer["action"], "image": None}
        if answer["action"] == "generate":
            model = answer["model"] if answer["model"] in morpheus.MODELS and morpheus.installed(answer["model"]) else "z-image-turbo"
            style = answer["style"] if answer["style"] in morpheus.STYLES else "none"
            size = next((r for r in RESOLUTIONS if r["width"] == answer["width"] and r["height"] == answer["height"]), RESOLUTIONS[0])
            ensure_comfy()
            with gpu:
                meta = morpheus.run(answer["prompt"], style, model, size["width"], size["height"], ghost=ghost,
                                    progress=progress_for(req))
            result["image"] = ghost_keep(meta) if ghost else meta
        return result

    def generate(self, req):
        """The manual form: exact prompt and settings, no Claude in between."""
        prompt = str(req.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Bitte einen Prompt eingeben.")
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
            raise ValueError("Bitte beschreiben, was sich ändern soll.")
        model = req.get("model", "flux2-klein-4b-edit")
        if model not in morpheus.EDIT_MODELS:
            raise ValueError(f"Unbekanntes Modell '{model}'.")
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
                raise ValueError("Bitte ein Ausgangsbild wählen.")
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
            return self.send_json({"error": "Leerer Upload."}, 400)
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
            raise ValueError("Face Swap ist nicht installiert (.venv-faceswap fehlt, siehe README).")
        video = UPLOADS / str(req.get("video_id", "")).strip("./\\")
        if not req.get("photo") or not req.get("video_id") or not video.is_file():
            raise ValueError("Bitte ein Foto und ein Video wählen.")
        photo = base64.b64decode(str(req["photo"]).split(",", 1)[-1])
        job_id = f"{time.time_ns():x}"
        quality = req.get("quality") if req.get("quality") in ("fast", "premium") else "high"
        span = None  # only a section of the video, in seconds
        if req.get("start") is not None and req.get("end") is not None:
            start, end = float(req["start"]), float(req["end"])
            if end - start < 0.1:
                raise ValueError("Der gewählte Abschnitt ist zu kurz.")
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


def warm_up() -> None:
    """Start ComfyUI and render one tiny throwaway image with the default model, so the first real prompt
    doesn't pay for reading ~20 GB of weights from disk (123 s cold vs ~20 s warm)."""
    try:
        ensure_comfy()
        with gpu:
            morpheus.generate(morpheus.MODELS["z-image-turbo"]["workflow"]("warm up", 256, 256, 0, 1), ghost=True)
        print("Morpheus: ComfyUI ist warm.")
    except Exception as e:  # the first real render just pays the cold start then
        print(f"Morpheus: Aufwärmen fehlgeschlagen ({e}).")


if __name__ == "__main__":
    print(f"Morpheus läuft auf http://127.0.0.1:{PORT}")
    threading.Thread(target=warm_up, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
