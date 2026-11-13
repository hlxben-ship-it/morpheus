"""Agent chat: a local pi agent (pi-coding-agent in RPC mode) primed as Morpheus' prompt writer, talking to OpenRouter.

    agent_chat/SYSTEM.md     the priming (system prompt)
    agent_chat/skills/*/     Agent Skills (SKILL.md each); pi lists them, the agent reads one when it needs it
    agent_chat/config.json   provider, model, greeting

One pi process per chat session, spoken to over stdin/stdout JSON lines. The OpenRouter key comes from the
environment or a gitignored .env next to this file and only ever goes into the pi process's environment.
`fake` sessions never start pi and never cost money (tests, layout work).
"""
import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

import morpheus

HOME = morpheus.ROOT / "agent_chat"
CONFIG = json.loads((HOME / "config.json").read_text(encoding="utf-8"))
SESSIONS: dict[str, "Session"] = {}
LOCK = threading.Lock()
LONG_SIDE = 1024  # images are shrunk to this before they go to the model; enough to see, cheap in tokens
TIMEOUT = 180
MAX_IMAGES = 6  # per message


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    env = morpheus.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise ValueError("no OPENROUTER_API_KEY — set it in the environment or in C:\\dev\\morpheus\\.env")
    return key


def pi_command() -> list[str]:
    """node + pi's cli.js; the npm .cmd shim would put cmd.exe and its quoting between us and the arguments."""
    npm = Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules"
    for pkg in ("@earendil-works/pi-coding-agent", "@mariozechner/pi-coding-agent"):
        cli = npm / pkg / "dist" / "cli.js"
        if cli.exists():
            return [shutil.which("node") or "node", str(cli)]
    raise ValueError("pi is not installed (npm i -g @earendil-works/pi-coding-agent)")


def image_part(data_url: str) -> dict:
    """A data URL from the page → pi's ImageContent, shrunk to LONG_SIDE and sent as JPEG."""
    m = re.match(r"data:image/[\w.+-]+;base64,(.+)", data_url or "", re.S)
    if not m:
        raise ValueError("the image must be a data URL")
    try:
        img = Image.open(BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
    except Exception:
        raise ValueError("that file is not a picture I can read")
    img.thumbnail((LONG_SIDE, LONG_SIDE))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=88)
    return {"type": "image", "data": base64.b64encode(buf.getvalue()).decode("ascii"), "mimeType": "image/jpeg"}


class Session:
    def __init__(self, fake: bool, model: str):
        self.fake = fake
        self.model = model
        self.used = time.time()
        self.pictures = 0  # how many the user has sent; the next one is img<pictures + 1>
        self.lock = threading.Lock()  # one prompt at a time per session
        self.proc = None
        if fake:
            return
        # its own pi home: the user's ~/.pi/agent/auth.json would otherwise beat this key with whatever is saved there
        home = morpheus.ROOT / ".pi-agent"
        home.mkdir(exist_ok=True)
        write_models(home)
        env ={**os.environ, "OPENROUTER_API_KEY": api_key(), "PI_OFFLINE": "1", "PI_CODING_AGENT_DIR": str(home)}
        args = pi_command() + [
            "--mode", "rpc", "--no-session", "--offline", "--no-approve",
            "--provider", CONFIG["provider"], "--model", model, "--thinking", CONFIG.get("thinking", "off"),
            "--system-prompt", str(HOME / "SYSTEM.md"),
            "--tools", "read",  # only to read its own skills
            "--no-skills", "--skill", str(HOME / "skills"),
            "--no-extensions", "--no-prompt-templates", "--no-themes", "--no-context-files",
        ]
        self.proc = subprocess.Popen(args, cwd=HOME, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.err: list[str] = []
        threading.Thread(target=lambda: [self.err.append(l) for l in self.proc.stderr], daemon=True).start()

    def switch(self, model: str):
        """Change the model mid-chat; pi keeps the conversation."""
        with self.lock:
            if not self.fake:
                if self.proc.poll() is not None:
                    raise ValueError("the agent stopped: " + "".join(self.err[-5:]).strip())
                cmd_id = uuid.uuid4().hex
                self.proc.stdin.write(json.dumps({"id": cmd_id, "type": "set_model", "provider": CONFIG["provider"], "modelId": model}) + "\n")
                self.proc.stdin.flush()
                for line in self.proc.stdout:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "response" and ev.get("id") == cmd_id:
                        if not ev.get("success"):
                            raise ValueError(ev.get("error") or "pi refused the model")
                        break
            self.model = model

    def ask(self, message: str, images: list[dict]) -> str:
        with self.lock:
            self.used = time.time()
            if self.fake:
                time.sleep(0.4)
                seen = " and the picture" if images else ""
                return (f"(fake) Here is a 5 s LTX 2.3 prompt from your words{seen}.\n\n```prompt\n"
                        f"The subject slowly turns toward the camera while a light wind moves the hair. {message.strip()[:200]}"
                        "\n```")
            if self.proc.poll() is not None:
                raise ValueError("the agent stopped: " + "".join(self.err[-5:]).strip())
            cmd = {"id": uuid.uuid4().hex, "type": "prompt", "message": message}
            if images:
                cmd["images"] = images
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
            return self.read_reply(cmd["id"])

    def read_reply(self, cmd_id: str) -> str:
        """Read events until the agent ends; the reply is the last assistant message's text."""
        out = {"text": None, "error": None, "usage": {"in": 0, "out": 0, "cost": 0.0}}

        def pump():
            for line in self.proc.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "response" and ev.get("id") == cmd_id and not ev.get("success", True):
                    out["error"] = ev.get("error") or "the agent refused the prompt"
                    return
                if ev.get("type") == "agent_end":
                    msgs = [m for m in ev.get("messages", []) if m.get("role") == "assistant"]
                    last = msgs[-1] if msgs else {}
                    if last.get("stopReason") == "error" or last.get("errorMessage"):
                        out["error"] = last.get("errorMessage") or "the model returned an error"
                    for m in msgs:  # every model call of the run counts, e.g. the one that read a skill
                        u = m.get("usage") or {}
                        out["usage"]["in"] += u.get("input", 0) + u.get("cacheRead", 0) + u.get("cacheWrite", 0)
                        out["usage"]["out"] += u.get("output", 0)
                        out["usage"]["cost"] += (u.get("cost") or {}).get("total", 0)  # pi's estimate from its price table
                    out["text"] = "".join(c.get("text", "") for c in last.get("content", []) if c.get("type") == "text")
                    return
            out["error"] = "the agent stopped: " + "".join(self.err[-5:]).strip()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        t.join(TIMEOUT)
        if t.is_alive():
            self.close()
            raise ValueError(f"no answer within {TIMEOUT} s")
        if out["error"]:
            raise ValueError(out["error"])
        self.usage = out["usage"]
        return out["text"] or ""

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()


def reap():
    limit = CONFIG.get("idle_minutes", 20) * 60
    with LOCK:
        for sid, s in list(SESSIONS.items()):
            if time.time() - s.used > limit:
                s.close()
                del SESSIONS[sid]


def models() -> dict:
    return {m["id"]: m for m in CONFIG["models"]}


def pick(model) -> str:
    model = str(model or CONFIG["model"])
    if model not in models():
        raise ValueError(f"unknown model {model}")
    return model


def write_models(home: Path):
    """Models pi does not know yet (config "pi" field) go into the agent's own models.json, merged into openrouter."""
    extra = [{"id": m["id"], "name": m["label"], "input": ["text", "image"] if m["vision"] else ["text"],
              **{k: v for k, v in m["pi"].items() if k != "cost"},
              "cost": {"cacheRead": 0, "cacheWrite": 0, **m["pi"].get("cost", {})},
              "compat": {"supportsDeveloperRole": False}}
             for m in CONFIG["models"] if m.get("pi")]
    data = {"providers": {"openrouter": {"baseUrl": "https://openrouter.ai/api/v1", "api": "openai-completions",
                                         "apiKey": "$OPENROUTER_API_KEY", "models": extra}}}
    (home / "models.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def open_session(req: dict) -> dict:
    reap()
    sid = uuid.uuid4().hex
    s = Session(bool(req.get("fake")), pick(req.get("model")))
    with LOCK:
        SESSIONS[sid] = s
    return {"session": sid, "greeting": CONFIG["greeting"], "model": s.model, "fake": s.fake,
            "models": [{k: m[k] for k in ("id", "label", "vision", "price")} for m in CONFIG["models"]],
            "skills": sorted(p.parent.name for p in (HOME / "skills").glob("*/SKILL.md"))}


def set_model(req: dict) -> dict:
    s = SESSIONS.get(str(req.get("session", "")))
    if not s:
        raise ValueError("this chat has ended — start a new one")
    s.switch(pick(req.get("model")))
    return {"model": s.model}


def send(req: dict) -> dict:
    s = SESSIONS.get(str(req.get("session", "")))
    if not s:
        raise ValueError("this chat has ended — start a new one")
    message = str(req.get("message", "")).strip()
    if req.get("task"):  # what the host page wants, e.g. "LTX 2.3 video prompt for the Playground"
        message = f"[Context from the page: {str(req['task'])[:300]}]\n\n{message}"
    urls = list(req.get("images") or ([req["image"]] if req.get("image") else []))[:MAX_IMAGES]
    if urls and not models()[s.model]["vision"]:
        raise ValueError(f"{models()[s.model]['label']} can't see pictures — switch to a model with vision, or send text only")
    images = [image_part(u) for u in urls]
    if not message and not images:
        raise ValueError("say something")
    if images:  # img1, img2 … numbered across the chat; the page counts the same way for its labels
        names = [f"img{s.pictures + i + 1}" for i in range(len(images))]
        s.pictures += len(images)
        message = f"[Pictures in this message, in order: {', '.join(names)}]\n\n{message}"
    s.usage = {"in": 1200, "out": 180, "cost": 0.0021} if s.fake else None  # fake: plausible numbers for the layout
    reply = s.ask(message or "Look at this.", images)
    return {"reply": reply, "usage": s.usage}


def close(req: dict) -> dict:
    with LOCK:
        s = SESSIONS.pop(str(req.get("session", "")), None)
    if s:
        s.close()
    return {"ok": True}


ACTIONS = {"open": open_session, "send": send, "close": close, "model": set_model}
