"""Yumeko: turn an agent into a trained character LoRA, then use it, alone or in manga panels.

Every job is one self-contained folder, output/yumeko/<Agent>_<jobid>/, holding job.json, its pictures, its dataset and
its LoRA stages. Export or import is copying that folder. Manga panels live the same way in output/yumeko/_manga/<id>/.

Phase 1 base image with Claude's help, 2 a complementary base by hand, 3 a training set of nine at a time, 4 training,
5 using the LoRA. Morpheus balances Phase 3's tags itself: on this card local models were slow and ignored the counts.
"""
import json
import os
import random
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import agents
import csv
import morpheus

ROOT = morpheus.OUTPUT / "yumeko"
MANGA = ROOT / "_manga"
MODEL = "janku-v5"  # the one checkpoint proven end to end: training, LoRA, masked inpainting
CATALOG = json.loads((morpheus.ROOT / "yumeko_category_map.json").read_text(encoding="utf-8"))
TRAIN_PY = morpheus.ROOT / ".venv-train" / "Scripts" / "python.exe"
TRAINER = morpheus.ROOT / "tools" / "yumeko_train.py"
CHECKPOINT = morpheus.COMFY_MODELS / "checkpoints" / "JANKUTrainedChenkinNoobai_v50.safetensors"
MIN_SET = 30
STEPS = 2000
BIG_SET = 40  # a set above this has more to learn, so it gets more steps
BIG_STEPS = 2500
SIZE = (832, 1216)
NO_MARKS = "watermark, copyright, signature, artist name, text, "
SAFE = "nsfw, nude, loli, child, " + NO_MARKS  # added in front of the model's own negative on everything Yumeko renders
NO_MINORS = "loli, child, "  # stays on in Use, whatever the fields hold
NO_MINORS_OPEN = "underage, young, "  # added too whenever nudity isn't blocked
NO_NUDITY = "nsfw, nude, nipples, topless, "  # Use's "Block nudity" toggle, on by default
ADULT = "adult woman, mature female, "
STYLES = {
    "modern": {"label": "Modern anime", "tags": "modern anime style, clean lineart, cel shading, vibrant colors, "},
    "nineties": {"label": "90s anime", "tags": "1990s anime style, retro anime, cel animation, muted colors, film grain, "},
    "2.5d": {"label": "2.5D", "tags": "2.5d, semi-realistic anime, soft shading, detailed skin, volumetric lighting, "},
    "semireal": {"label": "Semi-real", "tags": "realistic, semi-realistic, cinematic lighting, depth of field, detailed face, "},
    "gothic": {"label": "Gothic", "tags": "gothic, dark atmosphere, dramatic shadows, ornate details, muted colors, "},
    "photoreal": {"label": "Photorealistic", "tags": "photorealistic, photo, realistic skin texture, 85mm lens, natural lighting, "},
    "3d": {"label": "3D render", "tags": "3d render, 3d, blender, subsurface scattering, soft global illumination, "},
    "novel": {"label": "Light novel", "tags": "light novel illustration, official art, detailed illustration, soft colors, "},
    "cel": {"label": "Cel shading", "tags": "cel shading, flat color, bold outlines, anime screencap, "},
    "lineart": {"label": "Clean lineart", "tags": "clean lineart, crisp lines, flat shading, minimal shading, white highlights, "},
    "cyberpunk": {"label": "Cyberpunk", "tags": "cyberpunk, neon lights, neon glow, futuristic, high contrast, "},
    "softfantasy": {"label": "Soft fantasy", "tags": "soft fantasy anime, pastel colors, thin lineart, gentle lighting, painterly background, muted colors, "},
}
DATA = threading.Lock()  # job.json while the queue thread and the page both write it
FILLING = set()  # jobs whose training-set queue is being topped up
QUEUE = 9  # pictures waiting for a verdict
KEEP_BACK = 5  # decided pictures that stay reachable with Back one
GPU = threading.Lock()  # one render or training at a time, and Yumeko's renders run one after another
ensure_comfy = lambda: None  # server.py hands in its own
ask_claude = None
CHAT_DIR = None  # server.py hands in its own
VARIED = ["shot", "view", "body_position", "expression", "setting", "lighting"]  # the facets Phase 3 balances
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "render": {"type": "boolean"},
        "variants": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "character": {"type": "string"}, "outfit": {"type": "string"}, "scene": {"type": "string"}},
            "required": ["label", "character", "outfit", "scene"]}},
    },
    "required": ["reply", "render", "variants"],
}
CHAT_SYSTEM = """You are Yumeko, lazy, playful and a little smug, helping design an anime character's base image. She is an
adult woman. Keep replies to two sentences. Your job is to fine tune her prompts from the user's reactions.
Each round is exactly three variants, each rendered as one picture, and you keep the full prompt of every variant:
- label: two or three words naming what sets this variant apart.
- character: Danbooru tags, comma separated, for what never changes on her: hair, eyes, face, marks, jewellery, glasses. No quality,
  count or age tags such as masterpiece, 1girl, solo or adult; those are added for you.
- outfit: Danbooru tags for her full signature outfit, head to toe, specific and with colours.
- scene: Danbooru tags for one simple pose, framing, expression and a plain or simple background.
Real Danbooru tags only, never invented phrases ("small chrome plates on right cheek"): the model doesn't know them. Use the
nearest real tags (cyborg, mechanical parts, facial mark) and say in your reply what the model can't draw exactly.
Always one adult character, fully clothed, always SFW; never describe a child.
Vary only what the user asks about and keep everything else word for word across the three. "Try different hair colors" means three
variants that differ only in hair colour. With no such request, offer three clearly different takes on her.
You get last round's variants. When the user picks pictures, look at them, say briefly what they share, and build the next three
from the picked variants. Set render true and give three variants when the request calls for new pictures. Set render false and give
no variants when you only answer."""
SECOND_SYSTEM = CHAT_SYSTEM + """

This is the second base image: the same woman in the same outfit as the first base, seen another way, so the two complement each other.
You get the first base's character and outfit tags. Copy them word for word into every variant unless the user asks to change them.
Open the first base picture with Read and check the new views against it. Vary the scene: side view, from behind, sitting, close-up and so on.
Show what the first base doesn't already show. Each variant is a different view unless the user asks otherwise."""
SECOND_OPENING = "Give me three different views of her that complement the first base. Render."

_entries = {e["tag"]: e for e in CATALOG["tags"]}


# ---------- jobs ----------
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name) or "Agent"


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9]+_[0-9a-f]+", job_id or ""):
        raise ValueError("Unknown job.")
    d = ROOT / job_id
    if not (d / "job.json").exists():
        raise ValueError("That job is gone.")
    return d


def load(job_id: str) -> dict:
    return json.loads((job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def save(job: dict) -> dict:
    job["updated"] = time.time()
    p = ROOT / job["id"] / "job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return job


def agent_look(agent_id: str) -> str:
    """Her look as tags. Packages keep only her persona; Soulforge's own record has the look."""
    f = morpheus.OUTPUT / "soulforge" / agent_id / "agent.json"
    return str(json.loads(f.read_text(encoding="utf-8")).get("look") or "") if f.exists() else ""


def create(req: dict) -> dict:
    name = str(req.get("name", "")).strip()
    a = {"id": None, "name": name} if name else agents.load(str(req.get("agent", "")))  # a name alone starts from scratch
    if not a or not _safe(a["name"]).strip("_"):
        raise ValueError("Pick an agent or name your character.")
    style = req.get("style") if req.get("style") in STYLES else "modern"
    jid = f"{_safe(a['name'])}_{time.time_ns():x}"
    (ROOT / jid).mkdir(parents=True)
    persona = a.get("persona", {})
    job = {"schema_version": 1, "id": jid, "created": time.time(), "phase": 1, "star": False, "busy": None, "error": None,
           "agent": {"id": a["id"], "name": a["name"], "trigger_token": f"{_safe(a['name']).lower()}_{jid[-5:]}",
                     "identity_summary": (agent_look(a["id"]) or str(persona.get("soul") or "")) if a["id"] else ""},
           "model": MODEL, "style": style, "bases": [], "phase1": {"session": None, "chat": [], "grid": [], "positive": "", "tags": []},
           "phase2": {"positive": "", "negative": "", "grid": []},
           "dataset": {"accepted": [], "batch": [], "counts": {}, "rounds": 0}, "train": {"state": "idle"}, "lora": None, "gallery": []}
    return save(job)


def jobs() -> list:
    if not ROOT.exists():
        return []
    out = []
    for d in ROOT.iterdir():
        f = d / "job.json"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            if j.get("phase") == 4:
                train_state(j)
            out.append({k: j.get(k) for k in ("id", "phase", "star", "busy", "updated", "error")} | {
                "agent": j["agent"]["name"], "cover": j["bases"][0]["file"] if j.get("bases") else None,
                "accepted": len(j["dataset"]["accepted"]), "finished": bool(j.get("lora")), "step": step_label(j)})
    return sorted(out, key=lambda j: (not j["star"], j["agent"].lower(), -j["phase"]))


def step_label(job: dict) -> str | None:
    """The autopilot step a job is at, as the page's tabs name it; None for a set picked by hand."""
    a = job.get("auto")
    if not a:
        return None
    if job["phase"] == 3:
        return "Her card" if a["state"] == "card" else "Review" if a.get("reviewed") else "Building"
    return {4: "Training", 5: "Use" if job.get("lora") else "Exam"}.get(job["phase"])


def delete(req: dict) -> dict:
    d = job_dir(req.get("id"))
    job = load(d.name)
    if str(req.get("confirm", "")).strip() != job["agent"]["name"]:
        raise ValueError("Type the agent's name to delete this job.")
    if job["train"].get("state") == "running":
        stop_training(job)
    shutil.rmtree(d)
    unregister(job)
    return {"ok": True}


def star(req: dict) -> dict:
    job = load(req.get("id"))
    job["star"] = not job["star"]
    return save(job)


def greeting() -> dict:
    js = sorted(jobs(), key=lambda j: j["updated"] or 0, reverse=True)
    if not js:
        return {"text": random.choice(["Oh. You came all this way and there isn't a single set yet. Tap Add, I'll supervise from here.",
                                       "Nothing to do? Perfect. Unless you press Add. Please don't press Add."])}
    j = js[0]
    name = j["agent"]
    return {"text": random.choice([
        f"Back for {name} again? I was promised a nap, and {name} was promised cheekbones.",
        f"{name} is still where you left her, unlike my motivation.",
        f"Let me guess: {name} needs one more tiny change. It's never tiny.",
        f"I checked on {name} while you were gone. Well, I thought about checking.",
    ])}


# ---------- rendering ----------
def _background(job: dict, label: str, work) -> dict:
    """Run work(job) on a thread while job["busy"] says what's happening; the page polls job.json."""
    if job.get("busy"):
        raise ValueError(f"Busy: {job['busy']}")
    if training_now():
        raise ValueError("A LoRA is training; the GPU is taken until it finishes.")
    job["busy"], job["error"] = label, None
    save(job)

    def run():
        current = job
        try:
            with GPU:
                ensure_comfy()
                current = work(job) or job
        except Exception as e:
            current["error"] = str(e)
        current["busy"] = None
        save(current)
    threading.Thread(target=run, daemon=True).start()
    return job


LIVE: dict[str, dict] = {}  # job id -> the Use render forming: step, total, preview count and the newest frame


def render(positive: str, negative: str | None, seed: int, size=SIZE, loras=(), prefix: str | None = None, safe: str = SAFE, progress=None) -> Image.Image:
    m = morpheus.MODELS[MODEL]
    default = m["workflow"].negative
    g = m["workflow"](positive, size[0], size[1], seed, m["steps"], negative=safe + (default if negative is None else negative), prefix=m["workflow"].prefix if prefix is None else prefix)
    return morpheus.generate(_with_loras(g, loras), progress=progress)


def _with_loras(graph: dict, loras) -> dict:
    """Chain full LoRA loaders, so a LoRA with trained text encoders keeps that part. Works for UNet-only LoRAs too."""
    model, clip = ["1", 0], ["1", 1]
    for i, (name, strength) in enumerate(loras):
        graph[f"6{i}"] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name, "strength_model": strength, "strength_clip": strength}}
        model, clip = [f"6{i}", 0], [f"6{i}", 1]
    for key, node in graph.items():
        if not (key.startswith("6") and node["class_type"] == "LoraLoader"):
            node["inputs"] = {k: clip if v == ["1", 1] else v for k, v in node["inputs"].items()}
    graph["8"]["inputs"]["model"] = model
    return graph


GRID = 3
MAX_PICKS = 3


def _picked(slot: dict, files) -> list:
    return [g for g in slot["grid"] if g["file"] in list(files or [])[:MAX_PICKS]]


def _grid(job: dict, positive: str, negative: str | None, prefix: str, slot: dict, seeds=()) -> None:
    """Fill slot["grid"] one picture at a time, saving each, so the page shows them as they land.
    Seeds of the pictures the user liked come first, so the next round stays near them."""
    pics = slot["grid"] = []
    seeds = list(seeds)
    for i in range(GRID):
        job["busy"] = f"Rendering {i + 1} of {GRID}"
        save(job)
        seed = seeds[i] if i < len(seeds) else random.randrange(2 ** 31)
        name = f"{prefix}_{time.time_ns():x}.png"
        render(positive, negative, seed).save(ROOT / job["id"] / name)
        pics.append({"file": name, "seed": seed, "positive": positive, "negative": negative or morpheus.MODELS[MODEL]["workflow"].negative})
        save(job)


def _variant_tags(v: dict) -> list:
    roles = (("character", "identity", "appearance"), ("outfit", "outfit", "clothing"), ("scene", "variable", "setting"))
    return [{"tag": t, "yumeko_category": cat, "role": role} for key, role, cat in roles for t in _split(v[key])]


def _render_variants(job: dict, variants: list, slot: dict, seeds: list) -> None:
    """One picture per variant, on one seed, so what differs between them is what the prompts changed."""
    pics = slot["grid"] = []
    seeds = seeds or [random.randrange(2 ** 31)]
    for i, v in enumerate(variants[:GRID]):
        job["busy"] = f"Rendering {i + 1} of {min(GRID, len(variants))}"
        save(job)
        seed = seeds[i % len(seeds)]
        name = f"p1_{time.time_ns():x}.png"
        positive = style_tags(job) + ADULT + "1girl, solo, " + ", ".join(v[k] for k in ("character", "outfit", "scene"))  # the prompt promises these; Akane came out two or three times per picture without
        render(positive, None, seed).save(ROOT / job["id"] / name)
        pics.append({"file": name, "seed": seed, "positive": positive, "negative": morpheus.MODELS[MODEL]["workflow"].negative, "label": v["label"],
                     "character": v["character"], "outfit": v["outfit"], "scene": v["scene"], "tags": _variant_tags(v)})
        save(job)


def previews(req: dict) -> dict:
    """Four clickable style previews of the agent, cached in the job. Each is a real render on the validated model."""
    job = load(req.get("id"))
    look = str(req.get("look") or job["agent"]["identity_summary"] or "1girl, long hair")

    def work(j):
        out = {}
        for key, s in STYLES.items():
            j["busy"] = f"Rendering {s['label']}"
            save(j)
            name = f"style_{key.replace('.', '')}.png"
            render(s["tags"] + ADULT + "1girl, solo, " + look + ", upper body, looking at viewer, simple background", None, 7).save(ROOT / j["id"] / name)
            out[key] = name
        j["style_previews"] = out
        return j
    return _background(job, "Rendering styles", work)


def start_over(req: dict) -> dict:
    """Back to the style picker. Pictures and chat go; the job, its name and its trigger stay."""
    job = load(req.get("id"))
    if job.get("busy") or job["phase"] > 3:
        raise ValueError("Only a job that isn't rendering and hasn't started training can start over.")
    FILLING.discard(job["id"])
    d = ROOT / job["id"]
    for f in d.iterdir():
        if f.suffix == ".png" or f.name.startswith("base-"):
            f.unlink(missing_ok=True)
    shutil.rmtree(d / "dataset", ignore_errors=True)
    job.update(phase=1, error=None, styles=[], style="modern", bases=[], style_previews={},
               phase1={"session": None, "chat": [], "grid": [], "positive": "", "tags": []}, phase2={"positive": "", "negative": "", "grid": []},
               dataset={"accepted": [], "batch": [], "counts": {}, "rounds": 0})
    for k in ("auto", "exam", "calib", "look"):  # the autopilot card belonged to the old bases; open_card returned it unchanged
        job.pop(k, None)
    return save(job)


def style_tags(job: dict) -> str:
    """All picked styles, without repeating a tag two styles share."""
    tags = [t for k in job.get("styles") or [job["style"]] for t in _split(STYLES[k]["tags"])]
    return ", ".join(dict.fromkeys(tags)) + ", "


def pick_style(req: dict) -> dict:
    job = load(req.get("id"))
    picks = [k for k in dict.fromkeys(req.get("styles") or []) if k in STYLES]
    if not picks:
        raise ValueError("Pick at least one style.")
    job["style"], job["styles"] = picks[0], picks
    return save(job)


def chat(req: dict) -> dict:
    job = load(req.get("id"))
    text = str(req.get("text", "")).strip()
    opening = req.get("opening") is True
    phase = job["phase"]
    if phase not in (1, 2):
        raise ValueError("Yumeko only chats during the base images.")
    if opening:
        if phase != 2 or job["phase2"].get("chat"):
            return job
        text = SECOND_OPENING
    if not text:
        raise ValueError("Say something first.")
    slot = f"phase{phase}"
    p = job[slot]
    p.setdefault("chat", [])
    p.setdefault("session", None)
    picks = _picked(p, req.get("picks"))
    if req.get("reference"):  # a picture to recreate; it stays with the job and every later turn can look at it
        import base64
        import io
        name = f"ref_{time.time_ns():x}.png"
        Image.open(io.BytesIO(base64.b64decode(str(req["reference"]).split(",", 1)[-1]))).convert("RGB").save(ROOT / job["id"] / name)
        p["reference"] = name
    if not opening:
        p["chat"].append({"role": "user", "text": text, "picked": len(picks)} | ({"reference": p["reference"]} if req.get("reference") else {}))

    def work(j):
        q = j[slot]
        j["busy"] = "Yumeko is thinking"
        save(j)
        recent = "\n".join(f"{m['role']}: {m['text']}" for m in q["chat"][-9:-1])
        context = f"[Character: {j['agent']['name']}. {j['agent']['identity_summary'] or 'New character, designed only from this chat.'}]\n[Conversation so far:\n{recent}]\n[Current tags: {q['positive']}]\n"
        last = [g for g in q["grid"] if g.get("character")]
        if last:
            context += "[Last round:\n" + "\n".join(f"{'PICKED ' if g in picks else ''}{g['label']}: character: {g['character']} | outfit: {g['outfit']} | scene: {g['scene']}" for g in last) + "]\n"
        base = j["bases"][0] if phase == 2 else None
        if base:
            kept = lambda role: ", ".join(t["tag"] for t in base["tags"] if t.get("role") == role)
            context += f"[First base. character: {kept('identity')} | outfit: {kept('outfit')} | scene: {kept('variable')}]\n"
        copies = []
        for i, g in enumerate(picks):  # she reads them from her working folder, the only place Read reaches
            copy = CHAT_DIR / f"yumeko_{j['id']}_{i}.png"
            shutil.copy(ROOT / j["id"] / g["file"], copy)
            copies.append(copy)
        if q.get("reference"):
            ref = CHAT_DIR / f"yumeko_{j['id']}_ref.png"
            shutil.copy(ROOT / j["id"] / q["reference"], ref)
            copies.append(ref)
            context += f"[The user's reference picture, the look to recreate as closely as the model allows. Open it with Read: {ref.name}]\n"
        if base:
            first = CHAT_DIR / f"yumeko_{j['id']}_base.png"
            shutil.copy(ROOT / j["id"] / base["file"], first)
            copies.append(first)
            context += f"[First base picture, open it with Read: {first.name}]\n"
        if picks:
            context += "[The user picked these pictures from the last grid. Open each with Read before answering: " + ", ".join(c.name for c in copies[:len(picks)]) + "]\n"
        try:
            answer, q["session"] = ask_claude(context + text, q["session"], ghost=True, see=bool(copies), system=SECOND_SYSTEM if base else CHAT_SYSTEM, schema=CHAT_SCHEMA)
        finally:
            for c in copies:
                c.unlink(missing_ok=True)
        q["chat"].append({"role": "yumeko", "text": answer["reply"]})
        if answer.get("render") and answer.get("variants"):
            _render_variants(j, answer["variants"], q, [g["seed"] for g in picks] or ([base["base_image"]["seed"]] if base else []))
        return j
    return _background(job, "Yumeko is thinking", work)


def _manifest(job: dict, pick: dict, anchor: str, role: str, tags: list) -> dict:
    source = ROOT / job["id"] / pick["file"]
    w, h = Image.open(source).size
    name = f"{anchor}.png"
    shutil.copy(source, ROOT / job["id"] / name)
    return {"schema_version": 1, "kind": "yumeko.base-image", "anchor_id": anchor, "anchor_role": role, "file": name, "agent": job["agent"],
            "base_image": {"file": name, "width": w, "height": h, "seed": pick["seed"]},
            "generation": {"model": job["model"], "anime_style": " + ".join(job.get("styles") or [job["style"]]), "positive_prompt": pick["positive"], "negative_prompt": pick["negative"]},
            "tags": tags}


def approve(req: dict) -> dict:
    """Phase 1 or 2: this grid picture is it. Its manifest sits beside it, and the job moves on."""
    job = load(req.get("id"))
    grid = job["phase1"]["grid"] if job["phase"] == 1 else job["phase2"]["grid"]
    pick = next((g for g in grid if g["file"] == req.get("file")), None)
    if not pick:
        raise ValueError("Pick a picture from the grid.")
    if job["phase"] == 1:
        tags = [t | {"source": "claude"} for t in pick.get("tags") or job["phase1"]["tags"]]
        job["bases"] = [_manifest(job, pick, "base-1", "initial", tags)]
        job["phase2"]["positive"], job["phase2"]["negative"] = pick["positive"], pick["negative"]
        job["phase"] = 3 if req.get("second") is not True else 2  # the second view is optional; the card comes next
    elif job["phase"] == 2:
        tags = [t | {"source": "claude" if pick.get("tags") else "human"} for t in pick.get("tags") or job["bases"][0]["tags"]]
        job["bases"].append(_manifest(job, pick, "base-2", "complementary", tags))
        job["phase"] = 3
    (ROOT / job["id"] / (job["bases"][-1]["anchor_id"] + ".json")).write_text(json.dumps(job["bases"][-1], ensure_ascii=False, indent=1), encoding="utf-8")
    return save(job)


def phase2_render(req: dict) -> dict:
    job = load(req.get("id"))
    if job["phase"] != 2:
        raise ValueError("This job isn't in Phase 2.")
    job["phase2"]["positive"] = str(req.get("positive", job["phase2"]["positive"]))
    job["phase2"]["negative"] = str(req.get("negative", job["phase2"]["negative"]))
    seeds = [g["seed"] for g in _picked(job["phase2"], req.get("picks"))]

    def work(j):
        q = j["phase2"]
        _grid(j, q["positive"], q["negative"], "p2", q, seeds)
        return j
    return _background(job, "Rendering", work)


# ---------- Danbooru tag search ----------
TAG_FILE = morpheus.ROOT / "data" / "danbooru_tags.csv"
TAG_KINDS = {0: "general", 1: "artist", 3: "series", 4: "character", 5: "meta"}
NO_SEARCH = re.compile(r"loli|shota|child|toddler|underage|preteen|teen|teenage|kindergarten|aged_down|elementary_school")
_tags = None


def _load_tags() -> list:
    global _tags
    if _tags is None:
        rows = []
        with TAG_FILE.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[2].isdigit():
                    continue
                aliases = [a for a in (row[3].split(",") if len(row) > 3 else []) if not a.startswith("/")]
                if any(NO_SEARCH.search(n) for n in [row[0]] + aliases):
                    continue
                rows.append((row[0], int(row[1]) if row[1].isdigit() else 0, int(row[2]), aliases))
        _tags = rows  # the file is sorted by post count, so the first hits are the most used
    return _tags


def search_tags(query: str, kinds=("general",), limit: int = 30) -> list:
    q = query.strip().lower().replace(" ", "_")
    if len(q) < 2:
        return []
    ranks = ([], [], [], [])
    for name, kind, count, aliases in _load_tags():
        if TAG_KINDS.get(kind, "general") not in kinds:
            continue
        rank = 0 if name == q else 1 if name.startswith(q) else 2 if q in name else 3 if any(q in a for a in aliases) else None
        if rank is not None and len(ranks[rank]) < limit:
            ranks[rank].append({"tag": name, "kind": TAG_KINDS.get(kind, "general"), "count": count})
    return (ranks[0] + ranks[1] + ranks[2] + ranks[3])[:limit]


# ---------- Phase 3: the set ----------
def conflicts(tags: list) -> list:
    """Tags sharing a single-choice facet that were picked together, like sitting + standing."""
    seen, bad = {}, []
    for t in tags:
        e = _entries.get(t)
        if e and e.get("facet") in CATALOG["single_choice"]:
            if e["facet"] in seen:
                bad.append(f"{seen[e['facet']]} + {t}")
            seen[e["facet"]] = t
    return bad


CLOTHES = re.compile(r"shirt|blouse|vest|neck ?tie|\btie\b|bow ?tie|skirt|dress|gown|robe|kimono|yukata|jacket|coat|cape|cloak|corset|sweater|hoodie|cardigan|uniform|"
                     r"pants|trousers|jeans|shorts|leggings|thighhighs|stockings|socks|pantyhose|boots|shoes|heels|sandals|gloves|armor|bikini|swimsuit|lingerie|\bbra\b|collar|sleeves|belt", re.I)
NOT_LOOK = {"1girl", "solo", "adult", "adult woman", "mature female", "anime style", "masterpiece", "best quality", "absurdres"}
# Each extra outfit is kept to shots that show it. Bikini and lingerie stay calm and few, because they pull hardest on the sexualised drift.
OUTFITS = [
    {"key": "signature", "label": "Signature outfit", "target": 24},
    {"key": "sundress", "label": "Sundress", "target": 3, "tags": "light blue sundress, sleeveless, bare shoulders"},
    {"key": "hoodie", "label": "Hoodie and jeans", "target": 3, "tags": "grey hoodie, blue jeans"},
    {"key": "tank", "label": "Tank top and shorts", "target": 3, "tags": "white tank top, denim shorts"},
    {"key": "kimono", "label": "Kimono", "target": 3, "tags": "floral kimono, obi"},
    {"key": "coat", "label": "Winter coat", "target": 3, "tags": "beige winter coat, scarf"},
    {"key": "bikini", "label": "Bikini", "target": 3, "tags": "blue bikini", "setting": ["beach", "poolside", "simple_background"], "lighting": ["daytime", "sunset", "overcast"],
     "view": ["from_side", "three_quarter_view"], "body_position": ["standing", "walking", "sitting"]},
    {"key": "lingerie", "label": "Lingerie", "target": 2, "tags": "white lace lingerie", "setting": ["bedroom_interior", "simple_background"],
     "body_position": ["standing", "sitting"], "expression": ["light_smile", "serious"], "lighting": ["daytime", "overcast"], "view": ["from_side", "three_quarter_view"]},
    {"key": "closeups", "label": "Face close-ups", "target": 4, "tags": "plain white top", "shot": ["close-up", "portrait"], "setting": ["simple_background"], "view": ["from_side", "three_quarter_view"]},
]
SHOW_OUTFIT = ["full_body", "cowboy_shot", "upper_body"]


def _split(text) -> list:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def outfit_plan(job: dict) -> dict:
    """The set's outfit groups. Made once, from the base image's identity tags, and then the user's to edit."""
    d = job["dataset"]
    if d.get("plan"):
        return d["plan"]
    tags = [t for t in identity_tags(job) if t.lower() not in NOT_LOOK]
    named = {t["tag"].replace("_", " ") for b in job["bases"] for t in b["tags"] if t.get("role") == "outfit"}
    worn = [t for t in tags if t in named] if named else [t for t in tags if CLOTHES.search(t)]  # older jobs never named it
    groups = [dict(g, on=True) for g in OUTFITS]
    for g in groups[1:-1]:  # she already owns this look, so it adds no contrast
        g["on"] = g["key"] not in ", ".join(worn).lower()
    d["plan"] = {"permanent": [t for t in tags if t not in worn], "signature": worn, "groups": groups}
    return d["plan"]


def kept(job: dict, key: str) -> int:
    d = job["dataset"]
    return sum(a.get("group", "signature") == key for a in d["accepted"]) + sum(bool(c.get("keep")) and c.get("group", "signature") == key for c in d["batch"])


def current_group(job: dict) -> dict | None:
    return next((g for g in outfit_plan(job)["groups"] if g["on"] and kept(job, g["key"]) < g["target"]), None)


def set_outfits(req: dict) -> dict:
    job = load(req.get("id"))
    if job["phase"] != 3:
        raise ValueError("This job isn't building its set.")
    p = outfit_plan(job)
    if "permanent" in req:
        p["permanent"] = _split(req["permanent"])
    if "signature" in req:
        p["signature"] = _split(req["signature"])
    for g in p["groups"]:
        e = (req.get("groups") or {}).get(g["key"])
        if e and g["key"] != "signature":
            if "on" in e:
                g["on"] = bool(e["on"])
            if "tags" in e:
                g["tags"] = str(e["tags"]).strip()
    return save(job)


def identity_tags(job: dict) -> list:
    out = []
    for b in job["bases"]:
        out += [t["tag"].replace("_", " ") for t in b["tags"] if t.get("role") in ("identity", "outfit") and t["tag"].replace("_", " ") not in out]
    return out


def next_group(job: dict) -> dict | None:
    """The first outfit that still needs pictures, counting those waiting for a verdict plus two spare for dismissals."""
    waiting = [c.get("group", "signature") for c in job["dataset"]["batch"] if c.get("keep") is None]
    return next((g for g in outfit_plan(job)["groups"] if g["on"] and kept(job, g["key"]) < g["target"]
                 and kept(job, g["key"]) + waiting.count(g["key"]) < g["target"] + 2), None)


def plan(job: dict, group: dict | None = None) -> list:
    """Pictures for one outfit: in each facet, the tag used least so far, counting those still in the queue. Ties go by chance."""
    counts = dict(job["dataset"]["counts"])
    for c in job["dataset"]["batch"]:
        if c.get("keep") is not False:
            for t in c.get("planned", []):
                counts[t] = counts.get(t, 0) + 1
    tags_of = {f: [e["tag"] for e in CATALOG["tags"] if e.get("facet") == f] for f in VARIED}
    style = style_tags(job)
    p, g = outfit_plan(job), group or current_group(job)
    if not g:
        raise ValueError("Every outfit is done. Start training.")
    signature = g["key"] == "signature"
    worn = p["signature"] if signature else _split(g["tags"])
    who = ", ".join(p["permanent"] + worn) or job["agent"]["identity_summary"]
    wf = morpheus.MODELS[MODEL]["workflow"]
    negative = None if signature or not p["signature"] else ", ".join(p["signature"]) + ", " + wf.negative  # stops her usual clothes leaking in
    size = 9 if signature else min(9, g["target"] - kept(job, g["key"]) + 2)
    out = []
    for i in range(size):
        picked = []
        for facet in VARIED:
            if facet == "body_position" and picked[0] in CATALOG["no_pose_shots"]:
                continue
            options = g.get(facet) or (SHOW_OUTFIT if facet == "shot" and not signature else tags_of[facet])
            least = min(counts.get(t, 0) for t in options)
            t = random.choice([o for o in options if counts.get(o, 0) == least])
            counts[t] = counts.get(t, 0) + 1
            picked.append(t)
        if conflicts(picked):
            raise ValueError(f"Conflicting tags planned: {conflicts(picked)}")
        out.append({"positive": style + ADULT + "1girl, solo, " + who + ", " + ", ".join(picked), "negative": negative, "tags": picked + worn, "planned": picked, "group": g["key"]})
    return out


def batch(req: dict) -> dict:
    """Top up the queue. Nothing to press: the page calls this on opening and every verdict does too."""
    job = load(req.get("id"))
    if job["phase"] != 3:
        raise ValueError("This job isn't building its set.")
    _fill(job["id"])
    return job


def _fill(job_id: str) -> None:
    """Keep QUEUE pictures waiting, oldest first. Takes the GPU one picture at a time so other renders can get in between."""
    if job_id in FILLING or training_now():
        return
    FILLING.add(job_id)

    def run():
        try:
            while job_id in FILLING:
                with DATA:
                    j = load(job_id)
                    if j["phase"] != 3 or sum(c.get("keep") is None for c in j["dataset"]["batch"]) >= QUEUE:
                        break
                    _commit(j, KEEP_BACK)
                    j["error"] = None
                    g = next_group(j)
                    save(j)
                if not g:
                    break
                p = plan(j, g)[0]
                seed = random.randrange(2 ** 31)
                name = f"cand_{time.time_ns():x}.png"
                with GPU:
                    ensure_comfy()
                    picture = render(p["positive"], p["negative"], seed)
                with DATA:
                    if job_id not in FILLING:
                        break
                    picture.save(ROOT / job_id / name)
                    j = load(job_id)
                    j["dataset"]["batch"].append({"file": name, "seed": seed, "positive": p["positive"], "negative": p["negative"], "planned": p["planned"], "tags": list(p["tags"]), "group": p["group"], "keep": None})
                    save(j)
        except Exception as e:
            with DATA:
                j = load(job_id)
                j["error"] = str(e)
                save(j)
        finally:
            FILLING.discard(job_id)
    threading.Thread(target=run, daemon=True).start()


def _commit(job: dict, keep_back: int = 0) -> None:
    """Move verdicts into the set. With keep_back, undecided pictures and the latest verdicts stay; without, nothing does."""
    d = job["dataset"]
    decided = [c for c in d["batch"] if c.get("keep") is not None]
    done = (decided[:-keep_back] if keep_back else d["batch"])
    for c in done:
        if c.get("keep"):
            d["accepted"].append({"file": c["file"], "tags": c["tags"], "planned": c["planned"], "group": c.get("group", "signature")})
            for t in c["tags"]:
                d["counts"][t] = d["counts"].get(t, 0) + 1
        else:
            (ROOT / job["id"] / c["file"]).unlink(missing_ok=True)
    d["batch"] = [c for c in d["batch"] if not any(c is x for x in done)]


def decide(req: dict) -> dict:
    job = load(req.get("id"))
    c = next((c for c in job["dataset"]["batch"] if c["file"] == req.get("file")), None)
    if not c:
        return job  # it moved into the set while the verdict was on its way
    if "keep" in req:
        c["keep"] = bool(req["keep"])
    if isinstance(req.get("tags"), list):
        c["tags"] = [str(t).strip() for t in req["tags"] if str(t).strip()]
    save(job)
    _fill(job["id"])
    return job


def remove_accepted(req: dict) -> dict:
    job = load(req.get("id"))
    d = job["dataset"]
    a = next((a for a in d["accepted"] if a["file"] == req.get("file")), None)
    if a:
        d["accepted"].remove(a)
        for t in a["tags"]:
            d["counts"][t] = max(0, d["counts"].get(t, 0) - 1)
        (ROOT / job["id"] / a["file"]).unlink(missing_ok=True)
    return save(job)


# ---------- Phase 4: training ----------
def training_now() -> bool:
    return any(p.get("state") == "running" for p in _train_states())


def _train_states():
    if ROOT.exists():
        for d in ROOT.iterdir():
            f = d / "job.json"
            if f.exists():
                yield json.loads(f.read_text(encoding="utf-8")).get("train", {})


def start_training(req: dict) -> dict:
    job = load(req.get("id"))
    if job["phase"] != 3:
        raise ValueError("This job isn't ready to train.")
    if len(job["dataset"]["accepted"]) + sum(bool(c.get("keep")) for c in job["dataset"]["batch"]) < MIN_SET:
        raise ValueError(f"Needs {MIN_SET} accepted pictures.")
    FILLING.discard(job["id"])  # undecided pictures go, kept ones stay
    with DATA:
        job = load(job["id"])
        _commit(job)
        save(job)
    if len(job["dataset"]["accepted"]) < MIN_SET:
        raise ValueError(f"Needs {MIN_SET} accepted pictures, you have {len(job['dataset']['accepted'])}.")
    if training_now():
        raise ValueError("Another LoRA is training.")
    if not GPU.acquire(timeout=180):  # the queue may be finishing its last picture
        raise ValueError("The GPU is rendering; try again when it's done.")
    try:
        d = ROOT / job["id"]
        trigger = job["agent"]["trigger_token"]
        data = d / "dataset" / f"5_{trigger}"
        shutil.rmtree(d / "dataset", ignore_errors=True)
        data.mkdir(parents=True)
        worn = outfit_plan(job)["signature"]
        for i, a in enumerate(job["dataset"]["accepted"]):
            shutil.copy(d / a["file"], data / f"{i:03}.png")
            tags = a["tags"] + (worn if "group" not in a else [])  # picked before outfits existed
            caption = a.get("caption") or ", ".join([trigger] + [t.replace("_", " ") for t in tags])  # autopilot sets write their own
            (data / f"{i:03}.txt").write_text(caption, encoding="utf-8")
        auto = bool(job.get("auto"))
        steps = (3000 if len(job["dataset"]["accepted"]) > 60 else 2500) if auto else BIG_STEPS if len(job["dataset"]["accepted"]) > BIG_SET else STEPS
        try:
            urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/free", b'{"unload_models":true,"free_memory":true}',
                                                           {"Content-Type": "application/json"}), timeout=10)
        except OSError:
            pass  # ComfyUI isn't running, so there is nothing on the GPU to free
        cmd = [str(TRAIN_PY), str(TRAINER), "--pretrained_model_name_or_path", str(CHECKPOINT), "--train_data_dir", str(d / "dataset"),
               "--output_dir", str(d / "lora"), "--output_name", "lora", "--max_train_steps", str(steps), "--save_every_n_steps", "500"] + (["--text-encoders"] if auto else [])
        log = open(d / "train.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=morpheus.ROOT / "tools" / "sd-scripts",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        job["phase"], job["train"] = 4, {"state": "running", "pid": p.pid, "started": time.time(), "step": 0, "total": steps, "stages": []}
        return save(job)
    finally:
        GPU.release()


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    return str(pid) in (out or "")


def train_state(job: dict) -> dict:
    """Read progress from the worker's log. The worker outlives a server restart, so nothing here holds it."""
    t = job["train"]
    if t.get("state") != "running":
        return t
    d = ROOT / job["id"]
    log = d / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace")[-4000:] if log.exists() else ""
    steps = re.findall(r"(\d+)/" + str(t["total"]) + r" \[", text)
    if steps:
        t["step"] = int(steps[-1])
    t["stages"] = sorted(f.name for f in (d / "lora").glob("lora-step*.safetensors")) if (d / "lora").exists() else []
    if not _alive(t["pid"]):
        last = d / "lora" / f"lora-step{t['total']:08}.safetensors"
        t["state"] = "done" if last.exists() else "failed"
        t["log_tail"] = "" if last.exists() else text[-600:]
        if last.exists():
            job["phase"] = 5
        save(job)
    return t


def stop_training(job: dict) -> dict:
    t = job["train"]
    if t.get("state") == "running":
        subprocess.run(["taskkill", "/PID", str(t["pid"]), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        t["state"] = "stopped"
        job["phase"] = 3
        job["dataset"]["accepted"] = job["dataset"]["accepted"]
    return save(job)


def stages_of_saved(job: dict) -> list:
    """The stages actually on disk."""
    return sorted(int(p.stem[-8:]) for p in (ROOT / job["id"] / "lora").glob("lora-step*.safetensors"))


def stages_of(job: dict) -> list:
    """The last three stages saved on disk. A full run gives total-1000, total-500 and total; a stopped run its newest three."""
    saved = sorted(int(p.stem[-8:]) for p in (ROOT / job["id"] / "lora").glob("lora-step*.safetensors"))
    if saved:
        return saved[-3:]
    total = job["train"].get("total", STEPS)
    return [total - 1000, total - 500, total]


def _stage_file(job: dict, stage: int) -> str:
    """Copy one stage to ComfyUI's loras folder and return its name."""
    name = f"yumeko_{job['id'][-10:]}_{stage}.safetensors"
    dst = morpheus.COMFY_MODELS / "loras" / name
    if not dst.exists():
        shutil.copy(ROOT / job["id"] / "lora" / f"lora-step{stage:08}.safetensors", dst)
    return name


def compare_stages(req: dict) -> dict:
    job = load(req.get("id"))
    if job["phase"] != 5:
        raise ValueError("Training isn't finished.")

    def work(j):
        out = {}
        for s in stages_of(j):
            j["busy"] = f"Testing stage {s}"
            save(j)
            name = f"stage_{s}.png"
            look = j.get("look") or {}
            render(j["agent"]["trigger_token"] + ", " + style_tags(j) + (look["positive"] + ", " if look.get("positive") else "") + ADULT + "1girl, solo, upper body, looking at viewer, smile, garden, daytime", look.get("negative"), 501,
                   loras=[(_stage_file(j, s), j["lora"]["strength"] if j.get("lora") else 1.0)]).save(ROOT / j["id"] / name)
            out[str(s)] = name
        j["stage_tests"] = out
        return j
    return _background(job, "Testing stages", work)


def choose_stage(req: dict) -> dict:
    job = load(req.get("id"))
    stage = int(req.get("stage", 0))
    if stage not in stages_of(job):
        raise ValueError("Unknown stage.")
    job["lora"] = {"file": _stage_file(job, stage), "stage": stage, "strength": float(req.get("strength", 1.0)), "trigger": job["agent"]["trigger_token"]}
    save(job)
    register(job)
    return job


# ---------- Phase 5 and Dreamforge ----------
def register(job: dict) -> None:
    """A finished LoRA shows up in Dreamforge under JANKU, trigger added by the workflow."""
    if job.get("lora"):
        morpheus.LORAS[f"yumeko-{job['id']}"] = {"label": job["agent"]["name"], "note": f"Yumeko LoRA, stage {job['lora']['stage']}",
                                                 "file": job["lora"]["file"], "model": MODEL, "strength": job["lora"]["strength"], "trigger": job["lora"]["trigger"]}


def unregister(job: dict) -> None:
    morpheus.LORAS.pop(f"yumeko-{job['id']}", None)
    if job.get("lora"):
        (morpheus.COMFY_MODELS / "loras" / job["lora"]["file"]).unlink(missing_ok=True)


def register_all() -> None:
    if ROOT.exists():
        for d in ROOT.iterdir():
            if (d / "job.json").exists():
                register(json.loads((d / "job.json").read_text(encoding="utf-8")))


def _without(text: str, tags: str) -> str:
    drop = {t.strip() for t in tags.split(",") if t.strip()}
    return ", ".join(t.strip() for t in text.split(",") if t.strip() and t.strip() not in drop)


def trained_words(job: dict) -> list:
    """The words her LoRA learned besides her name, offered as toggles in Use: the outfit trigger, plus any kept in her card."""
    words = [t.strip() for t in str((job.get("look") or {}).get("positive") or "").split(",") if t.strip()]
    a = job.get("auto")
    if a and (job.get("lora") or {}).get("recipe") == "v2":
        words.append(a.get("outfit_trigger") or job["agent"]["trigger_token"] + "_outfit")
    return list(dict.fromkeys(words))


def use_defaults(job: dict) -> dict:
    wf = morpheus.MODELS[MODEL]["workflow"]
    look = job.get("look") or {}  # a character whose design needs spelling out at render time
    # trained: words the LoRA learned besides her trigger (her outfit trigger), kept apart so they can be cleared for other clothes
    # identity: the few tags the calibration found the trigger alone misses; Use adds them silently while the Identity pill is on
    ident = ", ".join(x for x in [look.get("identity"), "not: " + look["identity_negative"] if look.get("identity_negative") else ""] if x)
    return {"quality": ADULT + "1girl, solo, " + wf.prefix + style_tags(job), "trained": look.get("positive") or "", "trained_options": trained_words(job),
            "pill_tags": {"identity": [look.get("identity") or "", look.get("identity_negative") or ""], "outfit": [look.get("outfit") or "", look.get("outfit_negative") or ""]},
            "identity": ident, "outfit_tags": ", ".join(x for x in [look.get("outfit"), "not: " + look["outfit_negative"] if look.get("outfit_negative") else ""] if x),
            "negative": NO_MARKS + _without(wf.negative, NO_NUDITY) + (", " + look["negative"] if look.get("negative") else ""), "always_positive": job["agent"]["trigger_token"], "always_negative": NO_MINORS.rstrip(", "), "nudity_block": NO_NUDITY.rstrip(", ")}


def use(req: dict) -> dict:
    job = load(req.get("id"))
    if not job.get("lora"):
        raise ValueError("Pick a stage first.")
    text = str(req.get("prompt", "")).strip()
    if not text:
        raise ValueError("Describe the picture.")
    d = use_defaults(job)
    quality = str(req["quality"]) if "quality" in req else d["quality"]
    trained = str(req["trained"]).strip().strip(",").strip() if "trained" in req else d["trained"]
    negative = str(req["negative"]) if req.get("negative") is not None else d["negative"]
    block = req.get("block_nudity") is not False  # on unless explicitly switched off
    ident = ident_on = req.get("identity") is not False  # the Identity pill, on unless switched off
    locked = NO_MINORS + (NO_NUDITY if block else NO_MINORS_OPEN)  # never in the editable fields, always sent
    strength = float(req.get("strength", job["lora"]["strength"]))
    size = (int(req.get("width", SIZE[0])), int(req.get("height", SIZE[1])))
    seed = int(req["seed"]) if str(req.get("seed", "")).strip() else random.randrange(2 ** 31)
    job["lora"]["strength"] = strength

    def work(j):
        name = f"use_{time.time_ns():x}.png"
        look = j.get("look") or {}
        # the page now writes the pill tags visibly into quality/negative (explicit), so nothing is added silently then
        outfit_on = bool(look.get("positive")) and str(look["positive"]) in trained.split(", ") and not req.get("explicit")
        ident = ident_on and not req.get("explicit")
        silent = [look.get("identity") if ident else "", look.get("outfit") if outfit_on else ""]  # calibrated: the trigger alone drifts on these
        silent_neg = [look.get("identity_negative") if ident else "", look.get("outfit_negative") if outfit_on else ""]
        positive = ", ".join(x.strip(", ") for x in [d["always_positive"], trained, *silent] if x and x.strip(", ")) + ", " + ("" if block else ADULT) + quality + text  # adult forced in when unblocked
        sent_negative = ", ".join(x.strip(", ") for x in [negative, *silent_neg] if x and x.strip(", "))
        live = LIVE[j["id"]] = {"step": 0, "total": 0, "n": 0, "width": size[0], "height": size[1]}

        def progress(step=None, total=None, preview=None):
            if step is not None:
                live["step"], live["total"] = step, total
            if preview is not None:
                live["frame"], live["n"] = preview, live["n"] + 1
        try:
            render(positive, sent_negative, seed, size, [(j["lora"]["file"], strength)], prefix="", safe=locked, progress=progress).save(ROOT / j["id"] / name)
        finally:
            LIVE.pop(j["id"], None)
        j["gallery"].insert(0, {"file": name, "prompt": text, "quality": quality, "negative": negative, "seed": seed, "strength": strength,
                                "width": size[0], "height": size[1], "trained": trained, "identity": ident, "block_nudity": block, "sent_positive": positive, "sent_negative": locked + sent_negative})
        return j
    return _background(job, "Rendering", work)


# ---------- Phase 6: manga panels ----------
def panel_dir(pid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]+", pid or ""):
        raise ValueError("Unknown panel.")
    return MANGA / pid


def panel_load(pid: str) -> dict:
    return json.loads((panel_dir(pid) / "panel.json").read_text(encoding="utf-8"))


def panel_save(p: dict) -> dict:
    p["updated"] = time.time()
    (panel_dir(p["id"]) / "panel.json").write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def panels() -> list:
    if not MANGA.exists():
        return []
    return sorted((json.loads(f.read_text(encoding="utf-8")) for f in MANGA.glob("*/panel.json")), key=lambda p: -p["updated"])


def finished() -> list:
    out = []
    if ROOT.exists():
        for d in ROOT.iterdir():
            if (d / "job.json").exists():
                j = json.loads((d / "job.json").read_text(encoding="utf-8"))
                if j.get("lora"):
                    # the composition has no LoRA: the full card describes her best; prose summaries are no tags
                    look = (j.get("auto") or {}).get("permanent") or (j.get("look") or {}).get("identity") or j["agent"].get("identity_summary") or ""
                    look = "" if ". " in look else look
                    lk = j.get("look") or {}
                    look = ", ".join(t.strip() for t in look.split(",") if t.strip() and t.strip() not in ("1girl", "solo"))
                    b0 = (j.get("bases") or [{}])[0].get("file")
                    out.append({"base": f"{d.name}/{b0}" if b0 else None, "id": j["id"], "name": j["agent"]["name"], "trigger": j["lora"]["trigger"], "file": j["lora"]["file"], "strength": j["lora"]["strength"], "look": look,
                                "pills": {"identity": ["", lk.get("identity_negative") or ""],  # her look is already in the description
                                          "outfit": [", ".join(x for x in [lk.get("positive"), lk.get("outfit")] if x), lk.get("outfit_negative") or ""]}})
    return out


# ---------- mangas: ordered panels, read as pages of four ----------
BOOKS = MANGA / "books.json"


def books() -> list:
    """Every manga with its panel order. Panels from before mangas existed land in "Loose panels" once."""
    data = json.loads(BOOKS.read_text(encoding="utf-8")) if BOOKS.exists() else []
    placed = {pid for b in data for pid in b["panels"]}
    loose = [p["id"] for p in sorted(panels(), key=lambda p: p.get("updated", 0)) if p["id"] not in placed]
    if loose:
        data.append({"id": f"{time.time_ns():x}", "title": "Loose panels", "panels": loose})
        _books_save(data)
    return data


def _books_save(data: list) -> None:
    MANGA.mkdir(parents=True, exist_ok=True)
    BOOKS.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def book_new(req: dict) -> dict:
    data = books()
    b = {"id": f"{time.time_ns():x}", "title": str(req.get("title") or "New manga").strip() or "New manga", "panels": []}
    data.append(b)
    _books_save(data)
    return b


def book_edit(req: dict) -> dict:
    """Rename, reorder (panels: the full new order), or move one panel here (add: panel id, taken out of any other manga)."""
    data = books()
    b = next((x for x in data if x["id"] == req.get("id")), None)
    if not b:
        raise ValueError("Unknown manga.")
    if "title" in req:
        b["title"] = str(req["title"]).strip() or b["title"]
    if isinstance(req.get("panels"), list) and sorted(req["panels"]) == sorted(b["panels"]):
        b["panels"] = req["panels"]
    if req.get("add"):
        for x in data:
            if req["add"] in x["panels"]:
                x["panels"].remove(req["add"])
        b["panels"].append(req["add"])
    _books_save(data)
    return b


def book_delete(req: dict) -> dict:
    """Delete a manga; its panels move to Loose panels, nothing is lost."""
    data = [b for b in books() if b["id"] != req.get("id")]
    _books_save(data)
    books()
    return {"ok": True}


PANEL_NEGATIVE = "nsfw, nude, nipples, see-through, lingerie, cleavage, text, speech bubble, watermark, signature, extra person"  # loli, child are locked anyway
PANEL_PROMPT = "manga panel. On the left: (her hair, eyes, outfit). On the right: (her hair, eyes, outfit). Scene: (place, time of day, light)."


def panel_new(req: dict) -> dict:
    """A new panel continues the manga: scene, characters and format come from the panel before it. The first one gets defaults."""
    pid = f"{time.time_ns():x}"
    prev = None
    if req.get("book"):
        data = books()
        b = next((x for x in data if x["id"] == req["book"]), None)
        if b:
            for x in reversed(b["panels"]):
                try:
                    prev = panel_load(x)
                    break
                except (OSError, ValueError):
                    continue
            b["panels"].append(pid)
            _books_save(data)
    panel_dir(pid).mkdir(parents=True)
    base = {"tags": "2girls, indoors, wide shot, ", "prompt": PANEL_PROMPT, "negative": PANEL_NEGATIVE, "width": 1216, "height": 832, "characters": []}
    if prev:  # masks don't carry over: the characters stand somewhere else in a new composition
        base = {k: prev[k] for k in base} | {"characters": [{k: v for k, v in c.items() if k != "masked"} for c in prev["characters"]]}
    return panel_save({"id": pid, "title": str(req.get("title") or "New panel"), **base, "versions": [], "composition": None, "busy": None, "error": None})


def panel_edit(req: dict) -> dict:
    p = panel_load(req.get("id"))
    for k in ("title", "tags", "prompt", "negative", "characters", "width", "height", "locked", "pass_tags", "pass_negative"):
        if k == "characters" and k in req:  # a reference is set by panel_ref only; a save from the page keeps it
            old = {x["id"]: x.get("ref") for x in p.get("characters", [])}
            for x in req[k]:
                if old.get(x.get("id")):
                    x["ref"] = old[x["id"]]
                else:
                    x.pop("ref", None)
        if k in req:
            p[k] = req[k]
    if req.get("current") in [v["file"] for v in p["versions"]]:
        p["current"] = req["current"]  # the version shown in the manga and painted on next; the list itself stays in order
    return panel_save(p)


PANEL_LIVE: dict[str, dict] = {}  # panel id -> sampler step and total of the render running on it
_PANEL_NOW = {"id": None}  # one render at a time (GPU lock), so one panel


def _panel_progress(step=None, total=None, preview=None):
    if _PANEL_NOW["id"] and step is not None:
        PANEL_LIVE[_PANEL_NOW["id"]] = {"step": step, "total": total}


def _panel_background(p: dict, label: str, work) -> dict:
    if p.get("busy"):
        raise ValueError(f"Busy: {p['busy']}")
    if training_now():
        raise ValueError("A LoRA is training; the GPU is taken until it finishes.")
    p["busy"], p["error"] = label, None
    panel_save(p)

    def run():
        try:
            with GPU:
                ensure_comfy()
                _PANEL_NOW["id"] = p["id"]
                work(p)
        except Exception as e:
            p["error"] = str(e)
        _PANEL_NOW["id"] = None
        PANEL_LIVE.pop(p["id"], None)
        p["busy"] = None
        panel_save(p)
    threading.Thread(target=run, daemon=True).start()
    return p


def panel_compose(req: dict) -> dict:
    """The composition: no LoRA, the scene and each character's look in words. together=true uses the LoRAs at once instead."""
    p = panel_load(req.get("id"))
    loras = [(c["file"], float(c.get("strength", 1.0)) * 0.7) for c in p["characters"]] if req.get("together") else []

    def work(q):
        triggers = ", ".join(c["trigger"] for c in q["characters"]) + ", " if req.get("together") and q["characters"] else ""
        positive = triggers + q["tags"].replace("_", " ") + ", " + q["prompt"]
        size = (int(q["width"]), int(q["height"]))
        name = f"comp_{time.time_ns():x}.png"
        seed = random.randrange(2 ** 31)
        render(positive, q["negative"] or None, seed, size, loras, progress=_panel_progress).save(panel_dir(q["id"]) / name)
        q["composition"] = name
        q["versions"] = [{"file": name, "label": "Composition", "seed": seed}] + q["versions"]
        q["current"] = name
    return _panel_background(p, "Rendering composition", work)


def _inpaint(source: Image.Image, mask: Image.Image, positive: str, negative: str | None, lora: str | None, strength: float, denoise: float, seed: int,
             wash: bool = False, context: float = 0.0) -> Image.Image:
    """One character's pass: crop around her mask, redo the crop large with her LoRA, paste back through a soft mask."""
    W, H = source.size
    mask = mask.convert("L").resize((W, H)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
    box = mask.getbbox()
    if not box:
        raise ValueError("That character's mask is empty; paint her first.")
    pad = max(48, int(context * max(box[2] - box[0], box[3] - box[1])))  # a fix sees more of her around the spot, so it continues her body
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad))
    cw, ch = box[2] - box[0], box[3] - box[1]
    scale = 1216 / max(cw, ch)
    rw, rh = int(cw * scale) // 8 * 8, int(ch * scale) // 8 * 8
    folder = morpheus.COMFY_INPUT / "yumeko"
    folder.mkdir(parents=True, exist_ok=True)
    crop, mcrop = source.crop(box).resize((rw, rh), Image.LANCZOS), mask.crop(box).resize((rw, rh), Image.LANCZOS)
    if wash:  # the old colours and shapes under the mask fade to a soft grey, so the tags decide them, not the picture
        # colours go, shapes stay: grey, flattened toward mid-grey, so folds and the body pose remain and the tags pick colours
        # (only the colours: the brightness stays, so a light wall doesn't turn into a grey patch; soft mask edges are left alone)
        washed = crop.convert("L").convert("RGB")
        crop = Image.composite(washed, crop, mcrop.point(lambda v: 255 if v > 200 else 0).filter(ImageFilter.GaussianBlur(3)))
    crop.save(folder / "crop.png")
    mcrop.convert("RGB").save(folder / "mask.png")
    m = morpheus.MODELS[MODEL]
    g = m["workflow"](positive, rw, rh, seed, m["steps"], negative=_dedupe(PASS_LOCKED + (negative + ", " if negative else "") + _no_nsfw(m["workflow"].negative)), prefix=m["workflow"].prefix)
    g["21"] = {"class_type": "LoadImage", "inputs": {"image": "yumeko/crop.png"}}
    g["22"] = {"class_type": "LoadImageMask", "inputs": {"image": "yumeko/mask.png", "channel": "red"}}
    g["23"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["1", 2]}}
    g["24"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["23", 0], "mask": ["22", 0]}}
    g["8"]["inputs"]["latent_image"], g["8"]["inputs"]["denoise"] = ["24", 0], denoise
    result = morpheus.generate(_with_loras(g, [(lora, strength)] if lora else []), progress=_panel_progress).resize((cw, ch), Image.LANCZOS)
    out = source.copy()
    out.paste(result, box[:2], mask.crop(box))
    return out


HEADCOUNT = re.compile(r"^(\d\+?|multiple|solo)[ _]?(girls?|boys?|others?)$|^solo$|^(group|crowd)$")


def pass_fixed() -> dict:
    """What every character pass sends on top of the panel's and her own tags, for the page to show."""
    w = morpheus.MODELS[MODEL]["workflow"]
    return {"positive": w.prefix + "(her trigger), " + ADULT + "1girl", "negative": _dedupe(PASS_LOCKED + _no_nsfw(w.negative))}


# Character passes: only loli and child are locked; nsfw and nude sit in the editable pass negative instead.
PASS_LOCKED = "loli, child, " + NO_MARKS


def _no_nsfw(tags: str) -> str:
    return ", ".join(t.strip() for t in tags.split(",") if t.strip() and t.strip().lower() not in ("nsfw", "nude"))


def _tidy(tags: str) -> str:
    """A typed prompt made safe: a stray "(" or ")" would weight everything after it, so unmatched ones are dropped; empty tags go."""
    out, open_ = [], []
    for ch in tags:
        if ch == "(":
            open_.append(len(out))
        elif ch == ")":
            if not open_:
                continue
            open_.pop()
        out.append(ch)
    for i in reversed(open_):
        del out[i]
    return ", ".join(t.strip() for t in "".join(out).split(",") if t.strip())


def _dedupe(tags: str) -> str:
    seen = []
    for t in (x.strip() for x in tags.split(",")):
        if t and t.lower() not in [s.lower() for s in seen]:
            seen.append(t)
    return ", ".join(seen)


def pass_tags(tags: str) -> str:
    """The scene tags a character pass gets: without head counts, since each pass paints one girl."""
    return ", ".join(t.strip() for t in (tags or "").split(",") if t.strip() and not HEADCOUNT.match(t.strip().lower()))


def panel_mask(req: dict) -> dict:
    """Save one character's painted mask, sent by the page as a PNG data URL."""
    import base64
    import io
    p = panel_load(req.get("id"))
    c = next((c for c in p["characters"] if c["id"] == req.get("character")), None)
    if not c:
        raise ValueError("Add that character first.")
    raw = base64.b64decode(str(req["mask"]).split(",", 1)[-1])
    Image.open(io.BytesIO(raw)).convert("L").save(panel_dir(p["id"]) / f"mask_{c['id']}.png")
    c["masked"] = True
    return panel_save(p)


def panel_pass(req: dict) -> dict:
    """Run the character's pass on the chosen version; all=true runs every masked character in order, on the composition."""
    p = panel_load(req.get("id"))
    start = str(req.get("from") or p.get("current") or (p["versions"][0]["file"] if p["versions"] else ""))
    if not start:
        raise ValueError("Render a composition first.")
    chosen = [c for c in p["characters"] if c.get("masked")] if req.get("all") else [c for c in p["characters"] if c["id"] == req.get("character")]
    if not chosen:
        raise ValueError("Paint a mask first.")

    def work(q):
        d = panel_dir(q["id"])
        base = start
        im = Image.open(d / base).convert("RGB")
        for c in chosen:
            q["busy"] = f"Painting {c['name']}"
            panel_save(q)
            seed = random.randrange(2 ** 31)
            # a locked composition sends its own pass tags and negative; unlocked, the scene tags without the head count
            # (a "3girls" in a pass that paints one girl inside her mask invites extra people)
            scene = q.get("pass_tags") if q.get("locked") else pass_tags(q["tags"])
            positive = _tidy(c["trigger"] + ", " + ADULT + "1girl, " + (c.get("action") or "") + ", " + (scene or "").replace("_", " "))
            base_neg = q.get("pass_negative") if q.get("locked") else q["negative"]
            neg = ", ".join(x.strip(", ") for x in [base_neg or "", c.get("negative") or ""] if x and x.strip(", ")) or None
            im = _inpaint(im, Image.open(d / f"mask_{c['id']}.png"), positive, neg, c["file"], float(c.get("strength", 1.0)), float(c.get("denoise", 0.62)), seed)
            name = f"pass_{time.time_ns():x}.png"
            im.save(d / name)
            q["versions"].insert(0, {"file": name, "label": c["name"], "seed": seed, "denoise": c.get("denoise", 0.62), "from": base})
            q["current"] = base = name  # the next character paints on this one
    return _panel_background(p, "Painting", work)


FIX_PRESETS = {"small": (0.5, False), "outfit": (0.85, True), "pose": (0.9, True), "qwen": (1.0, False), "sharp": (0.35, False)}
SHARP_TAGS = "detailed face, detailed eyes, detailed hair, detailed clothes, detailed body, sharp focus"  # default Change, wash
QWEN_LORAS = ("qwen-lightning", "qwen-anything2real", "qwen-multiple-angles", "qwen-sfw")


def _data_image(url: str) -> Image.Image:
    import base64
    import io
    try:
        return Image.open(io.BytesIO(base64.b64decode(str(url or "").split(",", 1)[-1])))
    except Exception:
        raise ValueError("That is no picture.")


def _base_of(c: dict) -> Path | None:
    """Her first base image, the picture her set was built from."""
    for d in ROOT.iterdir() if ROOT.exists() else []:
        f = d / "job.json"
        if f.exists():
            j = json.loads(f.read_text(encoding="utf-8"))
            if j.get("id") == c.get("lora") or (j.get("lora") or {}).get("file") == c.get("file"):
                b = (j.get("bases") or [{}])[0].get("file")
                return d / b if b and (d / b).exists() else None
    return None


def panel_ref(req: dict) -> dict:
    """A character's own Qwen reference for this panel (an outfit, an object); empty clears it back to her base image."""
    p = panel_load(req.get("id"))
    c = next((x for x in p["characters"] if x["id"] == req.get("character")), None)
    if not c:
        raise ValueError("Add that character first.")
    f = panel_dir(p["id"]) / f"ref_{c['id']}.png"
    if req.get("image"):
        im = _data_image(req["image"]).convert("RGB")
        im.thumbnail((1024, 1024))
        im.save(f)
        c["ref"] = f.name
    else:
        f.unlink(missing_ok=True)
        c.pop("ref", None)
    return panel_save(p)


def _qwen_spot(source: Image.Image, mask: Image.Image, sentence: str, ref: Path | None, loras: list, seed: int) -> Image.Image:
    """Qwen-Image-Edit on the whole panel, so she keeps her pose, size and place; the sentence names where she stands
    (from her mask), and only her mask is pasted back, so whatever Qwen did elsewhere is thrown away."""
    W, H = source.size
    if mask is not None:
        mask = mask.convert("L").resize((W, H)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
        box = mask.getbbox()
        x = (box[0] + box[2]) / 2 / W
        where = "on the left" if x < 0.38 else "on the right" if x > 0.62 else "in the middle"
        sentence = f"Edit only the woman {where} of image 1; leave the other people unchanged. " + sentence
    s = (1024 * 1024 / (W * H)) ** 0.5
    rw, rh = int(W * s) // 16 * 16, int(H * s) // 16 * 16
    folder = morpheus.COMFY_INPUT / "yumeko"
    folder.mkdir(parents=True, exist_ok=True)
    source.save(folder / "qcrop.png")
    g = morpheus.qwen_edit_workflow(sentence, "yumeko/qcrop.png", rw, rh, seed, 4, loras=loras)
    if ref:
        Image.open(ref).convert("RGB").save(folder / "qref.png")
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": "yumeko/qref.png"}}
        g["31"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["30", 0], "upscale_method": "lanczos", "megapixels": 0.5, "resolution_steps": 16}}
        for k in ("5", "6"):
            g[k]["inputs"]["image2"] = ["31", 0]
    result = morpheus.generate(g, progress=_panel_progress).convert("RGB").resize((W, H), Image.LANCZOS)
    if mask is None:
        return result
    out = source.copy()
    out.paste(result, (0, 0), mask)
    return out


def panel_fix(req: dict) -> dict:
    """Repaint one spot of the shown version: a painted mask, a few words, optionally a character (her LoRA and look).
    small: light, keeps the shapes (a smile); outfit: her outfit tags, old colours washed out; pose: strong, washed."""
    import base64
    import io
    p = panel_load(req.get("id"))
    preset = req.get("preset") if req.get("preset") in FIX_PRESETS else "small"
    denoise = max(0.2, min(1.0, float(req.get("denoise") or FIX_PRESETS[preset][0])))
    text = str(req.get("text") or "").strip(" ,")
    c = next((x for x in p["characters"] if x["id"] == req.get("character")), None)
    l = next((x for x in finished() if x["id"] == c.get("lora") or x["file"] == c.get("file")), None) if c else None
    if preset == "qwen":
        return _panel_fix_qwen(p, req, c)
    if preset == "outfit" and not (l and l["pills"]["outfit"][0]):
        raise ValueError("Her outfit needs a character with a signature outfit.")
    if preset == "sharp":  # resharpen: fixed detail tags, her words optional, the spot repainted larger and lightly
        text = ", ".join(x for x in [SHARP_TAGS, text] if x)
    if not text and preset != "outfit":
        raise ValueError("Write what should be there.")
    start = str(p.get("current") or (p["versions"][0]["file"] if p["versions"] else ""))
    if not start:
        raise ValueError("Render a composition first.")
    try:
        mask = Image.open(io.BytesIO(base64.b64decode(str(req.get("mask") or "").split(",", 1)[-1]))).convert("L")
    except Exception:
        raise ValueError("Paint over the spot first.")
    if not mask.getbbox():
        raise ValueError("Paint over the spot first.")
    # her card as it is set (description with pills, her negative) does the work; the fix words go first.
    # The outfit preset only adds her outfit tags when the card doesn't already carry them (the pill is off).
    card = (c.get("action") or "") if c else ""
    outfit = l["pills"]["outfit"][0] if preset == "outfit" and c and not c.get("outfit") else ""
    pos = ([c["trigger"], ADULT + "1girl"] if c else []) + [outfit, text, card]
    positive = _tidy(", ".join(x.strip(" ,") for x in pos if x and x.strip(" ,")))
    neg = [p.get("pass_negative") if p.get("locked") else p.get("negative"), c and c.get("negative"), l["pills"]["outfit"][1] if outfit else ""]
    negative = ", ".join(x.strip(" ,") for x in neg if x and x.strip(" ,")) or None
    label = (c["name"] if c else "Background") + " · " + {"small": "small change", "outfit": "outfit", "pose": "pose", "sharp": "resharpen"}[preset]

    def work(q):
        d = panel_dir(q["id"])
        seed = random.randrange(2 ** 31)
        q["busy"] = "Fixing " + label
        panel_save(q)
        im = _inpaint(Image.open(d / start).convert("RGB"), mask, positive, negative, c["file"] if c else None,
                      float(c.get("strength", 1.0)) if c else 1.0, denoise, seed, wash=FIX_PRESETS[preset][1], context=0.5)
        name = f"fix_{time.time_ns():x}.png"
        im.save(d / name)
        q["versions"].insert(0, {"file": name, "label": label, "kind": "fix", "text": text, "seed": seed, "denoise": denoise, "from": start})
        q["current"] = name
    return _panel_background(p, "Fixing", work)


def panel_qwen(req: dict) -> dict:
    """The panel's own Qwen edit: a sentence, LoRAs, an optional reference; with a painted mask only that area is pasted
    back, without one the whole picture is Qwen's."""
    p = panel_load(req.get("id"))
    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("Write in a sentence what should change.")
    start = str(p.get("current") or (p["versions"][0]["file"] if p["versions"] else ""))
    if not start:
        raise ValueError("Render a composition first.")
    mask = _data_image(req["mask"]).convert("L") if req.get("mask") else None
    if mask is not None and not mask.getbbox():
        mask = None
    loras = [k for k in (req.get("loras") or []) if k in QWEN_LORAS]
    ref = panel_dir(p["id"]) / p["qwen_ref"] if p.get("qwen_ref") else None
    sentence = (text + (" Use image 2 as the reference." if ref else "")
                + " All people are adult women."
                + " Keep everything else in image 1 exactly as it is, in the same art style.")

    def work(q):
        d = panel_dir(q["id"])
        seed = random.randrange(2 ** 31)
        q["busy"] = "Qwen edit" + (" (masked)" if mask is not None else " (whole picture)")
        panel_save(q)
        im = _qwen_spot(Image.open(d / start).convert("RGB"), mask, sentence, ref, loras, seed)
        name = f"qwen_{time.time_ns():x}.png"
        im.save(d / name)
        q["versions"].insert(0, {"file": name, "label": "Qwen edit" + (" · masked" if mask is not None else ""), "kind": "fix",
                                 "text": text, "seed": seed, "loras": loras, "from": start})
        q["current"] = name
    return _panel_background(p, "Qwen edit", work)


def panel_qwen_ref(req: dict) -> dict:
    """The panel Qwen edit's reference picture; empty clears it."""
    p = panel_load(req.get("id"))
    f = panel_dir(p["id"]) / "ref_qwen.png"
    if req.get("image"):
        im = _data_image(req["image"]).convert("RGB")
        im.thumbnail((1024, 1024))
        im.save(f)
        p["qwen_ref"] = f.name
    else:
        f.unlink(missing_ok=True)
        p.pop("qwen_ref", None)
    return panel_save(p)


def _panel_fix_qwen(p: dict, req: dict, c: dict | None) -> dict:
    text = str(req.get("text") or "").strip()
    if not text:
        raise ValueError("Write in a sentence what should change.")
    start = str(p.get("current") or (p["versions"][0]["file"] if p["versions"] else ""))
    if not start:
        raise ValueError("Render a composition first.")
    mask = _data_image(req.get("mask")).convert("L")
    if not mask.getbbox():
        raise ValueError("Paint over the spot first.")
    loras = [k for k in (req.get("loras") or []) if k in QWEN_LORAS]
    ref = (panel_dir(p["id"]) / c["ref"]) if c and c.get("ref") else (_base_of(c) if c else None)
    sentence = (text + (" Use image 2 only as the reference for the look, not for her pose." if ref else "")
                + " She is an adult woman: keep her pose, position, body size and adult proportions in image 1 exactly."
                + " Keep everything else in image 1 exactly as it is, in the same art style.")
    label = (c["name"] if c else "Background") + " · Qwen edit"

    def work(q):
        d = panel_dir(q["id"])
        seed = random.randrange(2 ** 31)
        q["busy"] = "Qwen edit: " + (c["name"] if c else "background")
        panel_save(q)
        im = _qwen_spot(Image.open(d / start).convert("RGB"), mask, sentence, ref, loras, seed)
        name = f"qwen_{time.time_ns():x}.png"
        im.save(d / name)
        q["versions"].insert(0, {"file": name, "label": label, "kind": "fix", "text": text, "seed": seed, "loras": loras, "from": start})
        q["current"] = name
    return _panel_background(p, "Qwen edit", work)


def panel_version_delete(req: dict) -> dict:
    """Drop one version from the list. Its file moves to the panel's trash folder, so a slip can be undone by hand."""
    p = panel_load(req.get("id"))
    if p.get("busy"):
        raise ValueError("A render is running on this panel; delete after it finishes.")
    v = next((x for x in p["versions"] if x["file"] == req.get("file")), None)
    if not v:  # only names from the list, never a path from the request
        raise ValueError("No such version.")
    d = panel_dir(p["id"]); (d / "trash").mkdir(exist_ok=True)
    if (d / v["file"]).exists():
        shutil.move(str(d / v["file"]), str(d / "trash" / v["file"]))
    p["versions"] = [x for x in p["versions"] if x is not v]
    if p.get("current") == v["file"]:
        p["current"] = None  # falls back to the newest
    if p.get("composition") == v["file"]:
        p["composition"] = None
    return panel_save(p)


def panel_delete(req: dict) -> dict:
    shutil.rmtree(panel_dir(req.get("id")), ignore_errors=True)
    if BOOKS.exists():
        _books_save([dict(b, panels=[x for x in b["panels"] if x != req.get("id")]) for b in json.loads(BOOKS.read_text(encoding="utf-8"))])
    return {"ok": True}


def state() -> dict:
    roster = []
    base = morpheus.ROOT / "agents"
    if base.exists():
        for d in sorted(base.iterdir()):
            a = agents.load(d.name)
            if a:
                roster.append({"id": a["id"], "name": a["name"], "title": a.get("title", ""), "look": agent_look(a["id"])})
    return {"jobs": jobs(), "agents": roster, "styles": {k: s["label"] for k, s in STYLES.items()}, "greeting": greeting()["text"],
            "min_set": MIN_SET, "training": training_now(), "trainer": TRAIN_PY.exists()}


def use_remove(req: dict) -> dict:
    """Delete one rendered Use picture: its gallery entry and the file."""
    job = load(req.get("id"))
    name = str(req.get("file", ""))
    entry = next((g for g in job["gallery"] if g["file"] == name), None)
    if not entry:
        raise ValueError("That picture is already gone.")
    job["gallery"].remove(entry)
    (ROOT / job["id"] / Path(name).name).unlink(missing_ok=True)
    save(job)
    return job


def _locked(action):
    def run(req: dict) -> dict:
        with DATA:
            return action(req)
    return run


ACTIONS = {"start_over": start_over, "outfits": _locked(set_outfits), "create": create, "delete": delete, "star": star, "previews": previews, "style": pick_style, "chat": chat, "approve": approve,
           "phase2": phase2_render, "batch": batch, "decide": _locked(decide), "unaccept": _locked(remove_accepted), "train": start_training,
           "stage_test": compare_stages, "stage": choose_stage, "use": use, "use_remove": _locked(use_remove),
           "stop": lambda r: stop_training(load(r.get("id"))),
           "panel_new": panel_new, "panel_edit": panel_edit, "panel_compose": panel_compose, "panel_mask": panel_mask,
           "panel_pass": panel_pass, "panel_delete": panel_delete, "panel_version_delete": panel_version_delete, "panel_fix": panel_fix, "panel_ref": panel_ref, "panel_qwen": panel_qwen, "panel_qwen_ref": panel_qwen_ref, "book_new": book_new, "book_edit": book_edit, "book_delete": book_delete}


def job_view(job_id: str) -> dict:
    job = load(job_id)
    if job["train"].get("state") == "running":
        train_state(job)
    if job["phase"] == 3 and not job["dataset"].get("plan"):
        outfit_plan(job)
        save(job)
    if job_id in LIVE:
        job["live"] = {k: v for k, v in LIVE[job_id].items() if k != "frame"}
    if job.get("auto"):
        import yumeko_auto
        yumeko_auto.view(job)
    wf = morpheus.MODELS[MODEL]["workflow"]
    return job | {"defaults": use_defaults(job), "sent": {"prefix": wf.prefix, "safe": SAFE, "sampler": "Euler a, normal", "steps": morpheus.MODELS[MODEL]["steps"], "cfg": 5.0, "checkpoint": CHECKPOINT.name}, "balance": {f: {e["tag"]: job["dataset"]["counts"].get(e["tag"], 0) for e in CATALOG["tags"] if e.get("facet") == f} for f in VARIED}}
