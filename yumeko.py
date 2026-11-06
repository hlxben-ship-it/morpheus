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
SAFE = "nsfw, nude, loli, child, "  # added in front of the model's own negative on everything Yumeko renders
NO_MINORS = "loli, child, "  # stays on in Use, whatever the fields hold
ADULT = "adult woman, mature female, "
STYLES = {
    "modern": {"label": "Modern anime", "tags": "modern anime style, clean lineart, cel shading, vibrant colors, "},
    "nineties": {"label": "90s anime", "tags": "1990s anime style, retro anime, cel animation, muted colors, film grain, "},
    "2.5d": {"label": "2.5D", "tags": "2.5d, semi-realistic anime, soft shading, detailed skin, volumetric lighting, "},
    "semireal": {"label": "Semi-real", "tags": "realistic, semi-realistic, cinematic lighting, depth of field, detailed face, "},
}
GPU = threading.Lock()  # one render or training at a time, and Yumeko's renders run one after another
ensure_comfy = lambda: None  # server.py hands in its own
ask_claude = None
VARIED = ["shot", "view", "body_position", "expression", "setting", "lighting"]  # the facets Phase 3 balances
CHAT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "render": {"type": "boolean"},
        "positive": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "object", "properties": {
            "tag": {"type": "string"}, "yumeko_category": {"type": "string"},
            "role": {"type": "string", "enum": ["identity", "outfit", "variable", "style"]}},
            "required": ["tag", "yumeko_category", "role"]}},
    },
    "required": ["reply", "render", "positive", "tags"],
}
CHAT_SYSTEM = """You are Yumeko, lazy, playful and a little smug, helping design an anime character's base image. She is an
adult woman. Keep replies to two sentences. Write positive as Danbooru tags, comma separated, describing one adult character,
fully clothed, always SFW; never describe a child. Include her stable look and one simple pose and background. List every tag in
tags with a category out of appearance, shot, pose, clothing, expression, action, setting, background, prop, lighting,
composition, style. Give each tag a role: identity for what never changes on her (hair, eyes, marks, jewellery, glasses), outfit for
every piece of her signature outfit, variable for the rest. Always define a full signature outfit, head to toe, in specific tags with
colours, and keep those tags word for word in later replies unless the user asks to change her clothes.
Set render true when the request calls for new pictures, false when you only answer."""
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
                "accepted": len(j["dataset"]["accepted"]), "finished": bool(j.get("lora"))})
    return sorted(out, key=lambda j: (not j["star"], j["agent"].lower(), -j["phase"]))


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


def render(positive: str, negative: str | None, seed: int, size=SIZE, loras=(), prefix: str | None = None, safe: str = SAFE) -> Image.Image:
    m = morpheus.MODELS[MODEL]
    default = m["workflow"].negative
    g = m["workflow"](positive, size[0], size[1], seed, m["steps"], negative=safe + (default if negative is None else negative), prefix=m["workflow"].prefix if prefix is None else prefix)
    return morpheus.generate(_with_loras(g, loras))


def _with_loras(graph: dict, loras) -> dict:
    model = ["1", 0]
    for i, (name, strength) in enumerate(loras):
        graph[f"6{i}"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model, "lora_name": name, "strength_model": strength}}
        model = [f"6{i}", 0]
    graph["8"]["inputs"]["model"] = model
    return graph


def _grid(job: dict, positive: str, negative: str | None, prefix: str) -> list:
    pics = []
    for i in range(9):
        job["busy"] = f"Rendering {i + 1} of 9"
        save(job)
        seed = random.randrange(2 ** 31)
        name = f"{prefix}_{time.time_ns():x}.png"
        render(positive, negative, seed).save(ROOT / job["id"] / name)
        pics.append({"file": name, "seed": seed, "positive": positive, "negative": negative or morpheus.MODELS[MODEL]["workflow"].negative})
    return pics


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


def pick_style(req: dict) -> dict:
    job = load(req.get("id"))
    if req.get("style") not in STYLES:
        raise ValueError("Unknown style.")
    job["style"] = req["style"]
    return save(job)


def chat(req: dict) -> dict:
    job = load(req.get("id"))
    text = str(req.get("text", "")).strip()
    if not text:
        raise ValueError("Say something first.")
    p = job["phase1"]
    p["chat"].append({"role": "user", "text": text})

    def work(j):
        q = j["phase1"]
        j["busy"] = "Yumeko is thinking"
        save(j)
        context = f"[Character: {j['agent']['name']}. {j['agent']['identity_summary'] or 'New character, designed only from this chat.'}]\n[Current tags: {q['positive']}]\n{text}"
        answer, q["session"] = ask_claude(context, q["session"], ghost=True, system=CHAT_SYSTEM, schema=CHAT_SCHEMA)
        q["chat"].append({"role": "yumeko", "text": answer["reply"]})
        if answer.get("positive"):
            q["positive"], q["tags"] = answer["positive"], answer.get("tags") or []
        if answer.get("render") and q["positive"]:
            q["grid"] = _grid(j, STYLES[j["style"]]["tags"] + ADULT + q["positive"], None, "p1")
        return j
    return _background(job, "Yumeko is thinking", work)


def _manifest(job: dict, pick: dict, anchor: str, role: str, tags: list) -> dict:
    source = ROOT / job["id"] / pick["file"]
    w, h = Image.open(source).size
    name = f"{anchor}.png"
    shutil.copy(source, ROOT / job["id"] / name)
    return {"schema_version": 1, "kind": "yumeko.base-image", "anchor_id": anchor, "anchor_role": role, "file": name, "agent": job["agent"],
            "base_image": {"file": name, "width": w, "height": h, "seed": pick["seed"]},
            "generation": {"model": job["model"], "anime_style": job["style"], "positive_prompt": pick["positive"], "negative_prompt": pick["negative"]},
            "tags": tags}


def approve(req: dict) -> dict:
    """Phase 1 or 2: this grid picture is it. Its manifest sits beside it, and the job moves on."""
    job = load(req.get("id"))
    grid = job["phase1"]["grid"] if job["phase"] == 1 else job["phase2"]["grid"]
    pick = next((g for g in grid if g["file"] == req.get("file")), None)
    if not pick:
        raise ValueError("Pick a picture from the grid.")
    if job["phase"] == 1:
        tags = [t | {"source": "claude"} for t in job["phase1"]["tags"]]
        job["bases"] = [_manifest(job, pick, "base-1", "initial", tags)]
        job["phase2"]["positive"], job["phase2"]["negative"] = pick["positive"], pick["negative"]
        job["phase"] = 2
    elif job["phase"] == 2:
        tags = [t | {"source": "human"} for t in job["bases"][0]["tags"]]
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

    def work(j):
        q = j["phase2"]
        q["grid"] = _grid(j, q["positive"], q["negative"], "p2")
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
NOT_LOOK = {"1girl", "solo", "adult", "adult woman", "mature female", "anime style"}
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


def plan(job: dict) -> list:
    """One batch for the current outfit: in each facet, the tag used least so far, counting this batch too. Ties go by chance."""
    counts = dict(job["dataset"]["counts"])
    tags_of = {f: [e["tag"] for e in CATALOG["tags"] if e.get("facet") == f] for f in VARIED}
    style = STYLES[job["style"]]["tags"]
    p, g = outfit_plan(job), current_group(job)
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
    job = load(req.get("id"))
    if job["phase"] != 3:
        raise ValueError("This job isn't building its set.")
    if any(c.get("keep") is None for c in job["dataset"]["batch"]):
        raise ValueError("Keep or discard every picture first.")
    _commit(job)
    plans = plan(job)

    def work(j):
        d = j["dataset"]
        d["batch"], d["rounds"] = [], d["rounds"] + 1
        for i, p in enumerate(plans):
            j["busy"] = f"Rendering {i + 1} of {len(plans)}"
            save(j)
            seed = random.randrange(2 ** 31)
            name = f"cand_{time.time_ns():x}.png"
            render(p["positive"], p["negative"], seed).save(ROOT / j["id"] / name)
            d["batch"].append({"file": name, "seed": seed, "positive": p["positive"], "negative": p["negative"], "planned": p["planned"], "tags": list(p["tags"]), "group": p["group"], "keep": None})
        return j
    return _background(job, "Planning", work)


def _commit(job: dict) -> None:
    d = job["dataset"]
    for c in d["batch"]:
        if c.get("keep"):
            d["accepted"].append({"file": c["file"], "tags": c["tags"], "planned": c["planned"], "group": c.get("group", "signature")})
            for t in c["tags"]:
                d["counts"][t] = d["counts"].get(t, 0) + 1
        else:
            (ROOT / job["id"] / c["file"]).unlink(missing_ok=True)
    d["batch"] = []


def decide(req: dict) -> dict:
    job = load(req.get("id"))
    c = next((c for c in job["dataset"]["batch"] if c["file"] == req.get("file")), None)
    if not c:
        raise ValueError("That candidate is gone.")
    if "keep" in req:
        c["keep"] = bool(req["keep"])
    if isinstance(req.get("tags"), list):
        c["tags"] = [str(t).strip() for t in req["tags"] if str(t).strip()]
    return save(job)


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
    if any(c.get("keep") is None for c in job["dataset"]["batch"]):
        raise ValueError("Keep or discard every picture first.")
    _commit(job)
    if len(job["dataset"]["accepted"]) < MIN_SET:
        raise ValueError(f"Needs {MIN_SET} accepted pictures, you have {len(job['dataset']['accepted'])}.")
    if training_now():
        raise ValueError("Another LoRA is training.")
    if not GPU.acquire(timeout=1):
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
            (data / f"{i:03}.txt").write_text(", ".join([trigger] + [t.replace("_", " ") for t in tags]), encoding="utf-8")
        steps = BIG_STEPS if len(job["dataset"]["accepted"]) > BIG_SET else STEPS
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8188/free", b'{"unload_models":true,"free_memory":true}',
                                                       {"Content-Type": "application/json"}), timeout=10)
        cmd = [str(TRAIN_PY), str(TRAINER), "--pretrained_model_name_or_path", str(CHECKPOINT), "--train_data_dir", str(d / "dataset"),
               "--output_dir", str(d / "lora"), "--output_name", "lora", "--max_train_steps", str(steps), "--save_every_n_steps", "500"]
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


def stages_of(job: dict) -> list:
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
            render(j["agent"]["trigger_token"] + ", " + STYLES[j["style"]]["tags"] + ADULT + "1girl, solo, upper body, looking at viewer, smile, garden, daytime", None, 501,
                   loras=[(_stage_file(j, s), 1.0)]).save(ROOT / j["id"] / name)
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


def use_defaults(job: dict) -> dict:
    wf = morpheus.MODELS[MODEL]["workflow"]
    return {"quality": ADULT + "1girl, solo, " + wf.prefix + STYLES[job["style"]]["tags"], "negative": "nude, " + wf.negative, "always_positive": job["agent"]["trigger_token"], "always_negative": NO_MINORS.rstrip(", ")}


def use(req: dict) -> dict:
    job = load(req.get("id"))
    if not job.get("lora"):
        raise ValueError("Pick a stage first.")
    text = str(req.get("prompt", "")).strip()
    if not text:
        raise ValueError("Describe the picture.")
    d = use_defaults(job)
    quality = str(req["quality"]) if "quality" in req else d["quality"]
    negative = str(req["negative"]) if req.get("negative") is not None else d["negative"]
    strength = float(req.get("strength", job["lora"]["strength"]))
    size = (int(req.get("width", SIZE[0])), int(req.get("height", SIZE[1])))
    seed = int(req["seed"]) if str(req.get("seed", "")).strip() else random.randrange(2 ** 31)
    job["lora"]["strength"] = strength

    def work(j):
        name = f"use_{time.time_ns():x}.png"
        positive = d["always_positive"] + ", " + quality + text
        render(positive, negative, seed, size, [(j["lora"]["file"], strength)], prefix="", safe=NO_MINORS).save(ROOT / j["id"] / name)
        j["gallery"].insert(0, {"file": name, "prompt": text, "quality": quality, "negative": negative, "seed": seed, "strength": strength,
                                "width": size[0], "height": size[1], "sent_positive": positive, "sent_negative": NO_MINORS + negative})
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
                    out.append({"id": j["id"], "name": j["agent"]["name"], "trigger": j["lora"]["trigger"], "file": j["lora"]["file"], "strength": j["lora"]["strength"]})
    return out


def panel_new(req: dict) -> dict:
    pid = f"{time.time_ns():x}"
    panel_dir(pid).mkdir(parents=True)
    return panel_save({"id": pid, "title": str(req.get("title") or "New panel"), "tags": "", "prompt": "", "negative": "", "width": 1024, "height": 1024,
                       "characters": [], "versions": [], "composition": None, "busy": None, "error": None})


def panel_edit(req: dict) -> dict:
    p = panel_load(req.get("id"))
    for k in ("title", "tags", "prompt", "negative", "characters", "width", "height"):
        if k in req:
            p[k] = req[k]
    return panel_save(p)


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
                work(p)
        except Exception as e:
            p["error"] = str(e)
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
        render(positive, q["negative"] or None, seed, size, loras).save(panel_dir(q["id"]) / name)
        q["composition"] = name
        q["versions"] = [{"file": name, "label": "Composition", "seed": seed}] + q["versions"]
    return _panel_background(p, "Rendering composition", work)


def _inpaint(source: Image.Image, mask: Image.Image, positive: str, negative: str | None, lora: str, strength: float, denoise: float, seed: int) -> Image.Image:
    """One character's pass: crop around her mask, redo the crop large with her LoRA, paste back through a soft mask."""
    W, H = source.size
    mask = mask.convert("L").resize((W, H)).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(6))
    box = mask.getbbox()
    if not box:
        raise ValueError("That character's mask is empty; paint her first.")
    pad = 48
    box = (max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad))
    cw, ch = box[2] - box[0], box[3] - box[1]
    scale = 1216 / max(cw, ch)
    rw, rh = int(cw * scale) // 8 * 8, int(ch * scale) // 8 * 8
    folder = morpheus.COMFY_INPUT / "yumeko"
    folder.mkdir(parents=True, exist_ok=True)
    source.crop(box).resize((rw, rh), Image.LANCZOS).save(folder / "crop.png")
    mask.crop(box).resize((rw, rh), Image.LANCZOS).convert("RGB").save(folder / "mask.png")
    m = morpheus.MODELS[MODEL]
    g = m["workflow"](positive, rw, rh, seed, m["steps"], negative=SAFE + (negative or m["workflow"].negative), prefix=m["workflow"].prefix)
    g["21"] = {"class_type": "LoadImage", "inputs": {"image": "yumeko/crop.png"}}
    g["22"] = {"class_type": "LoadImageMask", "inputs": {"image": "yumeko/mask.png", "channel": "red"}}
    g["23"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["1", 2]}}
    g["24"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["23", 0], "mask": ["22", 0]}}
    g["8"]["inputs"]["latent_image"], g["8"]["inputs"]["denoise"] = ["24", 0], denoise
    result = morpheus.generate(_with_loras(g, [(lora, strength)])).resize((cw, ch), Image.LANCZOS)
    out = source.copy()
    out.paste(result, box[:2], mask.crop(box))
    return out


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
    start = str(req.get("from") or (p["versions"][0]["file"] if p["versions"] else ""))
    if not start:
        raise ValueError("Render a composition first.")
    chosen = [c for c in p["characters"] if c.get("masked")] if req.get("all") else [c for c in p["characters"] if c["id"] == req.get("character")]
    if not chosen:
        raise ValueError("Paint a mask first.")

    def work(q):
        d = panel_dir(q["id"])
        im = Image.open(d / start).convert("RGB")
        for c in chosen:
            q["busy"] = f"Painting {c['name']}"
            panel_save(q)
            seed = random.randrange(2 ** 31)
            positive = c["trigger"] + ", " + ADULT + "1girl, fully clothed, " + (c.get("action") or "") + ", " + q["tags"].replace("_", " ")
            im = _inpaint(im, Image.open(d / f"mask_{c['id']}.png"), positive, q["negative"] or None, c["file"], float(c.get("strength", 1.0)), float(c.get("denoise", 0.62)), seed)
            name = f"pass_{time.time_ns():x}.png"
            im.save(d / name)
            q["versions"].insert(0, {"file": name, "label": c["name"], "seed": seed, "denoise": c.get("denoise", 0.62)})
    return _panel_background(p, "Painting", work)


def panel_delete(req: dict) -> dict:
    shutil.rmtree(panel_dir(req.get("id")), ignore_errors=True)
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


ACTIONS = {"outfits": set_outfits, "create": create, "delete": delete, "star": star, "previews": previews, "style": pick_style, "chat": chat, "approve": approve,
           "phase2": phase2_render, "batch": batch, "decide": decide, "unaccept": remove_accepted, "train": start_training,
           "stage_test": compare_stages, "stage": choose_stage, "use": use,
           "stop": lambda r: stop_training(load(r.get("id"))),
           "panel_new": panel_new, "panel_edit": panel_edit, "panel_compose": panel_compose, "panel_mask": panel_mask,
           "panel_pass": panel_pass, "panel_delete": panel_delete}


def job_view(job_id: str) -> dict:
    job = load(job_id)
    if job["train"].get("state") == "running":
        train_state(job)
    if job["phase"] == 3 and not job["dataset"].get("plan"):
        outfit_plan(job)
        save(job)
    wf = morpheus.MODELS[MODEL]["workflow"]
    return job | {"defaults": use_defaults(job), "sent": {"prefix": wf.prefix, "safe": SAFE, "sampler": "Euler a, normal", "steps": morpheus.MODELS[MODEL]["steps"], "cfg": 5.0, "checkpoint": CHECKPOINT.name}, "balance": {f: {e["tag"]: job["dataset"]["counts"].get(e["tag"], 0) for e in CATALOG["tags"] if e.get("facet") == f} for f in VARIED}}
