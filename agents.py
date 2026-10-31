"""Agent packages: one self-contained folder per agent that any app can read without knowing Soulforge.

    agents/<id>/agent.json   the manifest, format "glia-agent/1", every path relative to the folder
    agents/<id>/portrait.png the base image; idle/*.png are overlays of the same size and origin
    agents/<id>/bust.png     head and shoulders, for dialogue boxes

Soulforge keeps working on output/soulforge/<slug>/; export() writes the package from there whenever the agent is
brought to life, so the package always matches her current look. Morpheus's own chat reads its host from here.
"""
import json
import shutil
import time
from pathlib import Path

from PIL import Image

import morpheus

ROOT = morpheus.ROOT / "agents"
FORMAT = "glia-agent/1"
HOST = "morpheus"  # the agent who answers in Dreamforge's chat
MOODS = ["happy", "laugh", "surprised", "shy", "thinking", "sad", "angry", "yes", "no", "wave", "wink", "excited"]
# what each mood plays: a mouth, eyes held, a blush, a gesture of the head, an arm, an icon over her head, for hold ms.
# The same table as the page's EMOTES, as data, so every host plays the same face for the same mood
MOOD_RECIPES = {
    "happy": {"mouth": "smile", "blush": .7, "gesture": "tilt", "hold": 3200, "icon": "note"},
    "wave": {"mouth": "smile", "arm": "wave", "gesture": "tilt", "hold": 2800},
    "laugh": {"mouth": "open3", "eyes": "closed", "blush": 1, "gesture": "laugh", "hold": 2200, "chuckle": True, "icon": "note"},
    "surprised": {"mouth": "open2", "gesture": "jolt", "hold": 1800, "icon": "bang", "brow": 1},
    "sad": {"mouth": "frown", "eyes": "half", "gesture": "droop", "hold": 3800, "icon": "drop", "arm": "heart", "gloom": 1},
    "angry": {"mouth": "frown", "eyes": "half", "gesture": "shake", "hold": 2600, "icon": "vein", "brow": -1},
    "thinking": {"mouth": "smirk", "eyes": "glance", "gesture": "ponder", "hold": 3400, "icon": "ask", "arm": "chin"},
    "yes": {"mouth": "smile", "gesture": "nod", "hold": 1500},
    "no": {"mouth": "frown", "gesture": "shake", "hold": 1600},
    "shy": {"mouth": "smirk", "eyes": "glance_l", "blush": 1, "gesture": "duck", "hold": 3000, "icon": "heart", "arm": "heart"},
    "wink": {"mouth": "smirk", "eyes": "wink", "gesture": "cock", "hold": 1300, "icon": "star"},
    "excited": {"mouth": "open3", "blush": .5, "gesture": "hop", "hold": 2200, "icon": "star", "brow": 1},
}
# while the host waits on her model: a slow ponder, a glance up and back, a slow blink, over and over
LOOPS = {
    "thinking": {"every": 4200, "steps": [
        {"at": 0, "eyes": "glance", "gesture": "ponder", "mouth": "smirk"},
        {"at": 1900, "eyes": None},
        {"at": 2300, "blink": "slow"},
        {"at": 3100, "eyes": "glance_l"},
        {"at": 3900, "eyes": None},
    ]},
}
IDLE = ["half", "closed", "glance", "glance_l", "smirk", "blush", "smile", "frown", "open1", "open2", "open3"]
VOICE = ("short and vivid, one to three sentences; confident, warm, a little mysterious and playful; "
         "speaks in the first person, now and then a dream image (weaving, a spark, the moon)")


def title_of(agent: dict) -> str:
    """'the Dream Shaper' out of 'Morpheus is the Dream Shaper, an enigmatic …'."""
    first = agent.get("soul", "").split(",")[0].split(".")[0]
    _, sep, rest = first.partition(" is ")
    return rest.strip() if sep and len(rest) < 40 else ""


def system_prompt(agent: dict, title: str, voice: str) -> str:
    return (f"You are {agent['name']}{', ' + title if title else ''}. {agent.get('soul', '')}\n"
            f"Every reply is spoken by you, in character: {voice}. Never say you are an AI model or an assistant; "
            f"you are {agent['name']}. Stay helpful first: the persona colours the words, it never changes what you "
            f"do. Never use Markdown, emojis or stage directions such as *smiles*; your face shows the mood.")


def bust_box(idle: dict, size: tuple[int, int]) -> list[int]:
    """Head and shoulders, square, from the eye box: the eyes sit at 40% height, the shoulders in, on the art grid."""
    W, H = size
    B = idle.get("block", 6)
    ew = idle["eyes"][2] - idle["eyes"][0]
    cx, ey = (idle["eyes"][0] + idle["eyes"][2]) / 2, (idle["eyes"][1] + idle["eyes"][3]) / 2
    side = min(W, H, round(ew * 4.2 / B) * B)
    x0 = max(0, min(W - side, round((cx - side / 2) / B) * B))
    y0 = max(0, min(H - side, round((ey - side * .4) / B) * B))
    return [x0, y0, x0 + side, y0 + side]


def export(agent: dict) -> Path:
    """Write agents/<slug>/ from the agent's current, living look. Needs the idle set of the current base."""
    idle = agent.get("idle") or {}
    if not agent.get("base") or idle.get("base") != agent["base"]["file"]:
        raise ValueError(f"{agent['name']} isn't alive with her current look; bring her to life first.")
    out = ROOT / agent["slug"]
    tmp = ROOT / f".{agent['slug']}-{int(time.time())}"
    (tmp / "idle").mkdir(parents=True)
    src = lambda f: morpheus.OUTPUT / f
    shutil.copyfile(src(agent["base"]["file"]), tmp / "portrait.png")
    portrait = Image.open(tmp / "portrait.png")
    box = bust_box(idle, portrait.size)
    portrait.crop(box).save(tmp / "bust.png")
    overlays = {}
    for name, f in idle.get("frames", {}).items():
        if src(f).exists():
            shutil.copyfile(src(f), tmp / "idle" / f"{name}.png")
            overlays[name] = f"idle/{name}.png"
    for name in ("rig", "plate"):
        if idle.get(name) and src(idle[name]).exists():
            shutil.copyfile(src(idle[name]), tmp / "idle" / f"{name}.png")
    scene = {}
    sc = agent.get("scene") or {}
    if sc.get("motif") or sc.get("world") or sc.get("cut"):
        (tmp / "scene").mkdir()
        for name, f in [("emblem", sc.get("motif")), ("cut", sc.get("cut") if sc.get("base") == agent["base"]["file"] else None),
                        *[(f"world_{l}", (sc.get("world") or {}).get(l)) for l in ("sky", "far", "near")]]:
            if f and src(f).exists():
                shutil.copyfile(src(f), tmp / "scene" / f"{name}.png")
                scene[name] = f"scene/{name}.png"
        if (sc.get("world") or {}).get("light"):
            scene["light"] = sc["world"]["light"]
    title, voice = agent.get("title") or title_of(agent), agent.get("voice") or VOICE
    manifest = {
        "format": FORMAT,
        "id": agent["slug"],
        "name": agent["name"],
        "title": title,
        "persona": {"soul": agent.get("soul", ""), "voice": voice, "system_prompt": system_prompt(agent, title, voice)},
        "art": {"style": "16-bit pixel art", "pixel": idle.get("block", 6), "size": list(portrait.size),
                "portrait": "portrait.png", "bust": {"file": "bust.png", "box": box},
                "anchors": {"eyes": idle["eyes"], "mouth": idle["mouth"], "chest": idle.get("chest"), "waist": idle.get("waist")}},
        "idle": {"overlays": overlays, "rig": "idle/rig.png" if (tmp / "idle" / "rig.png").exists() else None,
                 "plate": "idle/plate.png" if (tmp / "idle" / "plate.png").exists() else None},
        "moods": MOODS,
        "mood_recipes": MOOD_RECIPES,
        "loops": LOOPS,
        "scene": scene,
        "colors": dialogue_colors(scene.get("light")),
        "voice_seed": sum(map(ord, agent["base"]["file"])) % 997,
        "created_by": "morpheus/soulforge",
        "updated": time.strftime("%Y-%m-%d"),
    }
    (tmp / "agent.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)  # only ever a package this function wrote; the source stays in output/soulforge
    tmp.rename(out)
    return out


def dialogue_colors(light: dict | None) -> dict:
    """The dialogue box in her world's light: the frame in its glow, a deep night fill."""
    glow = (light or {}).get("glow") or "#b48cff"
    return {"frame": glow, "fill": "#120c2a", "name": "#ffffff"}


def load(agent_id: str) -> dict | None:
    f = ROOT / agent_id / "agent.json"
    if not f.is_file():
        return None
    manifest = json.loads(f.read_text(encoding="utf-8"))
    return manifest if manifest.get("format") == FORMAT else None


def ensure(agent: dict) -> dict | None:
    """The package of a living agent, written now if it's missing or older than her look."""
    manifest = load(agent["slug"])
    current = (agent.get("idle") or {}).get("base") == (agent.get("base") or {}).get("file")
    portrait = ROOT / agent["slug"] / "portrait.png"
    if current and (not manifest or not portrait.exists()
                    or portrait.stat().st_mtime < (morpheus.OUTPUT / agent["base"]["file"]).stat().st_mtime
                    or manifest.get("updated") is None):
        export(agent)
        manifest = load(agent["slug"])
    return manifest
