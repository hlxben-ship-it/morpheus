"""Agent packages: one self-contained folder per agent that any app can read without knowing Soulforge.

    agents/<id>/agent.json   the manifest, format "glia-agent/2", every path relative to the folder
    agents/<id>/portrait.png the base image; idle/*.png are overlays of the same size and origin
    agents/<id>/bust.png     head and shoulders, for dialogue boxes

Soulforge keeps working on output/soulforge/<slug>/; export() writes the package from there whenever the agent is
brought to life, so the package always matches her current look. Morpheus's own chat reads its host from here.
"""
import json
import re
import shutil
import time
from pathlib import Path

from PIL import Image

import morpheus

ROOT = morpheus.ROOT / "agents"
FORMAT = "glia-agent/2"
FORMATS = ["glia-agent/1", FORMAT]  # read; older ones are migrated in memory
HOST = "morpheus"  # the agent who answers in Dreamforge's chat
MOODS = ["happy", "laugh", "surprised", "shy", "thinking", "sad", "angry", "yes", "no", "wave", "wink", "excited"]
# what each mood plays: a mouth, eyes held, a blush, a gesture of the head, an arm, an icon over her head, for hold ms.
# One table in web/glia-data.json, read by the page, the player and this package, so every host plays the same face
DATA = json.loads((Path(__file__).parent / "web" / "glia-data.json").read_text(encoding="utf-8"))
MOOD_RECIPES = DATA["moods"]
WORLDS = DATA["worlds"]  # the shared scene themes: name, light, the layers that have a second variant
SCENES = morpheus.OUTPUT / "soulforge" / "_scenes"
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
    rest = rest.strip()
    # only a name-like title ("the Dream Shaper"); "a calm" is the start of a description, not a title
    return rest if sep and rest.lower().startswith("the ") and len(rest) < 40 else ""


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
    sc = agent.get("scene") or {}
    scene = {"default": None, "cut": None, "emblem": None, "worlds": {}}
    for name, f in [("emblem", sc.get("motif")), ("cut", sc.get("cut") if sc.get("base") == agent["base"]["file"] else None)]:
        if f and src(f).exists():
            (tmp / "scene").mkdir(exist_ok=True)
            shutil.copyfile(src(f), tmp / "scene" / f"{name}.png")
            scene[name] = f"scene/{name}.png"
    world = sc.get("world") or {}
    if all(world.get(l) and src(world[l]).exists() for l in ("sky", "far", "near")):  # her own world: home
        (tmp / "worlds" / "home").mkdir(parents=True)
        for l in ("sky", "far", "near"):
            shutil.copyfile(src(world[l]), tmp / "worlds" / "home" / f"{l}.png")
        scene["worlds"]["home"] = {"name": f"{agent['name']}'s world", "light": world.get("light"),
                                   "layers": {l: f"worlds/home/{l}.png" for l in ("sky", "far", "near")}}
    for theme, w in WORLDS.items():  # every shared theme, with its second variants
        layers = {}
        for l in ("sky", "far", "near"):
            for alt in ("", "2"):
                f = SCENES / f"{theme}_{l}{alt}.png"
                if f.exists():
                    (tmp / "worlds" / theme).mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, tmp / "worlds" / theme / f"{l}{alt}.png")
                    layers[l + alt] = f"worlds/{theme}/{l}{alt}.png"
        if all(l in layers for l in ("sky", "far")):
            scene["worlds"][theme] = {"name": w["name"], "light": w["light"], "layers": layers}
    scene["default"] = "home" if "home" in scene["worlds"] else None
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
        "moods": {m: MOOD_RECIPES[m] for m in MOODS if m in MOOD_RECIPES},
        "loops": LOOPS,
        "scene": scene,
        "colors": dialogue_colors((scene["worlds"].get("home") or {}).get("light")),
        "models": {"picture": "NoobAI-XL 1.1 (pixel art from prompt tags)", "edits": "Qwen-Image-Edit 2511",
                   "animation": "FLUX.2 [klein] 4B"},
        "voice_seed": sum(map(ord, agent["base"]["file"])) % 997,
        "created_by": "morpheus/soulforge",
        "updated": time.strftime("%Y-%m-%d"),
    }
    (tmp / "agent.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)  # only ever a package this function wrote; the source stays in output/soulforge
    tmp.rename(out)
    return out


KIT = Path(__file__).parent / "export"  # what goes into every export besides her package: FORMAT.md, demo.html
PLAYER = Path(__file__).parent / "web" / "glia-avatar.js"


def player_version() -> str:
    m = re.search(r"VERSION = '([^']+)'", PLAYER.read_text(encoding="utf-8"))
    return m.group(1) if m else "?"


def readme(m: dict) -> str:
    """How a product uses her: the live player, the stills, the persona."""
    moods = ", ".join(m["moods"])
    worlds = ", ".join(f"`{k}` ({w['name']})" for k, w in m["scene"]["worlds"].items())
    return f"""# {m['name']}{f", {m['title']}" if m.get('title') else ''}

An avatar for your product: a pixel-art character with a persona, made in Morpheus' Soulforge.
Format `{m['format']}`, player `glia-avatar.js` {player_version()}. Picture: {m['models']['picture']}; edits: {m['models']['edits']}; animation: {m['models'].get('animation', '')}.

> {m['persona']['soul']}

## Quick look

Double-click `demo.html`: it has the player and every image inlined, so it runs straight from the folder
(that's why it's big). Your own product loads `glia-avatar.js` and the folder instead, over http(s):
browsers don't load modules or `agent.json` from `file://`.

## Three ways to use her

### 1. The live player (web)

`glia-avatar.js` is one ES module, no dependencies. It breathes, blinks, sways her hair, follows the pointer
with her eyes, plays moods, moves her mouth to text and draws her world behind her.

```html
<div id="avatar" style="width:340px"></div>
<script type="module">
  import {{ GliaAvatar }} from './glia-avatar.js';
  const me = await GliaAvatar.load(document.querySelector('#avatar'), './');
  me.mood('happy');                       // returns how long the mood holds, in ms
  me.say('Hello! Nice to meet you.', {{ mood: 'happy' }});  // mouth only, the player is silent
  me.think(true);                         // pondering while your backend works; me.think(false) ends it
  await me.scene('city');                 // another world; me.scene(false) shows her without one
</script>
```

Moods: {moods}.
Worlds: {worlds}. Her own world (`home`) is the default.
Options of `load(el, root, opts)` / `mount(el, manifest, opts)` are documented at the top of `glia-avatar.js`.

### 2. Still images (any UI, no JS)

- `portrait.png`: her full picture; `bust.png`: head and shoulders, square
- `stills/<mood>.png`: one frame per mood, the same size as the portrait
- `scene/cut.png`: her figure cut out, transparent, to put over your own backdrop
- `worlds/<name>/sky.png, far.png, near.png`: the world layers, back to front

Scale them with nearest-neighbour (`image-rendering: pixelated`), never smooth: it's pixel art,
one art pixel is {m['art']['pixel']} image pixels.

### 3. Her persona in your LLM

Use `agent.json → persona.system_prompt` as the system prompt of your own model. To let her face follow the
answer, ask the model to start each reply with a mood tag and strip it before showing the text:

```
Start every reply with [mood:<one of {moods}>].
```

```js
const [, mood, text] = reply.match(/^\\[mood:(\\w+)\\]\\s*([\\s\\S]*)/) || [, null, reply];
me.say(text, {{ mood }}); if (mood) me.mood(mood);
```

## Files

See `FORMAT.md` for every field of `agent.json` (units, optional parts, how to render her your own way:
Unity, native, terminal) and how older formats migrate.
"""


def demo(out: Path, m: dict) -> str:
    """demo.html with the player and the whole package inlined, so it opens from file:// with a double-click."""
    import base64
    files = {f.relative_to(out).as_posix(): "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
             for f in out.rglob("*.png")}
    player = PLAYER.read_text(encoding="utf-8")
    package = json.dumps({"manifest": m, "files": files}, ensure_ascii=False).replace("</", r"<\/")
    page = (KIT / "demo.html").read_text(encoding="utf-8")
    return page.replace("/*PLAYER*/", player, 1).replace("/*PACKAGE*/null", package, 1)


def bundle(agent: dict, stills: dict[str, bytes]) -> tuple[str, bytes]:
    """The shareable zip of a living agent: her package fresh from her current look, the player, docs, demo, stills."""
    import io
    import zipfile
    out = export(agent)
    m = load(agent["slug"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
        z.write(PLAYER, "glia-avatar.js")
        z.write(KIT / "FORMAT.md", "FORMAT.md")
        z.writestr("demo.html", demo(out, m))
        z.writestr("README.md", readme(m))
        for mood, png in stills.items():
            if mood in m["moods"] and png.startswith(b"\x89PNG"):
                z.writestr(f"stills/{mood}.png", png)
    return f"{agent['slug']}-{FORMAT.replace('/', '-')}.zip", buf.getvalue()


def dialogue_colors(light: dict | None) -> dict:
    """The dialogue box in her world's light: the frame in its glow, a deep night fill."""
    glow = (light or {}).get("glow") or "#b48cff"
    return {"frame": glow, "fill": "#120c2a", "name": "#ffffff"}


def load(agent_id: str) -> dict | None:
    f = ROOT / agent_id / "agent.json"
    if not f.is_file():
        return None
    manifest = json.loads(f.read_text(encoding="utf-8"))
    try:
        return migrate(manifest)
    except ValueError:
        return None


def _v1_to_v2(m: dict) -> dict:
    """/1 had two mood tables (a list and the recipes) and only her own world, flat in scene."""
    m = dict(m)
    recipes = m.pop("mood_recipes", None) or {}
    sc = m.get("scene") or {}
    worlds = {}
    if sc.get("world_sky"):
        worlds["home"] = {"name": f"{m['name']}'s world", "light": sc.get("light"),
                          "layers": {"sky": sc["world_sky"], "far": sc.get("world_far"), "near": sc.get("world_near")}}
    m.update(format="glia-agent/2", moods=recipes,
             scene={"default": "home" if worlds else None, "cut": sc.get("cut"), "emblem": sc.get("emblem"), "worlds": worlds})
    return m


MIGRATIONS = {"glia-agent/1": _v1_to_v2}


def migrate(manifest: dict) -> dict:
    """An older manifest, upgraded in memory step by step to FORMAT (the same chain as the player's migrate())."""
    m = manifest
    for _ in range(10):
        if m.get("format") == FORMAT:
            return m
        step = MIGRATIONS.get(m.get("format"))
        if not step:
            raise ValueError(f"can't read format {m.get('format')!r}; this Morpheus reads {', '.join(FORMATS)}")
        m = step(m)
    raise ValueError("migration loop")


def ensure(agent: dict) -> dict | None:
    """The package of a living agent, written now if it's missing or older than her look."""
    manifest = load(agent["slug"])
    current = (agent.get("idle") or {}).get("base") == (agent.get("base") or {}).get("file")
    portrait = ROOT / agent["slug"] / "portrait.png"
    if current and (not manifest or not portrait.exists()
                    or portrait.stat().st_mtime < (morpheus.OUTPUT / agent["base"]["file"]).stat().st_mtime
                    or manifest.get("updated") is None or json.loads((ROOT / agent["slug"] / "agent.json").read_text(encoding="utf-8")).get("format") != FORMAT):
        export(agent)
        manifest = load(agent["slug"])
    return manifest
