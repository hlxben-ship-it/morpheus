"""Dawnforge Playground: one free local-video page per folder, output/playground/<id>/.

The page's state lives here (model, selects, prompt, frames, two LoRA paths + strengths), the LoRA header read and the
clip shelf. Rendering runs in the engine worker (engine/), started by server.py. The id is the folder name, so
/dawnforge/playground/<id> can be rebuilt from a folder on disk.
"""
import base64
import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import morpheus
import playground_models as pm

ROOT = morpheus.OUTPUT / "playground"
ID = re.compile(r"[0-9a-z_]{1,40}")
# The models (stages, steps, options) are the schema in playground_models.json. The rest is shared by all models.
LISTS = {"durations": [2, 3, 4, 5, 6, 8, 10],
         "aspects": ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]}
FRAMES = ("start", "end")
HEADER_MAX = 64 << 20  # a safetensors header is JSON; far above this is not a LoRA


def folder(play_id: str) -> Path:
    if not ID.fullmatch(str(play_id)):
        raise ValueError("This playground does not exist.")
    return ROOT / play_id


def options() -> dict:
    """What the page's pickers offer: the model schema plus the shared lists."""
    return {"models": pm.models(), **LISTS}


def fresh_state(play_id: str) -> dict:
    model = pm.models()[0]
    return {"id": play_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model["id"],
            "params": pm.resolve(model, {}), "duration": 5, "resolution": model["resolutions"][0]["id"], "aspect": "21:9",
            "prompt": "", "start": None, "end": None,
            "loras": [{"path": "", "strength": 1.0}, {"path": "", "strength": 1.0}]}


def write(state: dict) -> dict:
    """Through a temp name, so the file on disk is always a whole one."""
    path = folder(state["id"]) / "playground.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)
    return state


def new(req: dict) -> dict:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    play_id = stamp
    n = 1
    while (ROOT / play_id).exists():
        n += 1
        play_id = f"{stamp}_{n}"
    return write(fresh_state(play_id))


def load(play_id: str) -> dict:
    try:
        state = json.loads((folder(play_id) / "playground.json").read_text(encoding="utf-8"))
    except OSError:
        raise ValueError("This playground does not exist.")
    ms = pm.models()
    model = next((m for m in ms if m["id"] == state.get("model")), ms[0])  # a model taken out of the schema file: the first
    given = dict(state.get("params") or {})
    if "steps" in state:  # the one field of a playground made before the schema
        given.setdefault("main.steps", state.pop("steps"))
    state["model"], state["params"] = model["id"], pm.resolve(model, given)
    state["resolution"] = pm.resolve_resolution(model, state.get("resolution"))  # e.g. an old 480p: the model's default
    for kind in FRAMES:  # a frame the user deleted by hand is just gone
        if state.get(kind) and not (folder(play_id) / state[kind]).is_file():
            state[kind] = None
    for lora in state.get("loras", []):
        lora["missing"] = bool(lora.get("path")) and not Path(lora["path"]).is_file()
    return state


def save(req: dict) -> dict:
    state = load(str(req.get("id", "")))
    ms = pm.models()
    before = state["model"]
    for key, allowed in (("model", [m["id"] for m in ms]), ("duration", LISTS["durations"]),
                         ("aspect", LISTS["aspects"])):
        if key in req:
            if req[key] not in allowed:
                raise ValueError(f"Unknown {key}.")
            state[key] = req[key]
    model = next(m for m in ms if m["id"] == state["model"])
    if "resolution" in req:  # the resolutions are the model's own
        state["resolution"] = pm.resolve_resolution(model, req["resolution"], strict=True)
    else:
        state["resolution"] = pm.resolve_resolution(model, state["resolution"])
    if "params" in req:  # what the user set: anything the model does not allow is an error
        state["params"] = pm.resolve(model, req["params"], strict=True)
    elif state["model"] != before:  # another pipeline: its steps and options start from its own defaults
        state["params"] = pm.resolve(model, {})
    else:
        state["params"] = pm.resolve(model, state["params"])
    if "prompt" in req:
        state["prompt"] = str(req["prompt"])[:8000]
    if "loras" in req:
        loras = req["loras"]
        if not isinstance(loras, list) or len(loras) != 2:
            raise ValueError("Two LoRA slots expected.")
        state["loras"] = [{"path": str(x.get("path", "")), "strength": float(x.get("strength", 1.0))} for x in loras]
    if "start" in req and req["start"] is None:  # removing a frame; adding one goes through frame()
        state["start"] = None
    if "end" in req and req["end"] is None:
        state["end"] = None
    state.pop("missing", None)
    return load(write(state)["id"])


def frame(req: dict) -> dict:
    """A start or end picture as a data URL, stored in the playground folder."""
    kind = str(req.get("kind", ""))
    if kind not in FRAMES:
        raise ValueError("Start or end frame only.")
    head, _, body = str(req.get("data", "")).partition(",")
    ext = {"data:image/png;base64": "png", "data:image/jpeg;base64": "jpg", "data:image/webp;base64": "webp"}.get(head)
    if not ext or not body:
        raise ValueError("A PNG, JPEG or WebP picture, please.")
    state = load(str(req.get("id", "")))
    name = f"{kind}_{int(time.time())}.{ext}"
    (folder(state["id"]) / name).write_bytes(base64.b64decode(body))
    state[kind] = name
    state.pop("missing", None)
    return load(write(state)["id"])


def pick_lora(req: dict) -> dict:
    """A native Windows file dialog on this machine, so the page can keep the full path (a browser hides it).
    The server runs on the user's own PC; the dialog opens there."""
    # The owner form must be shown (invisible, topmost, activated): a dialog owned by a form that was never shown
    # opens behind the browser with no taskbar entry, and it looks as if nothing happens.
    script = ("Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Application]::EnableVisualStyles(); "
              "$f = New-Object System.Windows.Forms.Form; $f.Text = 'Morpheus'; $f.TopMost = $true; $f.Opacity = 0; "
              "$f.ShowInTaskbar = $false; $f.StartPosition = 'CenterScreen'; $f.Size = New-Object System.Drawing.Size(1, 1); "
              "$f.Show(); $f.Activate(); "
              "$d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = 'Choose a LoRA'; "
              "$d.Filter = 'LoRA (*.safetensors)|*.safetensors|All files|*.*'; "
              "if ($d.ShowDialog($f) -eq 'OK') { [Console]::Out.Write($d.FileName) }; $f.Close()")
    out = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", script], capture_output=True, text=True,
                         timeout=600).stdout.strip()
    return {"path": out}


def summarize(meta: dict | None, ranks: list) -> list:
    """Readable [label, value] rows from a LoRA's header. Only what the file says; what it does not say is named,
    so a missing trigger word or strength reads as 'not in this file' instead of as a bug."""
    m = {str(k): v for k, v in (meta or {}).items()}
    pick = lambda *keys: next((str(m[k]) for k in keys if m.get(k) not in (None, "", "None")), None)
    rows = []

    def add(label, value):
        if value:
            rows.append([label, value])

    add("Title", pick("modelspec.title", "ss_output_name"))
    base = pick("ss_sd_model_name", "ss_base_model_version", "modelspec.architecture")
    version = pick("ss_ltx_version")
    add("Base model", " · ".join(x for x in (("LTX " + version) if version else None, base) if x))
    kind = pick("modelspec.architecture")
    preset = pick("ss_lora_target_preset")
    add("Type", " · ".join(x for x in (kind, f"trained for {preset}" if preset else None) if x))
    dim = pick("ss_network_dim") or (str(ranks[0]) if ranks else None)
    alpha = pick("ss_network_alpha")
    if dim:
        text = f"rank {dim}" + (f", alpha {alpha}" if alpha else "")
        try:
            text += f" (alpha/rank = {float(alpha) / float(dim):g}: 1 means it was trained at full scale)"
        except (TypeError, ValueError):
            pass
        add("Size", text)
    add("Training", ", ".join(x for x in (
        f"{pick('ss_num_epochs', 'ss_epoch')} epochs" if pick("ss_num_epochs", "ss_epoch") else None,
        f"{pick('ss_max_train_steps', 'ss_steps')} steps" if pick("ss_max_train_steps", "ss_steps") else None,
        f"{pick('ss_num_train_items')} items" if pick("ss_num_train_items") else None) if x))
    add("Trained", (pick("modelspec.date") or "")[:10])
    # trigger words: explicit keys first, then kohya's tag counts
    words = pick(*[k for k in m if re.search(r"trigger|activation|trained_words|trigger_phrase", k, re.I)])
    if not words and m.get("ss_tag_frequency"):
        try:
            counts: dict = {}
            for tags in json.loads(m["ss_tag_frequency"]).values():
                for tag, n in tags.items():
                    counts[tag] = counts.get(tag, 0) + int(n)
            words = ", ".join(f"{t} ({n})" for t, n in sorted(counts.items(), key=lambda x: -x[1])[:12]) + "  [most used training tags]"
        except (ValueError, AttributeError, TypeError):
            pass
    rows.append(["Trigger words", words or "none stored in this file (the creator's page usually lists them)"])
    strength = pick(*[k for k in m if re.search(r"(recommend|suggest|prefer|default).*(strength|weight|scale)|^(weight|strength)$", k, re.I)])
    rows.append(["Suggested strength", strength or "none stored in this file"])
    return rows


def inspect(req: dict) -> dict:
    """Debug output: the safetensors header as it is. Reads the first bytes only, never the weights."""
    path = Path(str(req.get("path", "")))
    if not path.is_file():
        raise ValueError("File missing.")
    with open(path, "rb") as f:
        size = path.stat().st_size
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError("Not a safetensors file (too short).")
        n = struct.unpack("<Q", raw)[0]
        if n > HEADER_MAX or n + 8 > size:
            raise ValueError("Not a safetensors file (the header length does not fit).")
        try:
            header = json.loads(f.read(n))
        except ValueError:
            raise ValueError("Not a safetensors file (the header is not JSON).")
    meta = header.pop("__metadata__", None)
    tensors = {k: v for k, v in header.items() if isinstance(v, dict)}
    dtypes: dict = {}
    for v in tensors.values():
        dtypes[v.get("dtype")] = dtypes.get(v.get("dtype"), 0) + 1
    ranks = sorted({min(v["shape"]) for k, v in tensors.items()
                    if re.search(r"lora_(A|B|down|up)", k) and len(v.get("shape", [])) == 2})
    keys = [f"{k}  {v.get('dtype')} {v.get('shape')}" for k, v in list(tensors.items())[:60]]
    return {"path": str(path), "name": path.name, "summary": summarize(meta, ranks), "bytes": size, "header_bytes": n, "metadata": meta,
            "tensor_count": len(tensors), "dtypes": dtypes, "lora_ranks_guess": ranks,
            "first_keys": keys, "more_keys": max(0, len(tensors) - len(keys))}


CLIP_FORMAT = 1


def lora_info(slot: int, lora: dict) -> dict:
    """One used LoRA slot as it was at render time: path, strength and what the file's header says about itself."""
    path = Path(lora["path"])
    info = {"slot": slot, "path": lora["path"], "name": path.name, "strength": lora["strength"], "missing": not path.is_file()}
    if info["missing"]:
        return info
    info["bytes"] = path.stat().st_size
    try:
        info["header"] = inspect({"path": lora["path"]})["summary"]  # the readable rows, as on the page
    except ValueError as e:
        info["header_error"] = str(e)
    return info


def snapshot(play_id: str) -> dict:
    """The settings of a playground frozen at the moment Render is pressed. The render seed takes this when it starts, runs
    the render from it and hands the same dict to add_clip(), so the meta describes what was rendered even if the page
    was changed while it ran."""
    state = load(play_id)
    model = next(m for m in pm.models() if m["id"] == state["model"])
    res = next(r for r in model["resolutions"] if r["id"] == state["resolution"])
    return {"format": CLIP_FORMAT, "playground": state["id"], "prompt": state["prompt"],
            "model": {"id": model["id"], "label": model["label"]},
            "params": dict(state["params"]), "total_steps": pm.total_steps(model, state["params"]),
            "resolution": {k: res[k] for k in ("id", "label", "target", "render") if k in res},
            "aspect": state["aspect"], "duration": state["duration"],
            "start_frame": state["start"], "end_frame": state["end"],  # file names in the playground folder (never overwritten)
            "loras": [lora_info(n + 1, l) for n, l in enumerate(state["loras"]) if l["path"]]}


def make_thumb(mp4: Path, out: Path) -> None:
    """A small picture from the middle of the clip for the shelf. Best effort: no thumbnail is not an error."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2, 0))
        ok, frame = cap.read()
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            cv2.imwrite(str(out), cv2.resize(frame, (160, max(1, round(160 * h / w))), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass


def add_clip(play_id: str, video: Path, snap: dict, seed=None, seconds=None) -> dict:
    """A render has arrived: the clip goes into output/playground/<id>/clips/ and its meta json next to it
    (<name>.mp4, <name>.json, <name>_thumb.jpg). The json is written last, through a temp name, so the shelf never sees a
    clip without its mp4. `snap` is the snapshot() taken when the render started; seed and seconds (render time) are
    what only the render knows."""
    clips_dir = folder(play_id) / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem, n = stamp, 1
    while (clips_dir / f"{stem}.mp4").exists():
        n += 1
        stem = f"{stamp}_{n}"
    mp4 = clips_dir / f"{stem}.mp4"
    shutil.move(str(video), mp4)
    make_thumb(mp4, clips_dir / f"{stem}_thumb.jpg")
    meta = {**snap, "video": f"clips/{mp4.name}", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed, "render_seconds": seconds}
    tmp = clips_dir / f"{stem}.tmp"
    tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(clips_dir / f"{stem}.json")
    return meta


def clip_meta(play_id: str, video: str) -> tuple[Path, dict]:
    """A shelf clip of this playground (video = "playground/<id>/clips/<name>.mp4" as clips() lists it) and its meta."""
    clips_dir = (folder(play_id) / "clips").resolve()
    mp4 = (morpheus.OUTPUT / str(video)).resolve()
    if mp4.parent != clips_dir or mp4.suffix != ".mp4" or not mp4.is_file():
        raise ValueError("This clip is gone.")
    try:
        meta = json.loads(mp4.with_suffix(".json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        meta = {}
    return mp4, meta


def upscaled_snap(source: dict, video: str, method: str, label: str, factor: float) -> dict:
    """The meta of an upscaled clip: what the source was rendered with, plus the upscale and a link back to the source."""
    snap = {k: v for k, v in source.items() if k not in ("video", "created", "render_seconds", "upscale")}
    res = source.get("resolution")
    base = res.get("label") if isinstance(res, dict) else res
    snap["resolution"] = {**(res if isinstance(res, dict) else {}), "label": f"{base or 'clip'} ↑ {label} ×{factor:g}"}
    snap["upscale"] = {"method": method, "factor": factor, "source": video, "source_upscale": source.get("upscale")}
    return snap


SIZES: dict = {}  # (path, mtime) -> "720p" | "1080p" | "2K": read once per file


def size_label(mp4: Path) -> str:
    """The clip's real output size as the shelf badge shows it, from the file (so upscaled clips are right too):
    by the shorter side, under 600 px = 480p, under 900 px = 720p (1248×704, 1280×704), under 1300 = 1080p (1920×1088, ×1.5 upscales),
    else 2K (×2 upscales and up). Empty when the file can't be read."""
    key = (str(mp4), mp4.stat().st_mtime)
    if key not in SIZES:
        label = ""
        try:
            import cv2
            cap = cv2.VideoCapture(str(mp4))
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap.release()
            short = min(w, h)
            label = "" if not short else "480p" if short < 600 else "720p" if short < 900 else "1080p" if short < 1300 else "2K"
        except Exception:
            pass
        SIZES[key] = label
    return SIZES[key]


def clips() -> list:
    """The clip shelf, newest first: every Playground render, from all playgrounds (what add_clip() wrote:
    output/playground/<id>/clips/<name>.mp4 + <name>.json meta, optional <name>_thumb.jpg)."""
    out = []

    def rel(p: Path) -> str:
        return p.relative_to(morpheus.OUTPUT).as_posix()

    def read(js: Path) -> dict | None:
        try:
            return json.loads(js.read_text(encoding="utf-8-sig"))  # -sig: a meta edited by hand may carry a BOM
        except (OSError, ValueError):
            return None

    def name(x):  # {"label": ..} in the meta, or plain text in one written by hand
        return x.get("label") if isinstance(x, dict) else x

    for js in ROOT.glob("*/clips/*.json"):
        meta, mp4 = read(js), js.with_suffix(".mp4")
        if meta is None or not mp4.is_file():
            continue
        thumb = js.with_name(js.stem + "_thumb.jpg")
        label = " · ".join(str(x) for x in (name(meta.get("model")), f"{meta['duration']} s" if meta.get("duration") else None,
                                            name(meta.get("resolution"))) if x)
        out.append({"video": rel(mp4), "thumb": rel(thumb) if thumb.is_file() else "", "prompt": meta.get("prompt", ""),
                    "created": meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mp4.stat().st_mtime)),
                    "label": label or "Playground", "origin": "playground", "playground": js.parent.parent.name,
                    "size": size_label(mp4)})
    out.sort(key=lambda c: c["created"], reverse=True)
    return out[:60]


def delete_clip(req: dict) -> dict:
    """A Playground clip, its sidecar and thumbnail. (Dawnforge's own clips go through /api/wake/delete.)"""
    clips_dir = (folder(str(req.get("id", ""))) / "clips").resolve()
    mp4 = (morpheus.OUTPUT / str(req.get("video", ""))).resolve()
    if mp4.parent != clips_dir or mp4.suffix != ".mp4" or not mp4.is_file():
        raise ValueError("This clip is gone.")
    mp4.with_suffix(".json").unlink(missing_ok=True)
    mp4.with_name(f"{mp4.stem}_thumb.jpg").unlink(missing_ok=True)
    mp4.unlink()
    return {"ok": True}


ACTIONS = {"new": new, "save": save, "frame": frame, "pick_lora": pick_lora, "inspect": inspect, "delete_clip": delete_clip}
