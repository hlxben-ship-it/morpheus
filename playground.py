"""Dawnforge Playground: one free local-video page per folder, output/playground/<id>/.

The page's state lives here (model, selects, prompt, frames, two LoRA slots: library id + strength), the LoRA header read and the
clip shelf. Rendering runs in the WSL engine worker (engine/wsl_runner.py), started by server.py. The id is the folder name, so
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

import lora_library
import morpheus
import playground_models as pm

ROOT = morpheus.OUTPUT / "playground"
ID = re.compile(r"[0-9a-z_]{1,40}")
# The models (stages, steps, options) are the schema in playground_models.json. The rest is shared by all models.
LISTS = {"durations": [2, 3, 4, 5, 6, 8, 10],
         "aspects": ["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9"]}
FRAMES = ("start", "end")
REFS = 4  # H3 reference slots beside the main image (H3 takes up to 9)
HEADER_MAX = 64 << 20  # a safetensors header is JSON; far above this is not a LoRA
# a remix timeline entry: one of this playground's shelf clips, or (Studio) a still picture shown for N seconds
REMIX_REF = re.compile(r"clips/[0-9A-Za-z_.-]+\.mp4|images/[0-9A-Za-z_.-]+\.(?:png|jpe?g|webp)@(?:[0-9]{1,2})(?:\.[0-9])?")
STILL_MIN, STILL_MAX = 0.5, 30.0


def remix_ok(ref) -> bool:
    if not isinstance(ref, str) or not REMIX_REF.fullmatch(ref):
        return False
    return "@" not in ref or STILL_MIN <= float(ref.rsplit("@", 1)[1]) <= STILL_MAX


def remix_still(ref: str):
    """A still timeline entry -> (relative picture path, seconds); a clip -> (ref, None)."""
    if "@" not in ref:
        return ref, None
    path, sec = ref.rsplit("@", 1)
    return path, float(sec)
REMIX_MAX = 200


def folder(play_id: str) -> Path:
    if not ID.fullmatch(str(play_id)):
        raise ValueError("This playground does not exist.")
    return ROOT / play_id


def options() -> dict:
    """What the page's pickers offer: the model schema plus the shared lists."""
    ms = pm.models()
    return {"models": ms, "lora_families": {m["id"]: lora_library.family_of_model(m["id"]) for m in ms}, **LISTS}


def fresh_state(play_id: str) -> dict:
    model = pm.models()[0]
    return {"id": play_id, "kind": "video", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model["id"],
            "params": pm.resolve(model, {}), "duration": 5, "resolution": model["resolutions"][0]["id"], "aspect": "21:9",
            "prompt": "", "start": None, "end": None, "main_first": True, "refs": [None] * REFS, "last_ref": None,
            "loras": [{"id": "", "strength": 1.0}, {"id": "", "strength": 1.0}], "seed": None}


SEED_MAX = 2**31 - 1


def clean_seed(value):
    """The render seed the user set: a whole number 1 to 2^31-1, or None (empty field) for a random one per render."""
    if value is None or str(value).strip() == "":
        return None
    try:
        seed = int(str(value).strip())
    except ValueError:
        raise ValueError("Seed: a whole number, or empty for random.")
    if not 1 <= seed <= SEED_MAX:
        raise ValueError(f"Seed: 1 to {SEED_MAX}, or empty for random.")
    return seed


def h3_pictures(state: dict) -> list:
    """The pictures an H3 render sends, in H3's own numbering: the timeline anchors first (image_start, then image_end),
    then the references (image_refs). The one source of the <Picture N> names: the engine, the slot labels and the prompt
    writer all read this list. The main image ("start") is the first frame unless main_first is off, then it is a
    reference; the ref with last_ref is the last frame. H3 needs at least one reference: with anchors only, the first
    anchor is sent as a reference too (the spike setup), that copy has no visible name (shown False)."""
    main, refs = state.get("start"), list(state.get("refs") or [])[:REFS]
    refs += [None] * (REFS - len(refs))
    last = state.get("last_ref")
    if not (isinstance(last, int) and 0 <= last < REFS and refs[last]):
        last = None
    first = state.get("main_first", True) is not False
    out = []

    def add(slot, role, file, shown=True):
        out.append({"slot": slot, "role": role, "file": file, "tag": f"<Picture {len(out) + 1}>", "shown": shown})

    if main and first:
        add("main", "start", main)
    if last is not None:
        add(f"ref{last + 1}", "end", refs[last])
    if main and not first:
        add("main", "ref", main)
    for i, f in enumerate(refs):
        if f and i != last:
            add(f"ref{i + 1}", "ref", f)
    if out and not any(p["role"] == "ref" for p in out):
        add(out[0]["slot"], "ref", out[0]["file"], shown=False)
    return out


def write(state: dict) -> dict:
    """Through a temp name, so the file on disk is always a whole one."""
    state.pop("pictures", None)  # derived on load, never stored
    if state.pop("_clean", False):  # a write that is no settings edit (Studio bookkeeping, Remix timeline)
        pass
    elif state.get("kind") == "studio":  # Studio (studio.py): the open tile now differs from its rendered state
        state["dirty"] = True
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
    if state.get("kind") == "image":  # its settings are image_playground.py's; a video save would overwrite them
        raise ValueError("This is an image playground.")
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
    refs = list(state.get("refs") or [])[:REFS]
    state["refs"] = [r if r and (folder(play_id) / r).is_file() else None for r in refs + [None] * (REFS - len(refs))]
    state.setdefault("main_first", True)
    if not (isinstance(state.get("last_ref"), int) and 0 <= state["last_ref"] < REFS and state["refs"][state["last_ref"]]):
        state["last_ref"] = None
    state["pictures"] = h3_pictures(state)
    state["loras"] = clean_slots(state.get("loras"), model["id"])
    remix = state.get("remix")
    state["remix"] = [r for r in (remix if isinstance(remix, list) else []) if remix_ok(r)][:REMIX_MAX]
    try:  # a playground made before the seed field, or a seed edited by hand: random
        state["seed"] = clean_seed(state.get("seed"))
    except ValueError:
        state["seed"] = None
    return state


def clean_slots(slots, model_id: str) -> list:
    """Two {id, strength} slots. A slot whose id is not in the library, or is for another model family, is empty:
    that also drops the free file paths of playgrounds made before the library (no migration, seed decision #4)."""
    family = lora_library.family_of_model(model_id)
    known = {e["id"]: e for e in lora_library.entries()}
    out = []
    for slot in (list(slots) if isinstance(slots, list) else [])[:2]:
        slot = slot if isinstance(slot, dict) else {}
        entry = known.get(str(slot.get("id", "")))
        ok = entry and entry["family"] == family
        try:
            strength = float(slot.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        out.append({"id": entry["id"] if ok else "", "strength": strength,
                    **({"missing": True} if ok and not lora_library.file_path(entry).is_file() else {})})
    return out + [{"id": "", "strength": 1.0}] * (2 - len(out))


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
    if "defaults" in req:  # Studio: the tile's default prompt (None = the model's, studio_models.json)
        import studio_models
        state["defaults"] = studio_models.clean_defaults(req["defaults"])
    elif state.get("kind") == "studio" and state["model"] != before:
        state.pop("defaults", None)  # another model: its own defaults
    if "loras" in req:
        loras = req["loras"]
        if not isinstance(loras, list) or len(loras) != 2:
            raise ValueError("Two LoRA slots expected.")
        family = lora_library.family_of_model(state["model"])
        for x in loras:
            entry = lora_library.get(str((x or {}).get("id", ""))) if (x or {}).get("id") else None
            if (x or {}).get("id") and not entry:
                raise ValueError("This LoRA is not in the library.")
            if entry and entry["family"] != family:
                raise ValueError(f"{entry['name']} is a {lora_library.FAMILIES[entry['family']]} LoRA; this model takes {lora_library.FAMILIES.get(family, 'none')}.")
            if not 0 <= float((x or {}).get("strength", 1.0)) <= 2:
                raise ValueError("LoRA strength: 0 to 2.")
        state["loras"] = [{"id": str((x or {}).get("id", "")), "strength": float((x or {}).get("strength", 1.0))} for x in loras]
    if "start" in req and req["start"] is None:  # removing a frame; adding one goes through frame()
        state["start"] = None
    if "end" in req and req["end"] is None:
        state["end"] = None
    if "main_first" in req:
        state["main_first"] = bool(req["main_first"])
    if "last_ref" in req:  # at most one ref is the last frame: one index or None
        last = req["last_ref"]
        if last is not None and (not isinstance(last, int) or isinstance(last, bool) or not 0 <= last < REFS):
            raise ValueError("Unknown reference slot.")
        state["last_ref"] = last
    if "seed" in req:
        state["seed"] = clean_seed(req["seed"])
    if "remix" in req:  # the Remix timeline: shelf clips in play order, repeats allowed; a deleted clip may stay (export refuses it)
        remix = req["remix"]
        if not isinstance(remix, list) or len(remix) > REMIX_MAX or not all(remix_ok(r) for r in remix):
            raise ValueError("Unknown remix timeline.")
        state["remix"] = list(remix)
    state.pop("missing", None)
    if set(req) <= {"id", "remix"}:  # only the Remix timeline changed: no settings edit
        state["_clean"] = True
    return load(write(state)["id"])


def store_picture(state: dict, prefix: str, data) -> str:
    """A data URL picture into the playground folder; returns its file name."""
    head, _, body = str(data or "").partition(",")
    ext = {"data:image/png;base64": "png", "data:image/jpeg;base64": "jpg", "data:image/webp;base64": "webp"}.get(head)
    if not ext or not body:
        raise ValueError("A PNG, JPEG or WebP picture, please.")
    name = f"{prefix}_{time.time_ns() // 1_000_000}.{ext}"
    (folder(state["id"]) / name).write_bytes(base64.b64decode(body))
    return name


def ref_slot(req: dict) -> int:
    slot = req.get("slot")
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < REFS:
        raise ValueError("Unknown reference slot.")
    return slot


def frame(req: dict) -> dict:
    """A start or end picture as a data URL, stored in the playground folder."""
    kind = str(req.get("kind", ""))
    if kind not in FRAMES:
        raise ValueError("Start or end frame only.")
    state = load(str(req.get("id", "")))
    state[kind] = store_picture(state, kind, req.get("data"))
    state.pop("missing", None)
    return load(write(state)["id"])


def ref(req: dict) -> dict:
    """One H3 reference slot (0-3): a data URL picture is stored, data None empties the slot (and takes its "Last frame"
    toggle with it)."""
    slot = ref_slot(req)
    state = load(str(req.get("id", "")))
    if req.get("data") is None:
        state["refs"][slot] = None
        if state.get("last_ref") == slot:
            state["last_ref"] = None
    else:
        state["refs"][slot] = store_picture(state, f"ref{slot + 1}", req.get("data"))
    state.pop("missing", None)
    return load(write(state)["id"])


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
    """One used LoRA slot as it was at render time: the library entry, its file, the strength and what the header says."""
    entry = lora_library.get(lora["id"]) or {"id": lora["id"], "name": lora["id"], "family": "", "file": ""}
    path = lora_library.file_path(entry) if entry["file"] else Path("")
    info = {"slot": slot, "id": entry["id"], "name": entry["name"], "path": str(path), "file": entry["file"],
            "strength": lora["strength"], "missing": not entry["file"] or not path.is_file()}
    if entry.get("sha256"):  # which file exactly: a LoRA deleted and uploaded again under its name keeps the id
        info["sha256"] = entry["sha256"]
    if info["missing"]:
        return info
    info["bytes"] = path.stat().st_size
    try:
        info["header"] = inspect({"path": str(path)})["summary"]  # the readable rows, as on the page
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
    extra = {}
    prompt = state["prompt"]
    if state.get("kind") == "studio":  # Studio: the default prompt goes in front; the sidecar keeps both parts
        import studio_models
        d = studio_models.defaults(state["model"], state.get("defaults"))
        prompt = studio_models.join(d["prompt"], state["prompt"])
        extra = {"studio_prompt": state["prompt"], "defaults": state.get("defaults")}
    return {**extra, "format": CLIP_FORMAT, "playground": state["id"], "prompt": prompt,
            "model": {"id": model["id"], "label": model["label"]},
            "params": dict(state["params"]), "total_steps": pm.total_steps(model, state["params"]),
            "resolution": {k: res[k] for k in ("id", "label", "target", "render") if k in res},
            "aspect": state["aspect"], "duration": state["duration"],
            "start_frame": state["start"], "end_frame": state["end"],  # file names in the playground folder (never overwritten)
            "main_first": state["main_first"], "refs": list(state["refs"]), "last_ref": state["last_ref"],
            "pictures": h3_pictures(state),  # what an H3 render sends, numbered (render_settings reads this for H3)
            "loras": [lora_info(n + 1, l) for n, l in enumerate(state["loras"]) if l["id"]],
            "seed": state["seed"]}  # the seed the user fixed, or None; the clip meta gets the one the render really used


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
LENGTHS: dict = {}  # (path, mtime) -> seconds, for the Remix timeline


def clip_seconds(mp4: Path) -> float | None:
    key = (str(mp4), mp4.stat().st_mtime)
    if key not in LENGTHS:
        try:
            LENGTHS[key] = round(video_info(mp4)["seconds"], 2)
        except Exception:
            LENGTHS[key] = None
    return LENGTHS[key]


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


def clips(play_id: str | None = None) -> list:
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

    for js in (folder(play_id) if play_id else ROOT).glob("clips/*.json" if play_id else "*/clips/*.json"):
        meta, mp4 = read(js), js.with_suffix(".mp4")
        if meta is None or not mp4.is_file():
            continue
        thumb = js.with_name(js.stem + "_thumb.jpg")
        label = " · ".join(str(x) for x in (name(meta.get("model")), f"{meta['duration']} s" if meta.get("duration") else None,
                                            name(meta.get("resolution"))) if x)
        out.append({"video": rel(mp4), "thumb": rel(thumb) if thumb.is_file() else "", "prompt": meta.get("prompt", ""),
                    "created": meta.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mp4.stat().st_mtime)),
                    "label": label or "Playground", "origin": "playground", "playground": js.parent.parent.name,
                    "size": size_label(mp4), "seconds": clip_seconds(mp4), "looped": bool(meta.get("loop")),
                    "seed": meta.get("seed"), "render_seconds": meta.get("render_seconds"), "settings": bool(meta.get("model"))})
    out.sort(key=lambda c: c["created"], reverse=True)
    if not play_id:
        return out[:60]
    try:  # the starred clip leads the shelf, before the newest
        cover = json.loads((folder(play_id) / "playground.json").read_text(encoding="utf-8-sig")).get("cover")
    except (OSError, ValueError):
        cover = None
    for c in out:
        c["cover"] = bool(cover) and c["video"] == f"playground/{play_id}/{cover}"
    out.sort(key=lambda c: not c["cover"])
    return out


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


def loop_clip(req: dict) -> dict:
    """"Loop it": a new clip that plays the source forward, then in reverse (ping-pong), so it ends where it starts.
    The seam frames are dropped (the last source frame isn't shown twice at the turn, the first one isn't shown again
    at the end) so the loop plays without a stutter. No audio: reversed sound is noise. ffmpeg only, no GPU."""
    play_id, video = str(req.get("id", "")), str(req.get("video", ""))
    mp4, meta = clip_meta(play_id, video)
    if meta.get("loop"):
        raise ValueError("This clip is already a loop.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is not on PATH.")
    tmp = mp4.with_name(f"{mp4.stem}_loop.tmp.mp4")
    graph = ("[0:v]split[a][b];[b]trim=start_frame=1,setpts=PTS-STARTPTS,reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];"
             "[a][r]concat=n=2:v=1:a=0[v]")
    run = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-filter_complex", graph, "-map", "[v]", "-an",
                          "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(tmp)],
                         capture_output=True, text=True)
    if run.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg failed: {run.stderr.strip()[-300:]}")
    snap = {k: v for k, v in meta.items() if k not in ("video", "created", "render_seconds")}
    if snap.get("duration"):
        snap["duration"] = round(float(snap["duration"]) * 2, 1)
    snap["loop"] = {"source": video}
    return add_clip(play_id, tmp, snap, seed=meta.get("seed"), seconds=meta.get("render_seconds"))


def cut_clip(req: dict) -> dict:
    """"Cut": the section start..end (seconds) of a shelf clip becomes a new clip; the source stays. Cut on frames, not
    keyframes (re-encode, same quality as Loop it), so the new clip is exactly end_frame - start_frame frames long."""
    play_id, video = str(req.get("id", "")), str(req.get("video", ""))
    mp4, meta = clip_meta(play_id, video)
    try:
        start, end = float(req.get("start")), float(req.get("end"))
    except (TypeError, ValueError):
        raise ValueError("Cut needs a start and an end in seconds.")
    info = video_info(mp4)
    fps, total = info["fps"], round(info["seconds"] * info["fps"])
    first, last = round(start * fps), round(end * fps)
    if not (0 <= first < last <= total):
        raise ValueError(f"The cut has to lie inside the clip (0 to {total} frames) and hold at least one frame.")
    if first == 0 and last == total:
        raise ValueError("That is the whole clip: move a handle first.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is not on PATH.")
    tmp = mp4.with_name(f"{mp4.stem}_cut.tmp.mp4")
    run = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                          f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS", "-an",
                          "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(tmp)],
                         capture_output=True, text=True)
    if run.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg failed: {run.stderr.strip()[-300:]}")
    snap = {k: v for k, v in meta.items() if k not in ("video", "created", "render_seconds", "loop")}  # a cut is no ping-pong loop
    snap["duration"] = round((last - first) / fps, 1)
    snap["cut"] = {"source": video, "start_frame": first, "end_frame": last}
    return add_clip(play_id, tmp, snap, seed=meta.get("seed"), seconds=meta.get("render_seconds"))


def last_frame(req: dict) -> dict:
    """"Last frame as start": the clip's last picture becomes this playground's start frame. Read with ffmpeg on the
    server, not in the browser: a phone browser often never seeks a video it is not playing (2026-11-15)."""
    mp4, _ = clip_meta(str(req.get("clip_playground") or req.get("id", "")), str(req.get("video", "")))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is not on PATH.")
    # -sseof jumps to the last second; every frame of it comes out as one PNG after the other (-update 1 only overwrites
    # a file, never a pipe), so the last PNG in the stream is the clip's last frame. Taking the whole stream gave the
    # page the first one, ~20 frames early (2026-11-15)
    run = subprocess.run([ffmpeg, "-loglevel", "error", "-sseof", "-1", "-i", str(mp4), "-an",
                          "-f", "image2pipe", "-c:v", "png", "pipe:1"], capture_output=True)
    sig = b"\x89PNG\r\n\x1a\n"
    if run.returncode or sig not in run.stdout:
        raise ValueError(f"Could not read the last frame: {run.stderr.decode(errors='replace').strip()[-300:]}")
    data = "data:image/png;base64," + base64.b64encode(run.stdout[run.stdout.rindex(sig):]).decode()
    if req.get("data_only"):  # Studio "Resharpen last frame" puts it into the image half instead
        return {"data": data}
    if req.get("slot") is not None:  # dragged onto an H3 reference slot
        return ref({"id": req.get("id"), "slot": req.get("slot"), "data": data})
    return frame({"id": req.get("id"), "kind": "start", "data": data})


def video_info(mp4: Path) -> dict:
    """Width, height, fps, codec and length of a clip (OpenCV; the shelf's size badge reads it the same way)."""
    import cv2
    cap = cv2.VideoCapture(str(mp4))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, n, cc = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT), int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    if not (w and h and fps):
        raise ValueError(f"Could not read {mp4.name}.")
    return {"w": w, "h": h, "fps": round(fps, 3), "codec": cc, "seconds": n / fps}


def export_remix(req: dict) -> dict:
    """Remix export: the timeline's clips cut together, hard cuts, into output/playground/<id>/remix/remix_<ts>.mp4.
    All clips alike (size, fps, codec) → ffmpeg concat with stream copy (fast, lossless). Otherwise every clip is
    letterboxed into the first clip's frame and fps, then re-encoded (seed decision #2). No audio: the clips have none."""
    play_id = str(req.get("id", ""))
    state = load(play_id)
    if not state["remix"]:
        raise ValueError("The timeline is empty: drag clips from the shelf into it.")
    files, stills = [], []
    for ref_ in state["remix"]:
        rel, sec = remix_still(ref_)
        f = folder(play_id) / rel
        if not f.is_file():
            raise ValueError(f"A timeline clip is gone: {Path(rel).name}. Remove it from the timeline.")
        files.append(f)
        stills.append(sec)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is not on PATH.")
    infos = []
    for f, sec in zip(files, stills):
        if sec is None:
            infos.append(video_info(f))
        else:  # a still: its own size; fps and codec follow the clips
            from PIL import Image
            with Image.open(f) as im:
                infos.append({"w": im.width, "h": im.height, "fps": None, "codec": None, "seconds": sec, "still": True})
    clips = [i for i in infos if not i.get("still")]
    first = dict(clips[0] if clips else {**infos[0], "fps": 24.0})  # the frame: first clip's (or the first still's, 24 fps)
    for i in infos:
        if i.get("still"):
            i["fps"] = first["fps"]
    out_dir = folder(play_id) / "remix"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"remix_{stamp}.mp4"
    n = 1
    while out.exists():
        n += 1
        out = out_dir / f"remix_{stamp}_{n}.mp4"
    tmp = out.with_name(out.stem + ".tmp.mp4")
    copy = not any(stills) and all((i["w"], i["h"], i["fps"], i["codec"]) == (first["w"], first["h"], first["fps"], first["codec"]) for i in infos)
    if copy:
        listing = out.with_name(out.stem + ".txt")
        quote = lambda f: f.resolve().as_posix().replace("'", "'\\''")  # the concat list's own quoting
        listing.write_text("".join(f"file '{quote(f)}'\n" for f in files), encoding="utf-8")
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-an", "-c", "copy",
               "-movflags", "+faststart", str(tmp)]
    else:
        w, h = first["w"] - first["w"] % 2, first["h"] - first["h"] % 2  # libx264 wants even sides
        legs = "".join(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
                       f"setsar=1,fps={first['fps']},format=yuv420p[v{i}];" for i in range(len(files)))
        graph = legs + "".join(f"[v{i}]" for i in range(len(files))) + f"concat=n={len(files)}:v=1:a=0[v]"
        inputs = [a for f, sec in zip(files, stills)
                  for a in (("-i", str(f)) if sec is None else ("-loop", "1", "-framerate", str(first["fps"]), "-t", str(sec), "-i", str(f)))]
        cmd = [ffmpeg, "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
               "-map", "[v]", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(tmp)]
        listing = None
    try:
        run = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if listing:
            listing.unlink(missing_ok=True)
    if run.returncode or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"ffmpeg failed: {run.stderr.strip()[-300:]}")
    tmp.replace(out)
    return {"video": out.relative_to(morpheus.OUTPUT).as_posix(), "name": out.name, "copied": copy,
            "clips": len(files), "seconds": round(sum(i["seconds"] for i in infos), 2)}


# ---- Remaster (seed_2026_11_16_remix_remaster): an all-10Eros timeline re-rendered on 10Eros at 1080p (seed h3_shootout_rebuild) ----
REMASTER_FROM, REMASTER_TO, REMASTER_RES = "h3_10eros", "h3_10eros", "1080p"
REMASTER_SEC_PER_STEP = 9 * 60  # the H3 Flash note: 1080p is about 9 min per step (a ~5 s shot); Q6 unmeasured, a rough guide


def remaster_snap(play_id: str, ref_: str, meta: dict) -> dict:
    """The snapshot that renders a timeline clip again: everything from its meta (prompt, seed, pictures, frames, LoRAs,
    duration, aspect, steps), only the model and the resolution change. LoRAs are read from the library again."""
    model = next((m for m in pm.models() if m["id"] == REMASTER_TO), None)
    if not model:
        raise ValueError(f"The model {REMASTER_TO} is not in playground_models.json.")
    res = next((r for r in model["resolutions"] if r["id"] == REMASTER_RES), None)
    if not res:
        raise ValueError(f"{model['label']} has no {REMASTER_RES}.")
    if meta.get("seed") is None:
        raise ValueError(f"{Path(ref_).name} has no recorded seed, so it cannot be rendered again.")
    for p in meta.get("pictures") or []:
        if not (folder(play_id) / str(p.get("file", ""))).is_file():
            raise ValueError(f"A picture of {Path(ref_).name} is gone: {p.get('file')}.")
    params = dict(meta.get("params") or {})
    loras = []
    for n, old in enumerate(meta.get("loras") or []):  # the same LoRA files as the original render, or no remaster
        now = lora_info(int(old.get("slot") or n + 1), {"id": old.get("id", ""), "strength": old.get("strength", 1.0)})
        name = old.get("name") or old.get("id")
        if now["missing"]:
            raise ValueError(f"{Path(ref_).name} used the LoRA {name}, which is no longer in the library.")
        changed = (old.get("sha256") and now.get("sha256") != old["sha256"]) or                   (old.get("bytes") is not None and now.get("bytes") != old["bytes"])
        if changed:
            raise ValueError(f"The LoRA {name} is a different file now than when {Path(ref_).name} was rendered.")
        loras.append(now)
    snap = {k: v for k, v in meta.items() if k not in ("video", "created", "render_seconds", "seed", "upscale", "loop")}
    snap.update(model={"id": model["id"], "label": model["label"]}, params=params, total_steps=pm.total_steps(model, params),
                resolution={k: res[k] for k in ("id", "label", "target", "render") if k in res}, loras=loras,
                seed=meta["seed"], remaster_of=ref_)
    return snap


def remaster_plan(req: dict) -> dict:
    """What Remaster would do for this playground's timeline. Allowed only if every timeline clip is an H3 HyperFlow GGUF Q4_K_M
    render (an upscale or loop of one does not count) or already a remaster. Each distinct clip renders once; one that
    already has a remaster on the shelf is only swapped (that is how a cancelled run resumes)."""
    play_id = str(req.get("id", ""))
    state = load(play_id)
    if not state["remix"]:
        return {"ok": False, "reason": "The timeline is empty.", "shots": [], "swaps": {}, "seconds": 0}
    if any("@" in r for r in state["remix"]):
        return {"ok": False, "reason": "The timeline holds still pictures: Remaster renders clips only.", "shots": [], "swaps": {}, "seconds": 0}
    done = {}  # source ref -> its remaster on the shelf
    for j in sorted((folder(play_id) / "clips").glob("*.json")):
        try:
            m = json.loads(j.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if m.get("remaster_of") and (folder(play_id) / str(m.get("video", ""))).is_file():
            done[m["remaster_of"]] = m["video"]
    shots, swaps, seconds = [], {}, 0
    for ref_ in dict.fromkeys(state["remix"]):  # distinct, in timeline order
        name = Path(ref_).name
        if not (folder(play_id) / ref_).is_file():
            return {"ok": False, "reason": f"A timeline clip is gone: {name}.", "shots": [], "swaps": {}, "seconds": 0}
        try:
            meta = json.loads((folder(play_id) / ref_).with_suffix(".json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            meta = {}
        if meta.get("remaster_of"):
            continue  # already remastered
        if ref_ in done:
            swaps[ref_] = done[ref_]
            continue
        model = (meta.get("model") or {}).get("id")
        if model != REMASTER_FROM or meta.get("upscale") or meta.get("loop"):
            why = "is an upscale" if meta.get("upscale") else "is a loop" if meta.get("loop") else                   f"was rendered with {(meta.get('model') or {}).get('label') or 'no recorded model'}"
            return {"ok": False, "reason": f"Remaster needs every timeline clip from H3 10Eros-Max TURBO Q6_K: {name} {why}.",
                    "shots": [], "swaps": {}, "seconds": 0}
        steps = int((meta.get("params") or {}).get("main.steps") or 4)
        seconds += steps * REMASTER_SEC_PER_STEP * max(1.0, float(meta.get("duration") or 5) / 5)
        shots.append({"ref": ref_, "steps": steps, "duration": meta.get("duration")})
    if not shots and not swaps:
        return {"ok": False, "reason": "Every timeline clip is already remastered.", "shots": [], "swaps": {}, "seconds": 0}
    return {"ok": True, "reason": "", "shots": shots, "swaps": swaps, "seconds": round(seconds)}


def remaster_run(play_id: str, job: dict, render) -> dict:
    """Run a Remaster: swap in the remasters that exist, then render the rest one after the other. `render(snap)` does
    one engine render and returns (out, seed), out = {"file", "seconds"}. After each shot the new clip replaces its source
    on the timeline, so a Stop keeps what is finished and the next Remaster carries on from there."""
    plan = remaster_plan({"id": play_id})
    if not plan["ok"]:
        raise ValueError(plan["reason"])
    for old, new_ in plan["swaps"].items():
        swap_remix(play_id, old, new_)
    n = len(plan["shots"])
    job.update(shots=n, shots_done=0)
    for i, shot in enumerate(plan["shots"]):
        if job.get("_stop"):
            raise ValueError("Stopped.")
        job["shot"] = i + 1
        _, meta = clip_meta(play_id, f"playground/{play_id}/{shot['ref']}")
        snap = remaster_snap(play_id, shot["ref"], meta)
        snap["playground"] = play_id
        out, seed = render(snap)
        clip = add_clip(play_id, Path(out["file"]), snap, seed=seed, seconds=out["seconds"])
        swap_remix(play_id, shot["ref"], clip["video"])
        job["shots_done"] = i + 1
    return {"shots": n, "swapped": len(plan["swaps"]), "remix": load(play_id)["remix"]}


def swap_remix(play_id: str, old: str, new: str) -> list:
    """Every timeline entry of `old` becomes `new` (playground-relative clip refs); the rest of the cut stays."""
    state = load(play_id)
    state["remix"] = [new if r == old else r for r in state["remix"]]
    state["_clean"] = True
    write(state)
    return state["remix"]


def list_all() -> list:
    """The Local gallery: every playground, starred first, then newest first. The mini is the newest clip's thumbnail
    (none = the page shows a plain placeholder)."""
    out = []
    for js in ROOT.glob("*/playground.json"):
        try:
            st = json.loads(js.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if st.get("kind") == "image":  # Image Playground (image_playground.py): the starred or newest picture is the mini
            imgs = [p for p in (js.parent / "images").glob("*.png")] if (js.parent / "images").is_dir() else []
            cover = js.parent / str(st.get("cover") or "-")
            pick = cover if st.get("cover") and cover.is_file() else max(imgs, key=lambda p: p.stat().st_mtime, default=None)
            out.append({"id": js.parent.name, "kind": "image", "created": st.get("created", ""),
                        "updated": st.get("updated") or st.get("created", ""), "model": st.get("model", ""),
                        "prompt": (st.get("prompt") or "").strip()[:120], "clips": len(imgs), "starred": bool(st.get("starred")),
                        "thumb": pick.relative_to(morpheus.OUTPUT).as_posix() if pick else "", "video": ""})
            continue
        clips_dir = js.parent / "clips"
        metas = sorted(clips_dir.glob("*.json")) if clips_dir.is_dir() else []
        metas = [m for m in metas if m.with_suffix(".mp4").is_file()]
        thumb = next((m.with_name(m.stem + "_thumb.jpg") for m in reversed(metas) if m.with_name(m.stem + "_thumb.jpg").is_file()), None)
        newest = max(metas, key=lambda m: m.stat().st_mtime, default=None)  # the hover peek and fullscreen play this one
        cover = js.parent / str(st.get("cover") or "-")  # the starred clip, if it is still there, wins over the newest
        if st.get("cover") and cover.is_file() and cover.suffix == ".mp4":
            newest = cover.with_suffix(".json")
            if cover.with_name(cover.stem + "_thumb.jpg").is_file():
                thumb = cover.with_name(cover.stem + "_thumb.jpg")
        if st.get("kind") == "studio":  # Studio (studio.py): clips and pictures count together; a starred picture is the mini
            imgs = [p for p in (js.parent / "images").glob("*.png") if p.with_suffix(".json").is_file()] if (js.parent / "images").is_dir() else []
            if st.get("cover") and cover.is_file() and cover.suffix == ".png":
                thumb, newest = cover, None
            elif not thumb and imgs:
                thumb = max(imgs, key=lambda p: p.stat().st_mtime)
            metas = metas + imgs
            if st.get("side") == "image":
                st = {**st, "model": (st.get("image") or {}).get("model", ""), "prompt": (st.get("image") or {}).get("prompt", "")}
        out.append({"id": js.parent.name, "kind": st.get("kind") or "video", "side": st.get("side"), "created": st.get("created", ""),
                    "updated": st.get("updated") or st.get("created", ""), "model": st.get("model", ""),
                    "prompt": (st.get("prompt") or "").strip()[:120], "clips": len(metas), "starred": bool(st.get("starred")),
                    "thumb": thumb.relative_to(morpheus.OUTPUT).as_posix() if thumb else "",
                    "video": newest.with_suffix(".mp4").relative_to(morpheus.OUTPUT).as_posix() if newest else ""})
    out.sort(key=lambda p: p["updated"], reverse=True)
    out.sort(key=lambda p: not p["starred"])
    return out


def star(req: dict) -> dict:
    path = folder(str(req.get("id", ""))) / "playground.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("This playground does not exist.")
    state["starred"] = bool(req.get("starred"))
    tmp = path.with_suffix(".tmp")  # not write(): a star is no edit, "updated" stays
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"ok": True, "starred": state["starred"]}


def star_clip(req: dict) -> dict:
    """One clip per playground can be starred: it is the playground's mini, hover and fullscreen in the Local gallery and
    leads the shelf. Starring another moves the star; starred=False takes it off."""
    play_id = str(req.get("id", ""))
    d = folder(play_id)
    path = d / "playground.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("This playground does not exist.")
    if req.get("starred"):
        mp4 = (morpheus.OUTPUT / str(req.get("video", ""))).resolve()
        if mp4.parent != (d / "clips").resolve() or mp4.suffix != ".mp4" or not mp4.is_file():
            raise ValueError("This clip is gone.")
        state["cover"] = f"clips/{mp4.name}"
    else:
        state.pop("cover", None)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"ok": True, "cover": state.get("cover")}


def delete(req: dict) -> dict:
    """A whole playground: its settings, frames and every clip in it."""
    d = folder(str(req.get("id", "")))
    if not (d / "playground.json").is_file():
        raise ValueError("This playground does not exist.")
    shutil.rmtree(d)
    return {"ok": True}


def lora_users(lora_id: str) -> list:
    """Playgrounds whose saved slots name this library LoRA: [{id, title}]."""
    out = []
    for f in sorted(ROOT.glob("*/playground.json")):
        try:
            state = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if any(isinstance(l, dict) and l.get("id") == lora_id for l in state.get("loras", [])):
            out.append({"id": f.parent.name, "title": state.get("title") or f.parent.name})
    return out


def clear_lora(lora_id: str) -> None:
    """Empty every slot that names this LoRA (it was deleted from the library)."""
    for user in lora_users(lora_id):
        path = folder(user["id"]) / "playground.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["loras"] = [{"id": "", "strength": l.get("strength", 1.0)} if isinstance(l, dict) and l.get("id") == lora_id else l
                          for l in state.get("loras", [])]
        write(state)


def use_clip(req: dict) -> dict:
    """"Use these settings": the playground takes everything a shelf clip was rendered with (prompt, model, params,
    resolution, aspect, duration, frames, references, LoRAs) and its seed, so Render makes the same clip again.
    Frames are file names in this playground's folder; one deleted since is simply empty. A LoRA no longer in the
    library is dropped. A loop's duration is its source's (the loop doubled it)."""
    play_id = str(req.get("id", ""))
    _, meta = clip_meta(play_id, str(req.get("video", "")))
    model = meta.get("model")
    if not isinstance(model, dict) or not model.get("id"):
        raise ValueError("This clip has no settings to take.")
    duration = meta.get("duration")
    if duration and meta.get("loop"):
        duration = round(float(duration) / 2, 1)
    duration = int(duration) if duration and float(duration).is_integer() else duration
    by_slot = {l.get("slot"): l for l in meta.get("loras") or [] if lora_library.get(str(l.get("id", "")))}
    loras = [{"id": by_slot[n]["id"], "strength": by_slot[n]["strength"]} if n in by_slot else {"id": "", "strength": 1.0}
             for n in (1, 2)]
    res = meta.get("resolution")
    schema = next((m for m in pm.models() if m["id"] == model["id"]), None)
    params = pm.resolve(schema, meta.get("params") or {}) if schema else meta.get("params") or {}  # an old clip's stale params are dropped
    upd = {"id": play_id, "model": model["id"], "params": params, "prompt": meta.get("prompt", ""),
           "loras": loras, "seed": meta.get("seed"), "main_first": meta.get("main_first", True)}
    if isinstance(res, dict) and res.get("id"):
        upd["resolution"] = res["id"]
    if duration in LISTS["durations"]:
        upd["duration"] = duration
    if meta.get("aspect") in LISTS["aspects"]:
        upd["aspect"] = meta["aspect"]
    save(upd)
    state = load(play_id)
    here = lambda f: f if f and (folder(play_id) / str(f)).is_file() else None
    state["start"], state["end"] = here(meta.get("start_frame")), here(meta.get("end_frame"))
    refs = list(meta.get("refs") or [])[:REFS]
    state["refs"] = [here(r) for r in refs + [None] * (REFS - len(refs))]
    last = meta.get("last_ref")
    state["last_ref"] = last if isinstance(last, int) and 0 <= last < REFS and state["refs"][last] else None
    state.pop("missing", None)
    return load(write(state)["id"])


ACTIONS = {"star_clip": star_clip, "star": star, "delete": delete, "last_frame": last_frame, "loop": loop_clip, "cut": cut_clip, "new": new, "save": save, "frame": frame, "ref": ref, "inspect": inspect, "delete_clip": delete_clip, "use_clip": use_clip, "remix_export": export_remix, "remaster_plan": remaster_plan}
