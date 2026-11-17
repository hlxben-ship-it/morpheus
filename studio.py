"""Studio (Glia seed combined_playground): a Local playground of kind "studio" that does video and pictures on one page.

One folder output/playground/<id>/ like the other two kinds. playground.json holds the video half at the top level (so every
playground.py action works on it unchanged: render, upscale, Loop it, Cut, Remix, frames) and the image half under "image"
(image_playground.py reads and writes it there). "side" says which half the page shows; the model picker lists both
families and switching to a model of the other family switches the side.

Stage 2, stateful tiles: every shelf tile is a document. Its rendered state is its sidecar (never changes); its edited
state is a working copy. The editor (the working state above) shows one tile at a time: "open" names it ("new" = the
loose, unrendered draft, else "clips/<x>.mp4" | "images/<x>.png"), "dirty" says it differs from its rendered state, and
"tiles" keeps the edited copies of the tiles that are not open (plus the draft while a render is open).
"""
import base64
import json
import re

import image_models as im
import image_playground
import playground
import playground_models as pm

SIDES = ("video", "image")
META = {"id", "kind", "created", "updated", "cover", "starred", "tiles", "open", "dirty", "remix", "_clean", "pictures", "missing", "names"}
TILE = re.compile(r"(clips/[0-9A-Za-z_.-]+\.mp4|images/[0-9A-Za-z_.-]+\.png)")


def image_half(play_id: str) -> dict:
    """A fresh image state without the fields the file keeps at the top level."""
    st = image_playground.fresh_state(play_id)
    st.update(prompt="", negative="")  # the model's defaults are the Default fields (studio_models.json), not the prompt
    return {k: v for k, v in st.items() if k not in ("id", "kind", "created")}


def fresh_settings(play_id: str) -> dict:
    """An empty draft: the defaults of both halves, video side."""
    v = {k: v for k, v in playground.fresh_state(play_id).items() if k not in META}
    return {**v, "side": "video", "image": image_half(play_id)}


def _raw(play_id: str) -> dict:
    try:
        st = json.loads((playground.folder(play_id) / "playground.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("This playground does not exist.")
    if st.get("kind") != "studio":
        raise ValueError("This is not a Studio playground.")
    st.setdefault("open", "new")  # a Studio from before stage 2: its working state is the draft (seed Q4)
    st.setdefault("dirty", False)
    st.setdefault("tiles", {})
    return st


def _put(st: dict) -> dict:
    st["_clean"] = True  # bookkeeping, no settings edit
    return playground.write(st)


def settings_of(st: dict) -> dict:
    return {k: v for k, v in st.items() if k not in META}


def _set_settings(st: dict, s: dict) -> None:
    for k in [k for k in st if k not in META]:
        del st[k]
    st.update(s)


def new(req: dict) -> dict:
    st = playground.new({})
    st.update(kind="studio", side="video", image=image_half(st["id"]), open="new", dirty=False, tiles={})
    return playground.load(_put(st)["id"])


def _studio(play_id: str) -> dict:
    st = playground.load(play_id)
    if st.get("kind") != "studio":
        raise ValueError("This is not a Studio playground.")
    return st


def side(req: dict) -> dict:
    """Switch the page to the other half; with "model", that half also takes the model (picked from the one list)."""
    play_id, to = str(req.get("id", "")), req.get("side")
    if to not in SIDES:
        raise ValueError("Video or image.")
    model = req.get("model")
    if model is not None:
        ok = any(m["id"] == model for m in pm.models()) if to == "video" else bool(im.get(model))
        if not ok:
            raise ValueError("Unknown model.")
    if model is not None and to == "image":
        image_playground.save({"id": play_id, "model": model})
    st = _studio(play_id)
    st["side"] = to
    if model is None:
        st["_clean"] = True  # only looking at the other half is no edit
    if model is not None and to == "video":
        playground.write(st)
        return playground.save({"id": play_id, "model": model})
    return playground.load(playground.write(st)["id"])


def picture_frame(req: dict) -> dict:
    """A shelf picture of the image half becomes the video half's start (or end) frame, as if it was dropped there."""
    play_id = str(req.get("id", ""))
    _studio(play_id)
    kind = req.get("kind", "start")
    src = image_playground.image_path(play_id, str(req.get("image", "")))
    data = "data:image/png;base64," + base64.b64encode(src.read_bytes()).decode()
    return playground.frame({"id": play_id, "kind": kind, "data": data})


# ---- stage 2: stateful tiles ----

def tile_exists(play_id: str, ref: str) -> bool:
    if ref == "new":
        return True
    if not TILE.fullmatch(str(ref)):
        return False
    f = playground.folder(play_id) / ref
    return f.is_file() and f.with_suffix(".json").is_file()


def _rel(play_id: str, ref: str) -> str:
    """A tile as the shelf lists it: relative to output/ ("playground/<id>/clips/x.mp4")."""
    return (playground.folder(play_id) / ref).relative_to(playground.morpheus.OUTPUT).as_posix()


def _load_rendered(play_id: str, ref: str) -> None:
    """The editor takes the tile's rendered state (its sidecar), through the existing "Use these settings" paths."""
    if ref.startswith("clips/"):
        playground.use_clip({"id": play_id, "video": _rel(play_id, ref)})
        to = "video"
    else:
        image_playground.reuse({"id": play_id, "image": _rel(play_id, ref)})
        to = "image"
    st = _raw(play_id)
    try:  # a Studio render's sidecar keeps the prompt without the defaults: the editor gets both parts back
        meta = json.loads((playground.folder(play_id) / ref).with_suffix(".json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        meta = {}
    if "studio_prompt" in meta:
        half = st if to == "video" else st.setdefault("image", {})
        half["prompt"] = meta["studio_prompt"]
        if "studio_negative" in meta:
            half["negative"] = meta["studio_negative"]
        if meta.get("defaults"):
            half["defaults"] = meta["defaults"]
        else:
            half.pop("defaults", None)
    st.update(side=to, open=ref, dirty=False)
    _put(st)


def open_tile(req: dict) -> dict:
    """Click a tile: the open tile's working state is kept as its edited copy (the draft always, a render only when it
    has edits), then the clicked tile's edited copy, else its rendered state, fills the editor."""
    play_id, ref = str(req.get("id", "")), str(req.get("tile", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, ref):
        raise ValueError("This tile is gone.")
    cur = st["open"]
    if cur == ref:
        return info({"id": play_id})
    if cur == "new" or st["dirty"]:
        st["tiles"][cur] = settings_of(st)
    else:
        st["tiles"].pop(cur, None)
    if ref in st["tiles"]:
        _set_settings(st, st["tiles"].pop(ref))
        st.update(open=ref, dirty=ref != "new")
        _put(st)
    elif ref == "new":
        _set_settings(st, fresh_settings(play_id))
        st.update(open="new", dirty=False)
        _put(st)
    else:
        st["open"] = ref
        _put(st)
        _load_rendered(play_id, ref)
    return info({"id": play_id})


def back(req: dict) -> dict:
    """"Back to rendered": the open tile drops its edits."""
    play_id = str(req.get("id", ""))
    st = _raw(play_id)
    if st["open"] == "new":
        raise ValueError("The draft has not been rendered yet.")
    _load_rendered(play_id, st["open"])
    return info({"id": play_id})


def run_source(play_id: str):
    """The tile a render starts from (None when this is no Studio)."""
    try:
        return _raw(play_id)["open"]
    except ValueError:
        return None


def rendered(play_id: str, src, refs: list) -> None:
    """A render from tile `src` arrived as `refs`: the first new tile takes over the editor if it still showed `src`
    (its working state is exactly what was run, so it is not dirty); the source falls back to its rendered state (seed
    Q1), the draft to a fresh empty one."""
    if src is None or not refs:
        return
    try:
        st = _raw(play_id)
    except ValueError:
        return
    new_ref = str(refs[0]).split(f"{play_id}/", 1)[-1]  # "images/x.png" | "clips/x.mp4" (refs may carry the folder)
    if st["open"] == src:
        st.update(open=new_ref, dirty=False)
    if src == "new":
        st["tiles"]["new"] = fresh_settings(play_id)
    else:
        st["tiles"].pop(src, None)
    _put(st)


def banner_of(play_id: str, ref: str) -> str:
    """"Loop of …", "Cut of …", "Upscale of …", "Remaster of …" for a tile made from another one."""
    if not ref.startswith("clips/"):
        return ""
    try:
        meta = json.loads((playground.folder(play_id) / ref).with_suffix(".json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return ""
    name = lambda s: str(s or "").split("/")[-1]
    for key, label in (("loop", "Loop"), ("cut", "Cut"), ("upscale", "Upscale")):
        if isinstance(meta.get(key), dict):
            return f"{label} of {name(meta[key].get('source'))}"
    if meta.get("remaster_of"):
        return f"Remaster of {name(meta['remaster_of'])}"
    return ""


def info(req: dict) -> dict:
    """What the page needs to draw the tiles: the open one, whether it has edits, every tile with edits, the banner."""
    play_id = str(req.get("id", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, st["open"]):  # the open tile was deleted: its working state stays, as the draft
        st["tiles"].pop("new", None)
        st.update(open="new", dirty=False)
        _put(st)
    st["tiles"] = {k: v for k, v in st["tiles"].items() if tile_exists(play_id, k)}
    edited = [k for k in st["tiles"] if k != "new"] + ([st["open"]] if st["dirty"] and st["open"] != "new" else [])
    names = {k: v for k, v in (st.get("names") or {}).items() if tile_exists(play_id, k)}
    return {"open": st["open"], "dirty": st["dirty"], "edited": edited, "side": st.get("side", "video"),
            "banner": banner_of(play_id, st["open"]), "names": names}


def rename(req: dict) -> dict:
    """A tile's own name (the toolbar title and its shelf label); empty = back to the generated one."""
    play_id, ref = str(req.get("id", "")), str(req.get("tile", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, ref):
        raise ValueError("This tile is gone.")
    name = " ".join(str(req.get("name") or "").split())[:80]
    names = dict(st.get("names") or {})
    if name:
        names[ref] = name
    else:
        names.pop(ref, None)
    st["names"] = names
    _put(st)
    return info({"id": play_id})


ACTIONS = {"new": new, "side": side, "picture_frame": picture_frame, "open": open_tile, "back": back, "info": info, "rename": rename}
