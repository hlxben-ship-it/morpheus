"""Studio (Glia seed combined_playground): a Local playground of kind "studio" that does video and pictures on one page.

One folder output/playground/<id>/ like the other two kinds. playground.json holds the video half at the top level (so every
playground.py action works on it unchanged: render, upscale, Loop it, Cut, Remix, frames) and the image half under "image"
(image_playground.py reads and writes it there). "side" says which half the page shows; the model picker lists both
families and switching to a model of the other family switches the side.

Drafts (seed studio_drafts, replaces stage 2's edited copies): a rendered tile is read-only, its settings = its sidecar.
"drafts" = {"draft:<n>": {"settings", "from"}} are the editable tiles, any number. The editor (the working state above)
shows one tile: "open" names it ("draft:<n>" | "clips/<x>.mp4" | "images/<x>.png"); an open draft's slot is refreshed
from the working state when another tile opens. Run only from a draft; the draft stays open after it.
"""
import base64
import json
import re

import image_models as im
import image_playground
import playground
import playground_models as pm

SIDES = ("video", "image")
META = {"id", "kind", "created", "updated", "cover", "starred", "tiles", "open", "dirty", "drafts", "next_draft", "remix", "_clean", "pictures", "missing", "names"}
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
    if "drafts" not in st:
        _migrate(st)
    return st


def _migrate(st: dict) -> None:
    """A Studio from before seed studio_drafts: every working copy becomes its own draft, nothing is lost (Q2). The
    working state of the open tile is a draft when it was the loose one or had edits; a clean render stays open read-only."""
    tiles, dirty, cur = st.pop("tiles", None) or {}, st.pop("dirty", False), st.get("open", "new")
    names = st.get("names") or {}
    st["drafts"], st["next_draft"] = {}, 0
    if "new" in tiles:
        ref = _add_draft(st, tiles.pop("new"))
        if names.get("new"):
            names[ref] = names["new"]
    open_ref = cur
    if cur == "new" or dirty:
        open_ref = _add_draft(st, settings_of(st), None if cur == "new" else cur)
        if cur == "new" and names.get("new"):
            names[open_ref] = names["new"]
    for ref, s in tiles.items():
        _add_draft(st, s, ref)
    names.pop("new", None)
    if not st["drafts"]:
        _add_draft(st, fresh_settings(st["id"]))
    st["open"] = open_ref
    _put(st)


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
    st.update(kind="studio", side="video", image=image_half(st["id"]), open="draft:1", drafts={"draft:1": {"settings": {}, "from": None}}, next_draft=1)
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


# ---- drafts (seed studio_drafts): rendered tiles are read-only, drafts are the only editable tiles ----

def is_draft(ref) -> bool:
    return isinstance(ref, str) and ref.startswith("draft:")


def tile_exists(play_id: str, ref: str, st: dict = None) -> bool:
    if is_draft(ref):
        return ref in ((st if st is not None else _raw(play_id)).get("drafts") or {})
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
    st.update(side=to, open=ref)
    _put(st)


def _stash(st: dict) -> None:
    """The open draft's working state goes back into its slot (a rendered tile has nothing to keep)."""
    if is_draft(st["open"]) and st["open"] in st["drafts"]:
        st["drafts"][st["open"]]["settings"] = settings_of(st)


def _add_draft(st: dict, settings: dict, src=None) -> str:
    st["next_draft"] = int(st.get("next_draft") or 0) + 1
    ref = f"draft:{st['next_draft']}"
    st["drafts"][ref] = {"settings": settings, "from": src}
    return ref


def _show_draft(st: dict, ref: str) -> None:
    _set_settings(st, dict(st["drafts"][ref]["settings"]) or fresh_settings(st["id"]))
    st["open"] = ref


def open_tile(req: dict) -> dict:
    """Click a tile: a draft opens editable with its own settings, a render opens read-only with its rendered state."""
    play_id, ref = str(req.get("id", "")), str(req.get("tile", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, ref, st):
        raise ValueError("This tile is gone.")
    if st["open"] == ref:
        return info({"id": play_id})
    _stash(st)
    if is_draft(ref):
        _show_draft(st, ref)
        _put(st)
    else:
        st["open"] = ref
        _put(st)
        _load_rendered(play_id, ref)
    return info({"id": play_id})


def new_draft(req: dict) -> dict:
    """"+ New" (empty draft) or "Use these settings on new" (tile = a rendered tile: the draft starts with its settings)."""
    play_id, src = str(req.get("id", "")), req.get("tile")
    st = _raw(play_id)
    if src is not None:
        src = str(src)
        if is_draft(src) or not tile_exists(play_id, src, st):
            raise ValueError("Only a rendered tile can be used on a new draft.")
    _stash(st)
    _put(st)
    if src is None:
        st = _raw(play_id)
        settings = fresh_settings(play_id)
    else:
        _load_rendered(play_id, src)
        st = _raw(play_id)
        settings = settings_of(st)
    ref = _add_draft(st, settings, src)
    _show_draft(st, ref)
    _put(st)
    return info({"id": play_id})


def delete_draft(req: dict) -> dict:
    """Trash on a draft: gone at once (Q4). Was it open, the newest other draft opens, else a fresh empty one."""
    play_id, ref = str(req.get("id", "")), str(req.get("tile", ""))
    st = _raw(play_id)
    if not is_draft(ref):
        raise ValueError("Only a draft can be deleted here.")
    if ref not in st["drafts"]:
        raise ValueError("This draft is gone.")
    del st["drafts"][ref]
    (st.get("names") or {}).pop(ref, None)
    if st["open"] == ref:
        rest = list(st["drafts"])
        _show_draft(st, rest[-1] if rest else _add_draft(st, fresh_settings(play_id)))
    _put(st)
    return info({"id": play_id})


def run_source(play_id: str):
    """The tile a render starts from (None when this is no Studio). A rendered tile is read-only: no render from it."""
    try:
        st = _raw(play_id)
    except ValueError:
        return None
    if not is_draft(st["open"]):
        raise ValueError("A rendered tile is read-only. Use its settings on a new draft to render again.")
    return st["open"]


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


def _label(st: dict, ref: str) -> str:
    names = st.get("names") or {}
    if names.get(ref):
        return names[ref]
    src = st["drafts"][ref].get("from")
    return f"Draft · from {names.get(src) or src.split('/')[-1]}" if src else "Draft"


def info(req: dict) -> dict:
    """What the page needs: the open tile, whether it is read-only, the drafts (newest first), the banner, the names."""
    play_id = str(req.get("id", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, st["open"], st):  # the open tile was deleted: the newest draft opens
        rest = list(st["drafts"])
        _show_draft(st, rest[-1] if rest else _add_draft(st, fresh_settings(play_id)))
        _put(st)
    names = {k: v for k, v in (st.get("names") or {}).items() if tile_exists(play_id, k, st)}
    drafts = [{"ref": k, "label": _label(st, k), "from": d.get("from")} for k, d in reversed(list(st["drafts"].items()))]
    return {"open": st["open"], "readonly": not is_draft(st["open"]), "drafts": drafts, "side": st.get("side", "video"),
            "banner": banner_of(play_id, st["open"]), "names": names}


def rename(req: dict) -> dict:
    """A tile's own name (the toolbar title and its shelf label); empty = back to the generated one."""
    play_id, ref = str(req.get("id", "")), str(req.get("tile", ""))
    st = _raw(play_id)
    if not tile_exists(play_id, ref, st):
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


def resharpen_last(req: dict) -> dict:
    """Clip viewer "Resharpen last frame" (seed muse_resharpen stage 2): the clip's last frame goes into slot 1 of the image
    half's Resharpen helper, the clip's start frame (if it had one) into slot 2 as the persona, and the page switches to the
    image half. The prompt stays empty: a video prompt is no picture description (the helper falls back to its neutral one)."""
    play_id, clip_play, video = str(req.get("id", "")), str(req.get("clip_playground") or req.get("id", "")), str(req.get("video", ""))
    data = playground.last_frame({"id": play_id, "clip_playground": clip_play, "video": video, "data_only": True})["data"]
    _, meta = playground.clip_meta(clip_play, video)
    side({"id": play_id, "side": "image", "model": RESHARPEN})
    image_playground.save({"id": play_id, "prompt": "", "params": {**image_playground.load(play_id)["params"], "mode": "max"}})
    image_playground.input_picture({"id": play_id, "slot": 0, "data": data})
    start = meta.get("start_frame")
    persona = playground.folder(clip_play) / start if start else None
    if persona is not None and persona.is_file() and persona.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        mime = {".jpg": "jpeg", ".jpeg": "jpeg"}.get(persona.suffix.lower(), persona.suffix.lower()[1:])
        image_playground.input_picture({"id": play_id, "slot": 1, "data": f"data:image/{mime};base64," + base64.b64encode(persona.read_bytes()).decode()})
    else:
        image_playground.input_picture({"id": play_id, "slot": 1, "data": None})
    return playground.load(play_id)


RESHARPEN = "muse_resharpen"

ACTIONS = {"new": new, "side": side, "resharpen_last": resharpen_last, "picture_frame": picture_frame, "open": open_tile, "new_draft": new_draft, "delete_draft": delete_draft, "info": info, "rename": rename}
