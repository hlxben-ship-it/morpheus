"""Image Playground (Glia seed image_playground): a Local playground of kind "image".

Same home as the video Playground: output/playground/<id>/playground.json (kind "image"), its input pictures in the folder,
its results in images/<stem>.png + <stem>.json (every setting the picture was made with, G10). The model list and the
settings each model takes come from image_models.json (image_models.py). Renders go to the WSL warm worker (server.py).
"""
import json
import random
import shutil
import time
from pathlib import Path

import image_models as im
import lora_library
import morpheus
import playground

ROOT = playground.ROOT
LORAS = 4  # LoRA slots per image playground (2x2 on the page)
INPUTS = 4
SEED_MAX = 2**31 - 1


def options() -> dict:
    ms = im.models()
    return {"models": ms, "lora_families": {m["id"]: m["family"] for m in ms}, "families": lora_library.FAMILIES}


def fresh_state(play_id: str) -> dict:
    m = im.models()[0]
    return {"id": play_id, "kind": "image", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": m["id"],
            "params": im.defaults(m), "size": m["sizes"][0], "prompt": m.get("prompt_default", ""),
            "negative": m.get("negative_default", ""), "seed": None, "count": 1, "inputs": [None] * INPUTS,
            "loras": default_loras(m)}


def default_loras(model) -> list:
    """The model's default LoRAs (Qwen-Edit: Lightning) when they are in the library, then empty slots."""
    out = []
    for lid in model.get("default_loras") or []:
        e = lora_library.get(lid)
        if e and e["family"] == model["family"]:
            out.append({"id": lid, "strength": 1.0})
    return (out + [{"id": "", "strength": 1.0}] * LORAS)[:LORAS]


def new(req: dict) -> dict:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    play_id, n = stamp, 1
    while (ROOT / play_id).exists():
        n += 1
        play_id = f"{stamp}_{n}"
    return load(playground.write(fresh_state(play_id))["id"])


def _read(play_id: str) -> dict:
    try:
        st = json.loads((playground.folder(play_id) / "playground.json").read_text(encoding="utf-8-sig"))
    except OSError:
        raise ValueError("This playground does not exist.")
    if st.get("kind") == "studio":  # Studio (studio.py): the image half lives under "image", the star is the playground's one
        half = st.get("image") or {k: v for k, v in fresh_state(play_id).items() if k not in ("id", "kind", "created")}
        return {**half, "id": play_id, "kind": "image", "studio": True, **({"cover": st["cover"]} if st.get("cover") else {})}
    if st.get("kind") != "image":
        raise ValueError("This is a video playground.")
    return st


def _write(st: dict) -> dict:
    """playground.write, except for a Studio: there the state goes back under "image" and the rest of the file stays."""
    path = playground.folder(st["id"]) / "playground.json"
    try:
        whole = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        whole = {}
    if whole.get("kind") != "studio":
        return playground.write(st)
    whole["image"] = {k: v for k, v in st.items() if k not in ("id", "kind", "cover", "updated", "created", "studio")}
    if st.get("cover"):
        whole["cover"] = st["cover"]
    elif str(whole.get("cover") or "").startswith("images/"):
        whole.pop("cover", None)
    playground.write(whole)
    return st


def clean_seed(v):
    if v in (None, ""):
        return None
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ValueError("The seed is a whole number (empty = random).")
    if not 0 <= v <= SEED_MAX:
        raise ValueError(f"Seed: 0 to {SEED_MAX}.")
    return v


def clean_loras(slots, family) -> list:
    known = {e["id"]: e for e in lora_library.entries()}
    out = []
    for s in (slots if isinstance(slots, list) else [])[:LORAS]:
        s = s if isinstance(s, dict) else {}
        e = known.get(str(s.get("id", "")))
        ok = e and e["family"] == family
        try:
            strength = float(s.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        out.append({"id": e["id"] if ok else "", "strength": strength,
                    **({"missing": True} if ok and not lora_library.file_path(e).is_file() else {})})
    return out + [{"id": "", "strength": 1.0}] * (LORAS - len(out))


def load(play_id: str) -> dict:
    st = _read(play_id)
    ms = im.models()
    m = next((x for x in ms if x["id"] == st.get("model")), ms[0])
    st["model"] = m["id"]
    st["params"] = im.resolve(m, st.get("params") or {})
    st["size"] = im.resolve_size(m, st.get("size"))
    d = playground.folder(play_id)
    ins = list(st.get("inputs") or [])[:INPUTS]
    st["inputs"] = [f if f and (d / f).is_file() else None for f in ins + [None] * (INPUTS - len(ins))]
    st["loras"] = clean_loras(st.get("loras"), m["family"])
    try:
        st["seed"] = clean_seed(st.get("seed"))
    except ValueError:
        st["seed"] = None
    c = st.get("count")
    st["count"] = c if isinstance(c, int) and 1 <= c <= m.get("max_count", 1) else 1
    st.setdefault("prompt", ""); st.setdefault("negative", "")
    return st


def save(req: dict) -> dict:
    st = load(str(req.get("id", "")))
    before = st["model"]
    if "model" in req:
        if not im.get(req["model"]):
            raise ValueError("Unknown model.")
        st["model"] = req["model"]
    m = im.get(st["model"])
    switched = st["model"] != before
    if "params" in req:
        st["params"] = im.resolve(m, req["params"], strict=True)
    elif switched:
        st["params"] = im.defaults(m)
    st["size"] = im.resolve_size(m, req["size"], strict=True) if "size" in req else im.resolve_size(m, st["size"])
    if switched and st.get("studio"):  # Studio: the prompt stays the user's; the model's defaults live in "defaults"
        if "defaults" not in req:
            st.pop("defaults", None)
        if "loras" not in req:
            st["loras"] = default_loras(m)
    elif switched:  # another model: its default prompts and LoRAs, unless the page sends its own
        old = im.get(before) or {}
        if st.get("prompt", "") in ("", old.get("prompt_default", "")) and "prompt" not in req:
            st["prompt"] = m.get("prompt_default", "")
        if st.get("negative", "") in ("", old.get("negative_default", "")) and "negative" not in req:
            st["negative"] = m.get("negative_default", "")
        if "loras" not in req:
            st["loras"] = default_loras(m)
    for k, cap in (("prompt", 8000), ("negative", 4000)):
        if k in req:
            st[k] = str(req[k])[:cap]
    if "defaults" in req:  # Studio: the tile's default prompt / negative (None = the model's, studio_models.json)
        import studio_models
        st["defaults"] = studio_models.clean_defaults(req["defaults"])
    if "seed" in req:
        st["seed"] = clean_seed(req["seed"])
    if "count" in req:
        c = req["count"]
        if not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= m.get("max_count", 1):
            raise ValueError(f"Count: 1 to {m.get('max_count', 1)}.")
        st["count"] = c
    if "loras" in req:
        loras = req["loras"]
        if not isinstance(loras, list) or len(loras) != LORAS:
            raise ValueError(f"{LORAS} LoRA slots expected.")
        for x in loras:
            x = x or {}
            e = lora_library.get(str(x.get("id", ""))) if x.get("id") else None
            if x.get("id") and not e:
                raise ValueError("This LoRA is not in the library.")
            if e and e["family"] != m["family"]:
                raise ValueError(f"{e['name']} is a {lora_library.FAMILIES[e['family']]} LoRA; {m['label']} takes {lora_library.FAMILIES[m['family']]}.")
            try:
                s = float(x.get("strength", 1.0))
            except (TypeError, ValueError):
                raise ValueError("LoRA strength must be a number.")
            if not 0 <= s <= 2:
                raise ValueError("LoRA strength: 0 to 2.")
        st["loras"] = [{"id": str((x or {}).get("id", "")), "strength": float((x or {}).get("strength", 1.0))} for x in loras]
    if "inputs" in req:  # only clearing / reordering stored files; new pictures go through input()
        ins = req["inputs"]
        if not isinstance(ins, list) or len(ins) != INPUTS or any(f is not None and f not in st["inputs"] for f in ins):
            raise ValueError("Unknown input pictures.")
        st["inputs"] = ins
    return load(_write(st)["id"])


def slot_of(req) -> int:
    s = req.get("slot")
    if not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < INPUTS:
        raise ValueError("Unknown input slot.")
    return s


def input_picture(req: dict) -> dict:
    """Upload/drop into an input slot (data URL), or data None empties it."""
    s = slot_of(req)
    st = load(str(req.get("id", "")))
    st["inputs"][s] = None if req.get("data") is None else playground.store_picture(st, f"in{s + 1}", req.get("data"))
    return load(_write(st)["id"])


def image_path(play_id: str, image: str) -> Path:
    """A shelf picture of this playground ("playground/<id>/images/<name>.png", as images() lists it)."""
    d = (playground.folder(play_id) / "images").resolve()
    p = (morpheus.OUTPUT / str(image)).resolve()
    if p.parent != d or p.suffix != ".png" or not p.is_file():
        raise ValueError("This picture is gone.")
    return p


def use_as_input(req: dict) -> dict:
    """"Use as input": a shelf picture is copied into an input slot (G7)."""
    play_id = str(req.get("id", ""))
    src = image_path(play_id, str(req.get("image", "")))
    s = slot_of(req)
    st = load(play_id)
    name = f"in{s + 1}_{time.time_ns() // 1_000_000}.png"
    shutil.copyfile(src, playground.folder(play_id) / name)
    st["inputs"][s] = name
    if (im.get(st["model"]) or {}).get("helper"):  # Muse Resharpen: the picture's own prompt, editable (grooming G3)
        try:
            meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8-sig"))
            st["prompt"] = str(meta.get("studio_prompt") if meta.get("studio_prompt") is not None else meta.get("prompt", ""))[:8000]
        except (OSError, ValueError):
            pass
    return load(_write(st)["id"])


def lora_entries(st, m) -> list:
    out = []
    for n, slot in enumerate(st["loras"], 1):
        if not slot["id"]:
            continue
        e = lora_library.get(slot["id"])
        if not e or not lora_library.file_path(e).is_file():
            raise ValueError("A LoRA file is missing: delete it in Settings and upload it again.")
        out.append({"slot": n, "id": e["id"], "name": e["name"], "strength": slot["strength"], "path": str(lora_library.file_path(e).resolve()),
                    "keywords": e.get("keywords", "")})
    return out


def snapshot(play_id: str) -> dict:
    """Everything a render uses, frozen when Render is pressed; also what each picture's sidecar keeps (G10)."""
    st = load(play_id)
    m = im.get(st["model"])
    d = playground.folder(play_id)
    inputs = [f for f in st["inputs"] if f]
    lo, hi = m["inputs"]["min"], m["inputs"]["max"]
    if len(inputs) < lo:
        raise ValueError(f"{m['label']} needs {lo} picture{'s' if lo > 1 else ''}: add {'it' if lo == 1 else 'them'} under Inputs.")
    inputs = inputs[:hi]
    extra, prompt, negative = {}, st["prompt"], st["negative"]
    if st.get("studio"):  # Studio: the default prompt / negative go in front; the sidecar keeps both parts
        import studio_models
        dflt = studio_models.defaults(m["id"], st.get("defaults"))  # not `d`: that is the folder the inputs are hashed from
        prompt, negative = studio_models.join(dflt["prompt"], st["prompt"]), studio_models.join(dflt["negative"], st["negative"])
        extra = {"studio_prompt": st["prompt"], "studio_negative": st["negative"], "defaults": st.get("defaults")}
    if m.get("helper") and not prompt.strip():  # Muse Resharpen without a prompt: the neutral default (grill #8)
        prompt = m.get("prompt_default", "")
    if m.get("helper"):  # the sidecar names the helper, its source picture and the persona whose face is swapped back in
        extra = {**extra, "helper": m["helper"], "source": inputs[0], "persona": inputs[1] if len(inputs) > 1 else None}
    if not prompt.strip() and m["family"] != "sdxl":
        raise ValueError("Write a prompt first.")
    seed = st["seed"] if st["seed"] is not None else random.randrange(1, SEED_MAX)
    import hashlib
    hashes = [hashlib.sha256((d / f).read_bytes()).hexdigest()[:16] for f in inputs]
    return {**extra, "model": {"id": m["id"], "label": m["label"], "family": m["family"]}, "params": st["params"], "size": st["size"],
            "prompt": prompt, "negative": negative if m.get("negative") else "", "seed": seed, "count": st["count"],
            "inputs": inputs, "input_hashes": hashes, "loras": lora_entries(st, m),
            "total_steps": st["params"].get("steps", 6) * st["count"] * (2 if st["params"].get("hires", 1) > 1 else 1) if m["family"] == "sdxl"
            else RESHARPEN_STEPS[st["params"].get("mode", "cheap")] * st["count"] if m.get("helper") == "resharpen" and st["params"].get("mode") != "cheap"
            else st["params"].get("steps", 6) * st["count"]}


# Resharpen stage 2 (seed muse_resharpen): Max = H3-Image edit of picture 1 (+ persona as <Picture 2>), Cheap = Muse img2img,
# Swap only = no worker job. The face swap with the persona runs afterwards in server.image_render (faceswap.py --image).
RESHARPEN_STEPS = {"max": 6, "swap": 1}
RESHARPEN_H3_SIZE = "2048x2048"  # H3 renders 4 MP anyway; with "KI" picture 1's proportions are kept
RESHARPEN_H3_PROMPT = ("<Picture 1> as a sharp, detailed photo. Keep <Picture 1>'s pose, expression, framing, hair, clothes, "
                       "lighting and background exactly the same.")


def resharpen_mode(snap: dict) -> str | None:
    """"max" / "cheap" / "swap" for a Resharpen snapshot, None for every other model."""
    return snap["params"].get("mode", "cheap") if snap.get("helper") == "resharpen" else None


def worker_job(play_id: str, snap: dict) -> dict | None:
    """The warm worker job for a snapshot (engine/wsl/warm_worker.py, kind "image"); None = no render (Resharpen "Swap only")."""
    m = im.get(snap["model"]["id"])
    d = playground.folder(play_id)
    w, h = im.wh(snap["size"])
    p = snap["params"]
    pics = [str((d / f).resolve()) for f in snap["inputs"]]
    mode = resharpen_mode(snap)
    if mode == "swap":
        return None
    if mode == "max":
        rw, rh = im.wh(RESHARPEN_H3_SIZE)
        prompt = RESHARPEN_H3_PROMPT  # picture 1 only: a whole start frame as <Picture 2> made H3 copy its scene (2026-11-19)
        if snap["prompt"].strip() and snap["prompt"] != m.get("prompt_default"):  # the source's own prompt describes the scene
            prompt += " Scene: " + snap["prompt"].strip()
        return {"kind": "image", "family": "h3_image", "settings": {
            "model_type": "minimax_h3_image_pruned", "prompt": prompt, "resolution": f"{rw}x{rh}", "num_inference_steps": 6,
            "guidance_scale": 1.0, "guidance_phases": 1, "sample_solver": "euler", "batch_size": snap["count"], "seed": snap["seed"],
            "image_mode": 1, "config": "distilled_4b", "image_refs": pics[:1], "video_prompt_type": "KI"}}
    loras = [{"path": l["path"], "strength": l["strength"]} for l in snap["loras"]]
    if m["family"] == "h3_image":
        s = {"model_type": "minimax_h3_image_pruned", "prompt": snap["prompt"], "resolution": f"{w}x{h}", "num_inference_steps": 6,
             "guidance_scale": 1.0, "guidance_phases": 1, "sample_solver": "euler", "batch_size": snap["count"], "seed": snap["seed"],
             "image_mode": 1, "config": "distilled_4b"}
        if pics:
            s.update(image_refs=pics, video_prompt_type=p.get("edit_mode", "KI"))
            if p.get("ref_size", "full") != "full":  # the worker maps it to a short edge (engine/wsl/ref_cap.py, seed h3_image_edit_speed)
                s["_h3_ref_size"] = p["ref_size"]
        if loras:
            s["activated_loras"] = [l["path"] for l in loras]
            s["loras_multipliers"] = " ".join(f"{l['strength']:g}" for l in loras)
        return {"kind": "image", "family": "h3_image", "settings": s}
    params = {"prompt": snap["prompt"], "negative": snap["negative"], "width": w, "height": h, "seed": snap["seed"], "loras": loras,
              **{k: v for k, v in p.items()}}
    if m["family"] == "sdxl":
        params["checkpoint"] = m["checkpoint"]
        params["init"] = pics[0] if pics else None
        if not pics:
            params["denoise"] = 1.0
    else:
        params["images"] = pics
        if m.get("checkpoint"):  # Krea 2: which finetune (GGUF file in /root/models/image/krea2)
            params["checkpoint"] = m["checkpoint"]
        if mode == "cheap":  # img2img from picture 1, no references (engine/wsl/image/krea2.py); the persona is only swapped
            params["init"], params["images"] = pics[0], []
            params.pop("mode", None)
    return {"kind": "image", "family": m["family"], "count": snap["count"], "params": params}


def add_images(play_id: str, files: list, seeds: list, snap: dict, seconds=None) -> list:
    """Results arrive: each picture moves to images/<stem>.png with <stem>.json (all settings + its own seed), json last."""
    out_dir = playground.folder(play_id) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for n, f in enumerate(files):
        seed = seeds[n] if n < len(seeds) else (snap["seed"] + n)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stem, k = f"{stamp}_{seed}", 1
        while (out_dir / f"{stem}.png").exists():
            k += 1
            stem = f"{stamp}_{seed}_{k}"
        dst = out_dir / f"{stem}.png"
        src = Path(f)
        if src.suffix.lower() == ".png":
            shutil.move(str(src), dst)
        else:  # WanGP may write jpg/webp: store as PNG for one format on the shelf
            from PIL import Image
            Image.open(src).save(dst); src.unlink(missing_ok=True)
        meta = {**snap, "seed": seed, "count": 1, "batch_of": len(files), "image": f"images/{dst.name}",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "render_seconds": seconds}
        tmp = out_dir / f"{stem}.tmp"
        tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(out_dir / f"{stem}.json")
        metas.append(meta)
    return metas


def images(play_id: str) -> list:
    """The shelf, newest first, the starred picture leading."""
    d = playground.folder(play_id)
    try:
        top = json.loads((d / "playground.json").read_text(encoding="utf-8-sig"))
        cover, hearts = top.get("cover"), set((top.get("hearts") or {}).values())
    except (OSError, ValueError):
        cover, hearts = None, set()
    out = []
    for js in (d / "images").glob("*.json"):
        png = js.with_suffix(".png")
        if not png.is_file():
            continue
        try:
            meta = json.loads(js.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        rel = png.relative_to(morpheus.OUTPUT).as_posix()
        m = meta.get("model") or {}
        try:  # the real size: H3 "keep picture 1's proportions" ignores the chosen one (reads only the PNG header)
            from PIL import Image
            with Image.open(png) as im_:
                size = f"{im_.width}x{im_.height}"
        except OSError:
            size = meta.get("size", "")
        out.append({"image": rel, "created": meta.get("created", ""), "prompt": meta.get("prompt", ""), "seed": meta.get("seed"),
                    "model": m.get("label", ""), "size": size, "render_seconds": meta.get("render_seconds"),
                    "cover": cover == f"images/{png.name}", "batch_of": meta.get("batch_of") or 1,
                    "heart": f"images/{png.name}" in hearts})
    out.sort(key=lambda c: c["created"], reverse=True)
    out.sort(key=lambda c: not c["cover"])
    return out


def delete_image(req: dict) -> dict:
    p = image_path(str(req.get("id", "")), str(req.get("image", "")))
    p.with_suffix(".json").unlink(missing_ok=True)
    p.unlink()
    return {"ok": True}


def star_image(req: dict) -> dict:
    """One starred picture per playground: the Local gallery's mini, leads the shelf (like star_clip)."""
    play_id = str(req.get("id", ""))
    st = _read(play_id)
    if req.get("starred"):
        st["cover"] = "images/" + image_path(play_id, str(req.get("image", ""))).name
    else:
        st.pop("cover", None)
    path = playground.folder(play_id) / "playground.json"
    if json.loads(path.read_text(encoding="utf-8-sig")).get("kind") == "studio":  # the star sits on the Studio's top level
        whole = json.loads(path.read_text(encoding="utf-8-sig"))
        if st.get("cover"):
            whole["cover"] = st["cover"]
        elif str(whole.get("cover") or "").startswith("images/"):
            whole.pop("cover", None)
        st = whole
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"ok": True, "cover": st.get("cover")}


def reuse(req: dict) -> dict:
    """"Reuse settings" (G10): model, params, size, prompts, LoRAs, seed, and the inputs that are still in the folder."""
    play_id = str(req.get("id", ""))
    p = image_path(play_id, str(req.get("image", "")))
    try:
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise ValueError("This picture has no settings to take.")
    mid = (meta.get("model") or {}).get("id")
    if not im.get(mid):
        raise ValueError("Its model is no longer in image_models.json.")
    by_slot = {l.get("slot"): l for l in meta.get("loras") or [] if lora_library.get(str(l.get("id", "")))}
    loras = [{"id": by_slot[n]["id"], "strength": by_slot[n]["strength"]} if n in by_slot else {"id": "", "strength": 1.0} for n in range(1, LORAS + 1)]
    m = im.get(mid)
    save({"id": play_id, "model": mid, "params": im.resolve(m, meta.get("params") or {}), "size": im.resolve_size(m, meta.get("size")),
          "prompt": meta.get("prompt", ""), "negative": meta.get("negative", ""), "seed": meta.get("seed"), "count": 1, "loras": loras})
    st = _read(play_id)
    d = playground.folder(play_id)
    ins = [f if f and (d / f).is_file() else None for f in (meta.get("inputs") or [])][:INPUTS]
    st["inputs"] = ins + [None] * (INPUTS - len(ins))
    return load(_write(st)["id"])


def heart_image(req: dict) -> dict:
    """The picked picture of one batch (Studio shelf shows a batch as one tile): {batch key: "images/<name>.png"} on
    the playground's top level. The key is the batch's first picture name (the page groups and names the batch);
    heart false clears it."""
    play_id, key = str(req.get("id", "")), str(req.get("batch", ""))[:200]
    if not key:
        raise ValueError("No batch.")
    path = playground.folder(play_id) / "playground.json"
    whole = json.loads(path.read_text(encoding="utf-8-sig"))
    hearts = whole.get("hearts") or {}
    if req.get("heart"):
        hearts[key] = "images/" + image_path(play_id, str(req.get("image", ""))).name
    else:
        hearts.pop(key, None)
    whole["hearts"] = hearts
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(whole, indent=1), encoding="utf-8")
    tmp.replace(path)
    return {"ok": True, "hearts": hearts}


ACTIONS = {"new": new, "save": save, "input": input_picture, "use_as_input": use_as_input, "delete_image": delete_image,
           "star_image": star_image, "heart_image": heart_image, "reuse": reuse}
