"""Studio editor layout per model (studio_models.json, seed combined_playground stage 3).

Read on every call (edit the file, reload the page). What a model's boxes show: prompt (negative yes/no, default prompt /
negative), references (main image, slots, toggles), LoRA slots. Steps, sizes and params stay in playground_models.json /
image_models.json, which the render plumbing validates against.
"""
import json
from pathlib import Path

import image_models as im
import playground_models as pm

FILE = Path(__file__).with_name("studio_models.json")
TOGGLES = {"first", "last"}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def _file() -> dict:
    try:
        data = json.loads(FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"studio_models.json: {e}")
    if not isinstance(data.get("defaults"), dict) or not isinstance(data.get("models"), dict):
        raise ValueError("studio_models.json: needs 'defaults' and 'models'.")
    return data


def check(cfg: dict, mid: str) -> dict:
    r, p, lo = cfg.get("references") or {}, cfg.get("prompt") or {}, cfg.get("loras") or {}
    if not isinstance(r.get("slots", 0), int) or not 0 <= r.get("slots", 0) <= 8:
        raise ValueError(f"studio_models.json: {mid}.references.slots must be 0 to 8.")
    if not set(r.get("toggles") or []) <= TOGGLES:
        raise ValueError(f"studio_models.json: {mid}.references.toggles: first and/or last.")
    if not isinstance(lo.get("slots", 0), int) or not 0 <= lo.get("slots", 0) <= 8:
        raise ValueError(f"studio_models.json: {mid}.loras.slots must be 0 to 8.")
    for k in ("default", "default_negative"):
        if not isinstance(p.get(k, ""), str):
            raise ValueError(f"studio_models.json: {mid}.prompt.{k} must be text.")
    return cfg


def models() -> dict:
    """Every Studio model (video + image), id -> {id, label, side, prompt, references, loras}."""
    data = _file()
    out = {}
    for side, ms in (("video", pm.models()), ("image", im.models())):
        for m in ms:
            own = data["models"].get(m["id"]) or {}
            cfg = _merge(data["defaults"][side], own)
            if side == "image":  # an image model's own defaults unless the Studio file sets them
                prompt = cfg["prompt"]
                own_p = own.get("prompt") or {}
                if "default" not in own_p:
                    prompt["default"] = (m.get("prompt_default") or "").strip().rstrip(",").strip()
                if "default_negative" not in own_p:
                    prompt["default_negative"] = (m.get("negative_default") or "").strip().rstrip(",").strip()
                if "negative" not in own_p:
                    prompt["negative"] = bool(m.get("negative"))
            out[m["id"]] = check({"id": m["id"], "label": m["label"], "side": side, **cfg}, m["id"])
    return out


def get(model_id: str) -> dict | None:
    return models().get(model_id)


def defaults(model_id: str, stored) -> dict:
    """The default prompt / negative a render uses: what the tile stored (None = the model's config)."""
    cfg = get(model_id) or {"prompt": {}}
    stored = stored if isinstance(stored, dict) else {}
    p = cfg["prompt"]
    dp = stored.get("prompt") if stored.get("prompt") is not None else p.get("default", "")
    dn = stored.get("negative") if stored.get("negative") is not None else p.get("default_negative", "")
    return {"prompt": dp, "negative": dn if p.get("negative") else ""}


def join(default: str, text: str) -> str:
    """The default in front of the prompt: "default, prompt" (either may be empty)."""
    parts = [x.strip().rstrip(",").strip() for x in (default or "", text or "")]
    return ", ".join(x for x in parts if x)


def clean_defaults(value) -> dict:
    """What a save may send as "defaults": {prompt, negative}, each text or None (None = the model's config)."""
    if not isinstance(value, dict) or set(value) - {"prompt", "negative"}:
        raise ValueError("Defaults: prompt and negative only.")
    out = {}
    for k in ("prompt", "negative"):
        v = value.get(k)
        if v is not None and not isinstance(v, str):
            raise ValueError("Defaults are text.")
        out[k] = None if v is None else v[:4000]
    return out
