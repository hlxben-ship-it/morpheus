"""The Image Playground model schema (image_models.json): load, check, and resolve params.

Same rules as playground_models.py: strict on what the page sends (unknown key, out of range, not a choice -> error), lenient
on what was saved earlier (keep what fits, clamp a range, else the default)."""
import json
import re
from pathlib import Path

FILE = Path(__file__).parent / "image_models.json"
FAMILIES = ("sdxl", "qwen_edit", "h3_image")
ID = re.compile(r"[a-z0-9_]{1,40}")
SIZE = re.compile(r"(\d{3,4})x(\d{3,4})")


def _fail(where, msg):
    raise ValueError(f"image_models.json: {where}: {msg}")


def check(data) -> list:
    ms = data.get("models") if isinstance(data, dict) else None
    if not isinstance(ms, list) or not ms:
        _fail("models", "a non-empty list is needed")
    seen = set()
    for m in ms:
        mid = m.get("id", "?")
        if not ID.fullmatch(str(mid)) or mid in seen:
            _fail(mid, "id: lowercase letters, digits, _; unique")
        seen.add(mid)
        if m.get("family") not in FAMILIES:
            _fail(mid, f"family: one of {', '.join(FAMILIES)}")
        if m["family"] == "sdxl" and not m.get("checkpoint"):
            _fail(mid, "an sdxl model names its checkpoint")
        inp = m.get("inputs") or {}
        if not (isinstance(inp.get("min"), int) and isinstance(inp.get("max"), int) and 0 <= inp["min"] <= inp["max"] <= 4):
            _fail(mid, "inputs: {min, max} with 0 <= min <= max <= 4")
        if not m.get("sizes") or not all(SIZE.fullmatch(s) for s in m["sizes"]):
            _fail(mid, "sizes: a list of WxH")
        for k, p in (m.get("params") or {}).items():
            if "choices" in p:
                if p.get("default") not in p["choices"]:
                    _fail(f"{mid}.{k}", "default must be one of the choices")
            elif p.get("fixed"):
                if "default" not in p:
                    _fail(f"{mid}.{k}", "a fixed param needs a default")
            elif not (p.get("min", 0) <= p.get("default", -1) <= p.get("max", -1)):
                _fail(f"{mid}.{k}", "min <= default <= max")
    return ms


def models() -> list:
    try:
        return check(json.loads(FILE.read_text(encoding="utf-8")))
    except json.JSONDecodeError as e:
        raise ValueError(f"image_models.json is not valid JSON: {e}")


def get(model_id) -> dict:
    ms = models()
    return next((m for m in ms if m["id"] == model_id), None)


def _fits(p, v):
    if "choices" in p:
        return v in p["choices"]
    if p.get("fixed"):
        return v == p["default"]
    return isinstance(v, (int, float)) and not isinstance(v, bool) and p["min"] <= v <= p["max"]


def defaults(model) -> dict:
    return {k: p["default"] for k, p in (model.get("params") or {}).items()}


def resolve(model, given: dict, strict=False) -> dict:
    """Params for model from given. strict: refuse anything wrong (a save from the page); else repair (a stored state)."""
    spec = model.get("params") or {}
    out = defaults(model)
    for k, v in (given or {}).items():
        if k not in spec:
            if strict:
                raise ValueError(f"{model['label']} has no setting {k!r}.")
            continue
        p = spec[k]
        if not _fits(p, v):
            if strict:
                raise ValueError(f"{p.get('label', k)}: {v!r} is not allowed.")
            if "min" in p and isinstance(v, (int, float)) and not isinstance(v, bool):
                v = min(max(v, p["min"]), p["max"])
            else:
                continue
        if "min" in p and isinstance(p["default"], int) and not isinstance(p.get("step"), float) and isinstance(v, float) and v.is_integer():
            v = int(v)
        out[k] = v
    return out


def resolve_size(model, size, strict=False) -> str:
    if size in model["sizes"]:
        return size
    if strict:
        raise ValueError(f"{model['label']}: size one of {', '.join(model['sizes'])}.")
    return model["sizes"][0]


def wh(size: str) -> tuple:
    m = SIZE.fullmatch(size)
    return int(m.group(1)), int(m.group(2))
