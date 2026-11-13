"""The model schema of the Dawnforge Playground: what each model's pipeline is made of and what can be set.

The data is playground_models.json (edit it, no restart: it is read on every request). The page draws its
"Pipeline" block from it and the server validates against it, so a new model or variant is one JSON entry.

    model   id, label, note (optional hint), resolutions [..], stages [..], options [..]
    resolution  id, label, target ("720p", "1080p", "2k"), render (optional): the resolution the model really
            generates at when an upscale to the target follows. The first entry is the default. Use the same ids
            across models ("1080p", "1080p_from_720p") so a model switch keeps the choice where it exists.
    stage   id, label, kind "sample" | "upscale", hint (optional small line under the label)
            sample:  size "reduced" | "target" (informational, used when there is no hint), steps = ONE of
                       {"min": 8, "max": 30, "default": 8}   a slider
                       {"choices": [4, 6, 8], "default": 4}  a segmented choice
                       {"default": 3, "fixed": true}         shown, not editable
            upscale: factor (e.g. 2), to "target" (informational); nothing to set. Leave an upscale that always
                     happens out of the file and say so in the next stage's hint instead.
    option  id, label, type "choice", choices [{id, label}], default, note (optional)

The settings a user can change are a flat dict ("params"): "<stage id>.steps" per sample stage and "<option id>"
per option, e.g. {"main.steps": 8, "refine.steps": 3, "upscale_mode": "default"}. The render seed reads the
same dict. Keys starting with "_" in the file are comments.
"""
import json
import re
from pathlib import Path

FILE = Path(__file__).with_name("playground_models.json")
KINDS = ("sample", "upscale")
NAME = re.compile(r"[a-z0-9_]{1,40}")


def models() -> list:
    try:
        data = json.loads(FILE.read_text(encoding="utf-8-sig"))
    except OSError:
        raise ValueError(f"{FILE.name} is missing.")
    except ValueError as e:
        raise ValueError(f"{FILE.name} is not valid JSON: {e}")
    check(data)
    return data["models"]


def check(data) -> None:
    def bad(msg):
        raise ValueError(f"{FILE.name}: {msg}")

    def whole(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    ms = data.get("models") if isinstance(data, dict) else None
    if not isinstance(ms, list) or not ms:
        bad("'models' must be a non-empty list.")
    ids = set()
    for m in ms:
        mid = m.get("id") if isinstance(m, dict) else None
        if not isinstance(mid, str) or not NAME.fullmatch(mid):
            bad(f"model id {mid!r}: lowercase letters, digits and _ only.")
        if mid in ids:
            bad(f"model id {mid!r} twice.")
        ids.add(mid)
        if not str(m.get("label", "")).strip():
            bad(f"{mid}: a label is needed.")
        rs = m.get("resolutions")
        if not isinstance(rs, list) or not rs:
            bad(f"{mid}: 'resolutions' must be a non-empty list.")
        rids = set()
        for r in rs:
            rid = r.get("id") if isinstance(r, dict) else None
            if not isinstance(rid, str) or not NAME.fullmatch(rid) or rid in rids:
                bad(f"{mid}: resolution id {rid!r} must be lowercase letters, digits, _ and unique.")
            rids.add(rid)
            if not str(r.get("label", "")).strip() or not str(r.get("target", "")).strip():
                bad(f"{mid}.{rid}: a label and a target are needed.")
            if "render" in r and (not isinstance(r["render"], str) or r["render"] == r["target"]):
                bad(f"{mid}.{rid}: 'render' is the lower resolution it generates at and must differ from the target.")
        stages = m.get("stages")
        if not isinstance(stages, list) or not stages:
            bad(f"{mid}: 'stages' must be a non-empty list.")
        keys = set()
        for st in stages:
            sid = st.get("id") if isinstance(st, dict) else None
            if not isinstance(sid, str) or not NAME.fullmatch(sid) or sid in keys:
                bad(f"{mid}: stage id {sid!r} must be lowercase letters, digits, _ and unique.")
            keys.add(sid)
            if st.get("kind") not in KINDS:
                bad(f"{mid}.{sid}: kind must be one of {', '.join(KINDS)}.")
            if not str(st.get("label", "")).strip():
                bad(f"{mid}.{sid}: a label is needed.")
            if "hint" in st and not isinstance(st["hint"], str):
                bad(f"{mid}.{sid}: the hint must be text.")
            if st["kind"] == "upscale":
                if "factor" in st and not (isinstance(st["factor"], (int, float)) and st["factor"] > 1):
                    bad(f"{mid}.{sid}: factor must be a number above 1.")
                continue
            s = st.get("steps")
            if not isinstance(s, dict) or not whole(s.get("default")):
                bad(f"{mid}.{sid}: steps needs a whole 'default'.")
            if s.get("fixed"):
                continue
            if "choices" in s:
                if not isinstance(s["choices"], list) or not all(whole(c) for c in s["choices"]) or s["default"] not in s["choices"]:
                    bad(f"{mid}.{sid}: steps 'choices' must be whole numbers that include the default.")
            elif not (whole(s.get("min")) and whole(s.get("max")) and s["min"] <= s["default"] <= s["max"]):
                bad(f"{mid}.{sid}: steps needs min <= default <= max.")
        for o in m.get("options", []):
            oid = o.get("id") if isinstance(o, dict) else None
            if not isinstance(oid, str) or not NAME.fullmatch(oid) or oid in keys:
                bad(f"{mid}: option id {oid!r} must be lowercase letters, digits, _ and unique.")
            keys.add(oid)
            cs = o.get("choices")
            if o.get("type") != "choice" or not isinstance(cs, list) or len(cs) < 2 or not all(isinstance(c, dict) and c.get("id") and c.get("label") for c in cs):
                bad(f"{mid}.{oid}: a 'choice' option needs at least two {{id, label}} choices.")
            if o.get("default") not in [c["id"] for c in cs]:
                bad(f"{mid}.{oid}: the default must be one of the choices.")


def specs(model: dict) -> dict:
    """The settable (and fixed) values of a model, keyed like the params."""
    out = {}
    for st in model["stages"]:
        if st["kind"] != "sample":
            continue
        s, key = st["steps"], f"{st['id']}.steps"
        if s.get("fixed"):
            out[key] = {"type": "fixed", "default": s["default"]}
        elif "choices" in s:
            out[key] = {"type": "choices", "choices": s["choices"], "default": s["default"]}
        else:
            out[key] = {"type": "range", "min": s["min"], "max": s["max"], "default": s["default"]}
    for o in model.get("options", []):
        out[o["id"]] = {"type": "choices", "choices": [c["id"] for c in o["choices"]], "default": o["default"]}
    return out


def resolve(model: dict, given, strict: bool = False) -> dict:
    """A complete params dict for this model. strict (what the user sent): anything unknown, fixed or out of range
    is an error. Not strict (what was saved earlier, maybe for another model): keep what fits, else the default,
    a number outside the range is clamped."""
    sp = specs(model)
    if strict and not isinstance(given, dict):
        raise ValueError("The settings must be an object.")
    given = given if isinstance(given, dict) else {}
    if strict:
        for key in given:
            if key not in sp:
                raise ValueError(f"'{key}' is not a setting of {model['label']}.")
            if sp[key]["type"] == "fixed" and given[key] != sp[key]["default"]:  # the page echoes fixed values back: fine unless changed
                raise ValueError(f"'{key}' is fixed for {model['label']}.")
    out = {}
    for key, s in sp.items():
        v = given.get(key, s["default"])
        if s["type"] == "fixed":
            v = s["default"]
        elif s["type"] == "range":
            whole = isinstance(v, int) and not isinstance(v, bool)
            if whole and s["min"] <= v <= s["max"]:
                pass
            elif strict:
                raise ValueError(f"{key}: {s['min']} to {s['max']}.")
            else:
                v = min(max(v, s["min"]), s["max"]) if whole else s["default"]
        elif v not in s["choices"] or isinstance(v, bool):
            if strict:
                raise ValueError(f"{key}: one of {', '.join(str(c) for c in s['choices'])}.")
            v = s["default"]
        out[key] = v
    return out


def resolve_resolution(model: dict, value, strict: bool = False) -> str:
    """A resolution id this model offers. Strict (what the user sent): anything else is an error. Not strict (what was
    saved earlier, maybe for another model): kept if the model has it, else the model's first (default) resolution."""
    ids = [r["id"] for r in model["resolutions"]]
    if value in ids:
        return value
    if strict:
        raise ValueError(f"Resolution for {model['label']}: one of {', '.join(ids)}.")
    return ids[0]


def total_steps(model: dict, params: dict) -> int:
    """All sampler steps of a render, for the progress bar's total."""
    return sum(params[f"{st['id']}.steps"] for st in model["stages"] if st["kind"] == "sample")
