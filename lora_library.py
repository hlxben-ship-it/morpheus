"""Video LoRA library: one folder per model family under models/loras/video/, one index (library.json) beside them.

LoRAs are downloaded by hand and uploaded on the Settings tab (raw bytes through /api/upload, then add() with the
upload id). The family is read from the safetensors header (detect_family); when the header does not say, the user
picks it. The Playground's two LoRA slots hold library ids, not paths (Glia seed video_lora_library).
WSL reads the files over /mnt/c: LoRAs are small enough that ext4 is not needed (unlike the checkpoints).
"""
import base64
import hashlib
import json
import re
import shutil
import struct
import tempfile
import threading
import time
from pathlib import Path

import morpheus

DIR = morpheus.ROOT / "models" / "loras" / "video"
UPLOADS = Path(tempfile.gettempdir()) / "morpheus-uploads"  # where server.py's /api/upload stores raw bytes
FAMILIES = {"ltx2": "LTX 2.3", "minimax_h3": "MiniMax H3",
            # Image Playground (Glia seed image_playground G6): one library, a family per image model kind
            "sdxl": "SDXL", "qwen_edit": "Qwen-Image-Edit", "h3_image": "H3-Image", "krea2": "Krea 2"}
HEADER_MAX = 64 << 20
PICTURES = {"data:image/png;base64": "png", "data:image/jpeg;base64": "jpg", "data:image/webp;base64": "webp"}
LOCK = threading.Lock()

# Header fingerprints, pinned from real files (2026-11-15): an LTX 2.3 LoRA (DR34ML4Y, the distill LoRA) names
# diffusion_model.transformer_blocks.N.{attn1,attn2,audio_attn1,...}.to_{q,k,v,gate_logits}; the H3 HyperFlow
# checkpoint names blocks.N.{attn.q_proj,mlp.fc1,adaln_proj}, so an H3 LoRA carries those module names.
LTX = re.compile(r"(^|[._])(audio_attn\d|to_gate_logits|audio_to_video_attn|video_to_audio_attn)([._]|$)")
H3 = re.compile(r"(^|[._])blocks[._]\d+[._](attn[._](q|k|v|out)_proj|mlp[._]fc[12]|adaln_proj)([._]|$)")
# kohya SDXL LoRAs name lora_unet_* / lora_te1_* / lora_te2_*; diffusers-style ones the UNet's down/up/mid blocks
SDXL = re.compile(r"^lora_unet_(input_blocks|output_blocks|middle_block|down_blocks|up_blocks|mid_block)_|^lora_te[12]_text_model|(^|[._])(down_blocks|up_blocks|mid_block)[._]\d")
# Qwen-Image(-Edit) DiT: transformer_blocks.N.{img_mod,txt_mod,img_mlp,txt_mlp,attn.add_q_proj,...}
# Krea 2 (ComfyUI / ai-toolkit names): blocks.N.attn.wq|wk|wv|wo|gate, txtfusion.*
KREA2 = re.compile(r"(^|[._])(txtfusion[._]|blocks[._]\d+[._]attn[._](w[qkvo]|gate)[._])")
QWEN = re.compile(r"(^|[._])transformer_blocks[._]\d+[._](img_mod|txt_mod|img_mlp|txt_mlp|attn[._]add_[qkv]_proj)")


def index_path() -> Path:
    return DIR / "library.json"


def entries() -> list:
    try:
        data = json.loads(index_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data.get("loras", []) if isinstance(data, dict) else []


def write_entries(items: list) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps({"format": 1, "loras": items}, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(index_path())


def get(lora_id: str) -> dict | None:
    return next((e for e in entries() if e["id"] == lora_id), None)


def file_path(entry: dict) -> Path:
    if entry.get("linked"):  # a file another part of Morpheus owns (Yumeko's character LoRAs): used in place, never moved or deleted
        return Path(entry["linked"])
    return DIR / entry["family"] / entry["file"]


def link(key: str, family: str, path: Path, props: dict, picture: Path | None = None) -> dict:
    """Add or refresh an entry that points at a file elsewhere (key = its owner's id, e.g. "yumeko-<job>").
    Name, keywords and strengths come from the owner on every call; the picture is copied once."""
    with LOCK:
        items = entries()
        e = next((x for x in items if x.get("link_key") == key), None)
        if not e:
            e = {"id": slug(props["name"], {x["id"] for x in items}), "family": family, "file": "", "picture": None,
                 "link_key": key, "family_source": "owner", "added": time.strftime("%Y-%m-%dT%H:%M:%S"), "url": "", "info": ""}
            items.append(e)
        e.update({k: v for k, v in props.items() if k in ("name", "keywords", "info", "strength_min", "strength_max")})
        if props.get("prompt") and not e.get("prompt"):  # a suggestion once; what the user writes in Settings stays
            e["prompt"] = props["prompt"]
        e["family"], e["linked"] = family, str(path)
        if picture and picture.is_file() and not e.get("picture"):
            (DIR / family).mkdir(parents=True, exist_ok=True)
            name = f"{e['id']}_{int(time.time())}{picture.suffix.lower()}"
            shutil.copyfile(picture, DIR / family / name)
            e["picture"] = name
        write_entries(items)
    return e


def unlink_key(key: str) -> None:
    """The owner dropped its file: the entry goes (the file was never ours)."""
    e = next((x for x in entries() if x.get("link_key") == key), None)
    if e:
        delete({"id": e["id"], "confirm": True})


def family_of_model(model_id: str) -> str | None:
    """The LoRA family a Playground model takes: by the WanGP model type it renders with."""
    from engine import jobs
    kind = jobs.MODELS.get(model_id, "")
    return next((f for f in FAMILIES if kind.startswith(f)), None)


def read_header(path: Path) -> dict:
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
    if not isinstance(header, dict):
        raise ValueError("Not a safetensors file.")
    return header


def detect_family(keys) -> str | None:
    """ltx2 / minimax_h3 / sdxl / qwen_edit / krea2 from the tensor names, None when they say none or several. An H3-Image LoRA
    looks like an H3 video one (same block names): the user picks h3_image by hand."""
    hits = {f for f, rx in (("ltx2", LTX), ("minimax_h3", H3), ("sdxl", SDXL), ("qwen_edit", QWEN), ("krea2", KREA2)) if any(rx.search(k) for k in keys)}
    return hits.pop() if len(hits) == 1 else None


def upload_file(upload_id: str) -> Path:
    path = UPLOADS / str(upload_id).strip("./\\")
    if not upload_id or not path.is_file() or path.parent != UPLOADS:
        raise ValueError("The upload is gone. Pick the file again.")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(req: dict) -> dict:
    """What the Settings form needs after the upload: the detected family and the header summary."""
    import playground
    path = upload_file(str(req.get("upload_id", "")))
    header = read_header(path)
    keys = [k for k in header if k != "__metadata__"]
    family = detect_family(keys)
    try:
        summary = playground.inspect({"path": str(path)})["summary"]
    except ValueError:
        summary = []
    return {"family": family, "bytes": path.stat().st_size, "tensors": len(keys), "summary": summary}


def clean_props(req: dict, old: dict | None = None) -> dict:
    old = old or {}
    name = str(req.get("name", old.get("name", ""))).strip()[:120]
    if not name:
        raise ValueError("Give the LoRA a name.")
    try:
        lo = float(req.get("strength_min", old.get("strength_min", 0.5)))
        hi = float(req.get("strength_max", old.get("strength_max", 1.0)))
    except (TypeError, ValueError):
        raise ValueError("Strength min and max must be numbers.")
    if not (0 <= lo <= hi <= 2):
        raise ValueError("Strength range: 0 ≤ min ≤ max ≤ 2.")
    url = str(req.get("url", old.get("url", ""))).strip()[:500]
    if url and not re.match(r"https?://", url):
        raise ValueError("The URL should start with http:// or https://.")
    return {"name": name, "url": url, "keywords": str(req.get("keywords", old.get("keywords", ""))).strip()[:300],
            "prompt": str(req.get("prompt", old.get("prompt", ""))).strip()[:4000],
            "info": str(req.get("info", old.get("info", ""))).strip()[:20000],
            "strength_min": round(lo, 2), "strength_max": round(hi, 2)}


def slug(name: str, taken: set) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "lora"
    out, n = base, 1
    while out in taken:
        n += 1
        out = f"{base}_{n}"
    return out


def store_picture(entry: dict, data_url) -> None:
    """A data URL replaces the picture, "" removes it, None leaves it."""
    if data_url is None:
        return
    folder = DIR / entry["family"]
    if entry.get("picture"):
        (folder / entry["picture"]).unlink(missing_ok=True)
        entry["picture"] = None
    if data_url:
        head, _, body = str(data_url).partition(",")
        ext = PICTURES.get(head)
        if not ext or not body:
            raise ValueError("A PNG, JPEG or WebP picture, please.")
        name = f"{entry['id']}_{int(time.time())}.{ext}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(base64.b64decode(body))
        entry["picture"] = name


def add(req: dict) -> dict:
    path = upload_file(str(req.get("upload_id", "")))
    header = read_header(path)
    detected = detect_family([k for k in header if k != "__metadata__"])
    family = str(req.get("family") or detected or "")
    if family not in FAMILIES:
        raise ValueError("The file does not say which model it is for: pick the family.")
    if detected and family != detected and not req.get("force"):
        raise ValueError(f"The header says {FAMILIES[detected]}, not {FAMILIES[family]}. Tick 'use anyway' to keep your pick.")
    props = clean_props(req)
    digest = sha256(path)
    with LOCK:
        items = entries()
        dup = next((e for e in items if e.get("sha256") == digest), None)
        if dup:
            raise ValueError(f"This file is already in the library as “{dup['name']}”.")
        entry = {"id": slug(props["name"], {e["id"] for e in items}), "family": family, **props, "file": "",
                 "original_name": str(req.get("filename", ""))[:200], "bytes": path.stat().st_size, "sha256": digest,
                 "family_source": "header" if detected == family else "user", "picture": None,
                 "added": time.strftime("%Y-%m-%dT%H:%M:%S")}
        entry["file"] = entry["id"] + ".safetensors"
        (DIR / family).mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(file_path(entry)))
        store_picture(entry, req.get("picture"))
        items.append(entry)
        write_entries(items)
    return entry


def update(req: dict) -> dict:
    with LOCK:
        items = entries()
        entry = next((e for e in items if e["id"] == req.get("id")), None)
        if not entry:
            raise ValueError("This LoRA is not in the library.")
        entry.update(clean_props(req, entry))
        store_picture(entry, req.get("picture"))
        write_entries(items)
    return entry


def delete(req: dict) -> dict:
    """Without confirm: who uses it. With confirm: the file, picture and entry go, and the slots using it are emptied."""
    import playground
    lora_id = str(req.get("id", ""))
    entry = get(lora_id)
    if not entry:
        raise ValueError("This LoRA is not in the library.")
    used = playground.lora_users(lora_id)
    if not req.get("confirm"):
        return {"id": lora_id, "used_by": used, "confirm": True}
    with LOCK:
        items = [e for e in entries() if e["id"] != lora_id]
        if not entry.get("linked"):
            file_path(entry).unlink(missing_ok=True)
        if entry.get("picture"):
            (DIR / entry["family"] / entry["picture"]).unlink(missing_ok=True)
        write_entries(items)
    playground.clear_lora(lora_id)
    return {"ok": True, "cleared": used}


def listing() -> dict:
    return {"families": FAMILIES, "loras": [{**e, "path": str(file_path(e)), "missing": not file_path(e).is_file()} for e in entries()]}


def picture(lora_id: str) -> Path | None:
    entry = get(lora_id)
    if not entry or not entry.get("picture"):
        return None
    path = DIR / entry["family"] / entry["picture"]
    return path if path.is_file() else None


ACTIONS = {"probe": probe, "add": add, "update": update, "delete": delete}
