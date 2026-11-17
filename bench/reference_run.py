"""Reference shot (Glia seed reference_prompt): render the fixed reference prompt through every image / video pipe.

    python bench/reference_run.py                      every installed image model
    python bench/reference_run.py --video              every video model (start frame + motion prompt, 5 s)
    python bench/reference_run.py --model z-image-turbo --model h3_image    just these (image or video ids)
    python bench/reference_run.py --grid output/reference/2026-11-17       rebuild the grid of a run, render nothing
    python bench/reference_run.py --dry                print what would render (resolved prompts), render nothing

Variants live in reference_prompt.json. Renders go through the running Morpheus server (:5196), so the GPU lock and the
ComfyUI <-> WSL engine switch stay the server's job; one model at a time. Each result lands in
output/reference/<run>/<model>.png|mp4 with <model>.json (model, variant, seed, size, time, VRAM peak); the copy the
server kept in the gallery / a temporary playground is removed again. Missing or failing models are logged and skipped.
Every run (re)writes index.html (relative links) and grid.html (self-contained, for the glia-bridge Canvas).
"""
import argparse
import base64
import json
import re
import shutil
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

API = "http://127.0.0.1:5196"
REF = ROOT / "reference_prompt.json"
OUT = ROOT / "output" / "reference"


# ---- variants ---------------------------------------------------------------------------------------------------

def load_ref(path: Path = REF) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_prompt(ref: dict, text: str, table: dict, seen=()) -> str:
    """'@master' / '@<model id>' (anywhere in the text) stands for that prompt; nested references resolve too."""
    def sub(m):
        key = m.group(1)
        if key in seen:
            raise ValueError(f"Prompt reference loop at @{key}.")
        if key == "master":
            return ref["master"]["prompt"]
        if key not in table:
            raise ValueError(f"Unknown prompt reference @{key}.")
        return resolve_prompt(ref, table[key]["prompt"], table, (*seen, key))
    return re.sub(r"@([A-Za-z0-9_.-]+)", sub, text)


def resolve_negative(ref: dict, value, default: str | None) -> str | None:
    """null = none; '@master' = the master negative; '+text' = the model default plus text; else the text itself.
    A model without a negative (default None) never gets one."""
    if value is None or default is None:
        return None
    if value == "@master":
        return ref["master"]["negative"]
    if value.startswith("+"):
        extra = value[1:].strip()
        return f"{default.rstrip(', ')}, {extra}" if default.strip() else extra
    return value


def image_jobs(ref: dict) -> list[dict]:
    """Every image variant, resolved: {id, pipe, prompt, negative, width, height, style, notes}."""
    import image_models
    import morpheus
    out = []
    for mid, v in ref["image"].items():
        if v["pipe"] == "comfy":
            if mid not in morpheus.MODELS:
                raise ValueError(f"{mid} is not in morpheus.MODELS.")
            default = morpheus.default_negative(mid)
        else:
            m = image_models.get(mid)
            if not m:
                raise ValueError(f"{mid} is not in image_models.json.")
            default = m.get("negative_default", "") if m.get("negative") else None
        w, h = (int(x) for x in v["size"].split("x"))
        out.append({"id": mid, "kind": "image", "pipe": v["pipe"], "style": v["style"], "width": w, "height": h,
                    "prompt": resolve_prompt(ref, v["prompt"], ref["image"]),
                    "negative": resolve_negative(ref, v.get("negative"), default), "notes": v.get("notes", "")})
    return out


def video_jobs(ref: dict) -> list[dict]:
    vt = ref["video"]
    return [{"id": mid, "kind": "video", "prompt": resolve_prompt(ref, v["prompt"], vt["models"]), "notes": v.get("notes", ""),
             "duration": vt["duration"], "aspect": vt["aspect"], "resolution": vt["resolution"],
             "start_frame": str(ROOT / vt["start_frame"])}
            for mid, v in vt["models"].items()]


def missing(job: dict) -> str | None:
    """Why a model can't render here (None = it can). ComfyUI files are checked; engine models are left to the engine."""
    if job["kind"] == "image" and job["pipe"] == "comfy":
        import morpheus
        if not morpheus.installed(job["id"]):
            return "model files not installed"
    return None


# ---- server -----------------------------------------------------------------------------------------------------

def post(path: str, data: dict) -> dict:
    req = urllib.request.Request(API + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(out["error"])
    return out


def get(path: str) -> dict:
    with urllib.request.urlopen(API + path, timeout=30) as r:
        return json.loads(r.read())


def run_job(path: str, req: dict) -> tuple[dict, int | None]:
    """A background job on the server, followed to the end; returns its result and the VRAM peak seen meanwhile."""
    job = post(path, {**req, "background": True})["job"]
    peak, last = None, None
    while True:
        st = get(f"/api/job/{job}")
        try:
            v = get("/api/status").get("vram")
            peak = max(peak or 0, v) if v else peak
        except OSError:
            pass
        phase = f"{st.get('phase', '')} {st.get('step', '')}/{st.get('total', '')}".strip()
        if phase != last:
            print(f"    {phase}", flush=True)
            last = phase
        if st["state"] == "done":
            return st["result"], peak
        if st["state"] == "error":
            raise RuntimeError(st.get("error", "failed"))
        time.sleep(2)


def render_image(job: dict, seed: int, dest: Path) -> dict:
    if job["pipe"] == "comfy":
        req = {"model": job["id"], "prompt": job["prompt"], "width": job["width"], "height": job["height"], "seed": seed}
        if job["negative"] is not None:
            req["negative"] = job["negative"]
        meta, peak = run_job("/api/generate", req)
        src = ROOT / "output" / meta["raw"]
        shutil.copyfile(src, dest)
        post("/api/delete", {"image": meta["image"]})  # out of the gallery again
        return {"seconds": meta["seconds"], "width": meta["width"], "height": meta["height"], "vram_peak": peak}
    pg = post("/api/image_playground/new", {})["id"]
    try:
        save = {"id": pg, "model": job["id"], "size": f"{job['width']}x{job['height']}", "prompt": job["prompt"],
                "seed": seed, "count": 1}
        if job["negative"] is not None:
            save["negative"] = job["negative"]
        post("/api/image_playground/save", save)
        res, peak = run_job("/api/image_playground/render", {"id": pg})
        meta = res["images"][0]
        src = ROOT / "output" / "playground" / pg / meta["image"]
        shutil.copyfile(src, dest)
        from PIL import Image
        w, h = Image.open(dest).size
        return {"seconds": meta.get("render_seconds"), "width": w, "height": h, "vram_peak": peak}
    finally:
        post("/api/playground/delete", {"id": pg})


def render_video(job: dict, seed: int, dest: Path) -> dict:
    pg = post("/api/playground/new", {})["id"]
    try:
        data = "data:image/png;base64," + base64.b64encode(Path(job["start_frame"]).read_bytes()).decode()
        post("/api/playground/frame", {"id": pg, "kind": "start", "data": data})
        post("/api/playground/save", {"id": pg, "model": job["id"], "duration": job["duration"], "aspect": job["aspect"],
                                      "resolution": job["resolution"], "prompt": job["prompt"], "seed": seed})
        meta, peak = run_job("/api/playground/render", {"id": pg})
        shutil.copyfile(ROOT / "output" / "playground" / pg / meta["video"], dest)
        return {"seconds": meta.get("render_seconds"), "vram_peak": peak, "params": meta.get("params"),
                "resolution": (meta.get("resolution") or {}).get("label")}
    finally:
        post("/api/playground/delete", {"id": pg})


# ---- grid -------------------------------------------------------------------------------------------------------

def results(run: Path) -> list[dict]:
    """The sidecars of a run, images first, in reference_prompt.json order."""
    order = {k: n for n, k in enumerate([*load_ref()["image"], *load_ref()["video"]["models"]])}
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in run.glob("**/*.json")]
    return sorted(metas, key=lambda m: (m["kind"] != "image", order.get(m["model"], 999), m["model"]))


def thumb(path: Path, size: int = 640) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tile(m: dict, src: str, full: str) -> str:
    secs = f"{m['seconds']:.0f} s" if isinstance(m.get("seconds"), (int, float)) else "–"
    if m.get("error"):
        body = f'<div class="err">{esc(m["error"])}</div>'
    elif m["kind"] == "video":
        body = f'<video src="{src}" controls loop muted playsinline preload="metadata"></video>'
    else:
        body = f'<a href="{full}" target="_blank"><img src="{src}" alt="{esc(m["model"])}"></a>'
    style = f' · {m["style"]}' if m.get("style") else ""
    return f'<figure>{body}<figcaption><b>{esc(m["model"])}</b><span>{secs}{style}</span></figcaption></figure>'


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reference shot {run}</title><style>
:root{{--bg:#1E1B3A;--fg:#F6EBDD;--mute:#b9b0c9;--card:#2a2650;--acc:#8A6CD8;--err:#F08AA8}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px "IBM Plex Mono",ui-monospace,monospace}}
h1{{font:600 20px Montserrat,system-ui,sans-serif;margin:0 0 4px}} h2{{font:600 15px Montserrat,system-ui,sans-serif;color:var(--acc);margin:20px 0 8px}}
p{{color:var(--mute);margin:0 0 12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border-radius:10px;overflow:hidden}}
img,video{{display:block;width:100%;aspect-ratio:7/9;object-fit:cover;background:#000}}
figcaption{{display:flex;justify-content:space-between;gap:8px;padding:6px 8px;font-size:12px}} figcaption span{{color:var(--mute)}}
.err{{aspect-ratio:7/9;display:grid;place-items:center;padding:12px;color:var(--err);font-size:12px;text-align:center}}
</style></head><body><h1>Reference shot · {run}</h1><p>seed {seed} · {count} results · click a picture for full size</p>{sections}</body></html>"""


def write_grid(run: Path) -> tuple[Path, Path]:
    metas = results(run)
    seed = load_ref()["seed"]

    def page(embed: bool) -> str:
        sections = []
        for kind, title in (("image", "Images"), ("video", "Video (5 s, start frame + motion prompt)")):
            tiles = []
            for m in (x for x in metas if x["kind"] == kind):
                f = run / m["file"] if m.get("file") else None
                if embed and f and f.exists():
                    src = thumb(f) if kind == "image" else "data:video/mp4;base64," + base64.b64encode(f.read_bytes()).decode()
                    full = src if kind == "video" else thumb(f, 1600)
                else:
                    src = full = m.get("file", "")
                tiles.append(tile(m, src, full))
            if tiles:
                sections.append(f'<h2>{title}</h2><div class="grid">{"".join(tiles)}</div>')
        return PAGE.format(run=esc(run.name), seed=seed, count=len(metas), sections="".join(sections))

    index, grid = run / "index.html", run / "grid.html"
    index.write_text(page(False), encoding="utf-8")
    grid.write_text(page(True), encoding="utf-8")
    return index, grid


# ---- main -------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", action="store_true", help="render the video models instead of the image models")
    ap.add_argument("--model", action="append", default=[], help="only this model id (repeatable; image or video)")
    ap.add_argument("--run", default=time.strftime("%Y-%m-%d"), help="run folder name under output/reference")
    ap.add_argument("--grid", help="only rebuild the grid of this run folder")
    ap.add_argument("--dry", action="store_true", help="print the resolved jobs, render nothing")
    a = ap.parse_args(argv)

    if a.grid:
        for p in write_grid(Path(a.grid)):
            print(p)
        return 0
    ref = load_ref()
    jobs = video_jobs(ref) if a.video else image_jobs(ref)
    if a.model:
        all_jobs = {j["id"]: j for j in image_jobs(ref) + video_jobs(ref)}
        unknown = [m for m in a.model if m not in all_jobs]
        if unknown:
            print(f"No variant in reference_prompt.json for: {', '.join(unknown)}")
            return 2
        jobs = [all_jobs[m] for m in a.model]
    if a.dry:
        print(json.dumps(jobs, indent=1, ensure_ascii=False))
        return 0

    run = OUT / a.run
    failed = 0
    for n, job in enumerate(jobs, 1):
        sub = run / "video" if job["kind"] == "video" else run
        sub.mkdir(parents=True, exist_ok=True)
        ext = "mp4" if job["kind"] == "video" else "png"
        dest = sub / f"{job['id']}.{ext}"
        side = {"model": job["id"], "kind": job["kind"], "style": job.get("style"), "seed": ref["seed"],
                "prompt": job["prompt"], "negative": job.get("negative"), "notes": job["notes"],
                "file": str(dest.relative_to(run)).replace("\\", "/"), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        print(f"[{n}/{len(jobs)}] {job['id']}", flush=True)
        why = missing(job)
        if why:
            print(f"    skipped: {why}")
            continue
        try:
            t = time.time()
            info = render_video(job, ref["seed"], dest) if job["kind"] == "video" else render_image(job, ref["seed"], dest)
            side.update(info, wall_seconds=round(time.time() - t, 1))
            print(f"    done in {side['wall_seconds']} s -> {dest}")
        except Exception as e:  # keep going: one broken model must not stop the run
            failed += 1
            side.update(error=str(e), file=None)
            dest.unlink(missing_ok=True)
            print(f"    FAILED: {e}")
        (sub / f"{job['id']}.json").write_text(json.dumps(side, indent=1, ensure_ascii=False), encoding="utf-8")
        write_grid(run)
    index, grid = write_grid(run)
    print(f"grid: {index}\ncanvas: {grid}\n{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
