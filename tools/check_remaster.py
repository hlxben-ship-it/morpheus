"""Remaster checks (seed_2026_11_16_remix_remaster): eligibility pass + block, the snapshot rewrite (model + resolution only),
duplicates once, the timeline swap, resume after a stop, and the run loop with a fake engine (no GPU, no WSL).

Runs against a throwaway output folder; touches nothing in output/.
    .venv\\Scripts\\python tools\\check_remaster.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus  # noqa: E402
import playground  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += not ok


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        return text in str(e)
    return False


with tempfile.TemporaryDirectory() as tmp:
    morpheus.OUTPUT = Path(tmp)
    playground.ROOT = Path(tmp) / "playground"
    pid = playground.new({})["id"]
    base = playground.folder(pid)
    clips = base / "clips"
    clips.mkdir(parents=True)
    (base / "start_1.png").write_bytes(b"png")

    def shot(name, model="h3_10eros", steps=6, seed=1234, **extra):
        (clips / f"{name}.mp4").write_bytes(b"mp4")
        meta = {"format": 1, "playground": pid, "prompt": f"prompt {name}", "model": {"id": model, "label": model},
                "params": {"main.steps": steps}, "total_steps": steps,
                "resolution": {"id": "720p", "label": "720p", "target": "720p"}, "aspect": "16:9", "duration": 5,
                "start_frame": "start_1.png", "end_frame": None, "main_first": True, "refs": [None] * 4, "last_ref": None,
                "pictures": [{"file": "start_1.png", "role": "start"}], "loras": [], "seed": seed,
                "video": f"clips/{name}.mp4", "created": "2026-11-16T10:00:00", "render_seconds": 240, **extra}
        (clips / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
        return f"clips/{name}.mp4"

    a, b = shot("a"), shot("b", steps=4, seed=99)
    ltx = shot("ltx", model="ltx23")
    flash = shot("flash", model="h3_gguf_q6")
    up = shot("up", upscale={"method": "relay"})
    lp = shot("lp", loop={"source": "x"})
    plan = lambda: playground.remaster_plan({"id": pid})
    setr = lambda l: playground.save({"id": pid, "remix": l})

    # eligibility: pass + block
    check("block: empty timeline", not plan()["ok"])
    setr([a, b, a])
    p = plan()
    check("pass: all H3 10Eros", p["ok"])
    check("duplicates render once, in timeline order", [s["ref"] for s in p["shots"]] == [a, b])
    check("estimate = steps x 9 min (6+4 steps)", p["seconds"] == 10 * 9 * 60)
    for bad, word in ((ltx, "ltx23"), (flash, "h3_gguf_q6"), (up, "upscale"), (lp, "loop")):
        setr([a, bad])
        q = plan()
        check(f"block: {Path(bad).stem} ({word}) disables Remaster and names the clip", not q["ok"] and Path(bad).name in q["reason"])
    setr([a, "clips/gone.mp4"])
    check("block: a deleted timeline clip", not plan()["ok"] and "gone" in plan()["reason"])

    # snapshot rewrite: only model + resolution change
    _, meta = playground.clip_meta(pid, f"playground/{pid}/{a}")
    snap = playground.remaster_snap(pid, a, meta)
    check("snap: model is 10Eros", snap["model"]["id"] == "h3_10eros")
    check("snap: resolution 1080p", snap["resolution"]["id"] == "1080p")
    check("snap: same seed / prompt / steps / pictures / duration / aspect",
          (snap["seed"], snap["prompt"], snap["params"], snap["pictures"], snap["duration"], snap["aspect"])
          == (1234, "prompt a", {"main.steps": 6}, meta["pictures"], 5, "16:9"))
    check("snap: links its source, drops render-only fields", snap["remaster_of"] == a and "video" not in snap and "render_seconds" not in snap)
    import lora_library  # noqa: E402
    lf = Path(tmp) / "style.safetensors"
    lf.write_bytes(b"x" * 10)
    lib = {"id": "style", "name": "Style", "family": "h3", "file": "style.safetensors", "sha256": "aaa"}
    lora_library.get = lambda i: dict(lib) if i == "style" else None
    lora_library.file_path = lambda e: lf
    playground.inspect = lambda req: {"summary": []}
    ml = {**meta, "loras": [{"slot": 1, "id": "style", "name": "Style", "strength": 0.8, "bytes": 10, "sha256": "aaa"}]}
    sl = playground.remaster_snap(pid, a, ml)
    check("lora: same file passes, strength kept", [(l["id"], l["strength"], l.get("sha256")) for l in sl["loras"]] == [("style", 0.8, "aaa")])
    lib["sha256"] = "bbb"
    check("block: LoRA re-uploaded as a different file (sha)", raises(lambda: playground.remaster_snap(pid, a, ml), "different file"))
    lib["sha256"] = "aaa"; lf.write_bytes(b"x" * 11)
    check("block: LoRA file changed size (old meta, no sha)", raises(lambda: playground.remaster_snap(pid, a, {**ml, "loras": [{**ml["loras"][0], "sha256": None}]}), "different file"))
    lf.unlink()
    check("block: LoRA gone from the library", raises(lambda: playground.remaster_snap(pid, a, ml), "no longer in the library"))
    check("block: no recorded seed", raises(lambda: playground.remaster_snap(pid, a, {**meta, "seed": None}), "no recorded seed"))
    check("block: a picture is gone", raises(lambda: playground.remaster_snap(pid, a, {**meta, "pictures": [{"file": "nope.png", "role": "start"}]}), "is gone"))

    # the run loop with a fake engine: sequential, swap after each shot, stop keeps finished shots, resume skips them
    calls = []

    class Fake:  # the engine: records the snapshot, writes a file; can stop like the real one
        stop_after = None

        def __call__(self, snap):
            calls.append({"model": snap["model"]["id"], "seed": snap["seed"], "res": snap["resolution"]["id"]})
            if Fake.stop_after is not None and len(calls) > Fake.stop_after:
                raise ValueError("Stopped.")
            f = Path(tmp) / f"render_{len(calls)}.mp4"
            f.write_bytes(b"new")
            return {"file": str(f), "seconds": 2}, snap["seed"]

    run = lambda job: playground.remaster_run(pid, job, Fake())
    try:
        setr([a, b, a])
        Fake.stop_after = 1
        job = {}
        check("stop mid-run raises Stopped.", raises(lambda: run(job), "Stopped"))
        tl = playground.load(pid)["remix"]
        check("finished shot swapped (both copies), the stopped one kept", tl[0] == tl[2] != a and tl[1] == b)
        check("job counts shots", job.get("shots") == 2 and job.get("shots_done") == 1)
        check("render 1: 10Eros, 1080p, the clip's own seed", calls[0] == {"model": "h3_10eros", "seed": 1234, "res": "1080p"})
        p = plan()
        check("resume: only the unfinished shot left", p["ok"] and [s["ref"] for s in p["shots"]] == [b])
        # the original put back on the timeline: swapped, not rendered again
        setr([a, b])
        p = plan()
        check("resume: an existing remaster is swapped, not re-rendered", p["swaps"] == {a: tl[0]} and [s["ref"] for s in p["shots"]] == [b])
        Fake.stop_after, calls[:] = None, []
        out = run({})
        tl = playground.load(pid)["remix"]
        check("resume run renders only b, seed 99", [c["seed"] for c in calls] == [99] and out["shots"] == 1 and out["swapped"] == 1)
        metas = [json.loads((base / r).with_suffix(".json").read_text(encoding="utf-8")) for r in tl]
        check("timeline is all remasters now", all(m.get("remaster_of") for m in metas))
        check("remaster meta: 10Eros 1080p + seed", [(m["model"]["id"], m["resolution"]["id"], m["seed"]) for m in metas]
              == [("h3_10eros", "1080p", 1234), ("h3_10eros", "1080p", 99)])
        check("originals stay on the shelf", (base / a).is_file() and (base / b).is_file())
        check("block: everything remastered", not plan()["ok"] and "already" in plan()["reason"])
    finally:
        pass  # nothing to restore: the engine is passed in

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
