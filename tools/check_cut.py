"""Cut checks (seed_2026_11_17_clip_cut): frame-exact trim of a shelf clip into a new clip, pass + block cases.

Runs against a throwaway output folder with an ffmpeg-made test clip; touches nothing in output/.
    .venv/Scripts/python tools/check_cut.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus  # noqa: E402
import playground  # noqa: E402

FF = shutil.which("ffmpeg")
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


def frames(p: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


with tempfile.TemporaryDirectory() as tmp:
    morpheus.OUTPUT = Path(tmp)
    playground.ROOT = Path(tmp) / "playground"
    pid = playground.new({})["id"]
    clips = playground.folder(pid) / "clips"
    clips.mkdir(parents=True)
    src = clips / "a.mp4"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x176:rate=16:duration=5",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "48", str(src)], check=True)
    (clips / "a.json").write_text(json.dumps({"model": "m", "duration": 5, "prompt": "p", "loop": {"source": "x"}}), encoding="utf-8")
    vid = f"playground/{pid}/clips/a.mp4"
    n_src = frames(src)

    # pass: 1.0 s .. 3.5 s = frames 16..56 = 40 frames, off a keyframe (gop 48), exact
    meta = playground.cut_clip({"id": pid, "video": vid, "start": 1.0, "end": 3.5})
    new = playground.folder(pid) / meta["video"]
    check(f"new clip has exactly 40 frames (got {frames(new)})", frames(new) == 40)
    check("source untouched", src.is_file() and frames(src) == n_src)
    check("meta: cut source + frames, duration, no loop flag",
          meta["cut"] == {"source": vid, "start_frame": 16, "end_frame": 56} and meta["duration"] == 2.5 and "loop" not in meta and meta["prompt"] == "p")
    check("shelf lists both clips", len(playground.clips(pid)) == 2)
    check("no temp file left", not list(clips.glob("*.tmp.mp4")))

    # pass: start at 0, end before the end; and the very last second
    check("cut from 0 works", frames(playground.folder(pid) / playground.cut_clip({"id": pid, "video": vid, "start": 0, "end": 1})["video"]) == 16)
    check("cut to the end works", frames(playground.folder(pid) / playground.cut_clip({"id": pid, "video": vid, "start": 4, "end": 5})["video"]) == 16)

    # block
    cut = lambda **k: (lambda: playground.cut_clip({"id": pid, "video": vid, **k}))  # noqa: E731
    check("block: start >= end", raises(cut(start=2, end=2), "inside the clip"))
    check("block: end past the clip", raises(cut(start=1, end=9), "inside the clip"))
    check("block: negative start", raises(cut(start=-1, end=2), "inside the clip"))
    check("block: whole clip", raises(cut(start=0, end=5), "whole clip"))
    check("block: no numbers", raises(cut(start="a", end=2), "start and an end"))
    check("block: missing clip", raises(lambda: playground.cut_clip({"id": pid, "video": f"playground/{pid}/clips/zz.mp4", "start": 0, "end": 1}), "gone"))
    check("block: path escape", raises(lambda: playground.cut_clip({"id": pid, "video": "../x.mp4", "start": 0, "end": 1}), "gone"))

print(f"{fails} failed" if fails else "all passed")
sys.exit(1 if fails else 0)
