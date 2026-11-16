"""Remix mode checks (seed_2026_11_16_remix_mode): timeline save rules and the export, both paths.

Runs against a throwaway output folder with ffmpeg-made test clips; touches nothing in output/.
    .venv\\Scripts\\python tools\\check_remix.py
"""
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


def clip(path: Path, size: str, seconds: int, fps: int = 16):
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


with tempfile.TemporaryDirectory() as tmp:
    morpheus.OUTPUT = Path(tmp)
    playground.ROOT = Path(tmp) / "playground"
    pid = playground.new({})["id"]
    clips = playground.folder(pid) / "clips"
    clips.mkdir(parents=True)
    clip(clips / "a.mp4", "640x352", 2)
    clip(clips / "b.mp4", "640x352", 3)
    clip(clips / "tall.mp4", "352x640", 1)

    # timeline save: pass + block
    st = playground.save({"id": pid, "remix": ["clips/a.mp4", "clips/b.mp4", "clips/a.mp4"]})
    check("save keeps order and repeats", st["remix"] == ["clips/a.mp4", "clips/b.mp4", "clips/a.mp4"])
    check("reload keeps the timeline", playground.load(pid)["remix"] == st["remix"])
    check("other saves leave the timeline alone", playground.save({"id": pid, "prompt": "x"})["remix"] == st["remix"])
    check("block: path escape", raises(lambda: playground.save({"id": pid, "remix": ["../x/clips/a.mp4"]}), "Unknown remix"))
    check("block: not a list", raises(lambda: playground.save({"id": pid, "remix": "clips/a.mp4"}), "Unknown remix"))
    check("block: too long", raises(lambda: playground.save({"id": pid, "remix": ["clips/a.mp4"] * 201}), "Unknown remix"))

    # export, stream-copy path: all alike
    r = playground.export_remix({"id": pid})
    out = morpheus.OUTPUT / r["video"]
    info = playground.video_info(out)
    check("copy path chosen for alike clips", r["copied"] is True)
    check(f"copy export length ≈ 2+3+2 s (got {info['seconds']:.2f})", abs(info["seconds"] - 7) < 0.3)
    check("export lands in remix/", out.parent.name == "remix" and out.is_file())

    # export, letterbox path: mixed aspect
    playground.save({"id": pid, "remix": ["clips/a.mp4", "clips/tall.mp4"]})
    r = playground.export_remix({"id": pid})
    info = playground.video_info(morpheus.OUTPUT / r["video"])
    check("re-encode path for mixed sizes", r["copied"] is False)
    check(f"letterboxed into the first clip's frame (got {info['w']}x{info['h']})", (info["w"], info["h"]) == (640, 352))
    check(f"re-encode length ≈ 3 s (got {info['seconds']:.2f})", abs(info["seconds"] - 3) < 0.3)

    # blocks: empty timeline, deleted clip
    playground.save({"id": pid, "remix": []})
    check("block: empty timeline", raises(lambda: playground.export_remix({"id": pid}), "timeline is empty"))
    playground.save({"id": pid, "remix": ["clips/a.mp4", "clips/b.mp4"]})
    (clips / "b.mp4").unlink()
    check("deleted clip stays on the timeline", playground.load(pid)["remix"] == ["clips/a.mp4", "clips/b.mp4"])
    check("block: deleted clip refuses the export", raises(lambda: playground.export_remix({"id": pid}), "gone: b.mp4"))
    check("no temp files left behind", not list((playground.folder(pid) / "remix").glob("*.tmp*")) and
          not list((playground.folder(pid) / "remix").glob("*.txt")))

print("ALL PASS" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
