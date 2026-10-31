"""The same edit through each edit model, side by side: tools/ab_edit.py <picture> "<instruction>" <out.png>

Each runs the way polish() does (edit, keep only what changed, restyle in pixel art); nothing lands in an agent.
"""
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import morpheus  # noqa: E402
import soulforge  # noqa: E402


def edit(source: Image.Image, instruction: str, model: str, restyle: bool = True) -> tuple[Image.Image, float]:
    staged = morpheus.COMFY_INPUT / "morpheus" / f"ab_{model}.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    source.save(staged)
    w, h = morpheus.edit_size(*source.size)
    m = morpheus.EDIT_MODELS[model]
    t = time.time()
    try:
        edited = morpheus.generate(m["workflow"](instruction, f"morpheus/{staged.name}", w, h, random.randrange(2**31), m["steps"]),
                                   ghost=True).convert("RGB").resize(source.size, Image.LANCZOS)
    finally:
        staged.unlink(missing_ok=True)
    mixed = Image.composite(edited, source, soulforge.change_mask(source, edited))
    if restyle:
        mixed = soulforge.restyle({"look": "", "base": {}}, mixed, source, soulforge.change_mask(source, edited, 40), staged)
    return mixed, time.time() - t


if __name__ == "__main__":
    src = Image.open(sys.argv[1]).convert("RGB")
    instruction, out = sys.argv[2], Path(sys.argv[3])
    shots = [("before", src, 0)]
    for model in sys.argv[4:] or ["flux2-klein-4b-edit", "qwen-image-edit-2511"]:
        pic, s = edit(src, instruction, model)
        shots.append((model, pic, s))
        print(model, round(s, 1), flush=True)
    w, h = src.size
    sheet = Image.new("RGB", (w * len(shots) // 2, h // 2 + 24), (20, 20, 24))
    for i, (name, pic, s) in enumerate(shots):
        sheet.paste(pic.resize((w // 2, h // 2)), (i * w // 2, 24))
        ImageDraw.Draw(sheet).text((i * w // 2 + 6, 5), f"{name} {s:.0f}s" if s else name, fill="white")
    sheet.save(out)
