"""A small addition edited on the full picture vs on a zoomed crop: tools/ab_zoom.py pic "instr" x0 y0 x1 y1 out.png"""
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus, soulforge

def qwen(pic):
    staged = morpheus.COMFY_INPUT / "morpheus" / "ab_zoom.png"
    pic.save(staged)
    w, h = morpheus.edit_size(*pic.size)
    m = morpheus.EDIT_MODELS["qwen-image-edit-2511"]
    return morpheus.generate(m["workflow"](instr, "morpheus/ab_zoom.png", w, h, random.randrange(2**31), 4), ghost=True).convert("RGB").resize(pic.size, Image.LANCZOS)

def finish(src, edited):
    clear = soulforge.change_mask(src, edited, 40)
    mixed = Image.composite(edited, src, soulforge.change_mask(src, edited))
    return Image.composite(mixed.quantize(palette=soulforge.edit_palette(src, mixed, clear), dither=Image.Dither.NONE).convert("RGB"), mixed, clear)

src = Image.open(sys.argv[1]).convert("RGB"); instr = sys.argv[2]
W, H = src.size
box = tuple(int(float(v) * (W if i % 2 == 0 else H)) for i, v in enumerate(sys.argv[3:7]))
full = finish(src, qwen(src))
crop = src.crop(box)
zoomed = src.copy(); zoomed.paste(finish(crop, qwen(crop)), box[:2])
sheet = Image.new("RGB", (W // 2 * 3, H // 2 + 20))
for i, (n, p) in enumerate([("before", src), ("full picture", full), ("zoomed crop", zoomed)]):
    sheet.paste(p.resize((W // 2, H // 2)), (i * W // 2, 20)); ImageDraw.Draw(sheet).text((i * W // 2 + 6, 4), n, fill="white")
sheet.save(sys.argv[7])
