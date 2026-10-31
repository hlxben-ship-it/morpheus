"""Qwen edit once, then three finishes side by side: raw, palette snap only, full NoobAI restyle."""
import random, sys, time
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus, soulforge

src = Image.open(sys.argv[1]).convert("RGB")
instruction, out = sys.argv[2], Path(sys.argv[3])
staged = morpheus.COMFY_INPUT / "morpheus" / "ab_rs.png"
src.save(staged)
w, h = morpheus.edit_size(*src.size)
m = morpheus.EDIT_MODELS["qwen-image-edit-2511"]
edited = morpheus.generate(m["workflow"](instruction, "morpheus/ab_rs.png", w, h, random.randrange(2**31), 4), ghost=True).convert("RGB").resize(src.size, Image.LANCZOS)
soft, hard = soulforge.change_mask(src, edited), soulforge.change_mask(src, edited, 40)
raw = Image.composite(edited, src, soft)
snap = Image.composite(raw.quantize(palette=soulforge.edit_palette(src, raw, hard), dither=Image.Dither.NONE).convert("RGB"), raw, hard)
full = soulforge.restyle({"look": sys.argv[4] if len(sys.argv) > 4 else ""}, raw, src, hard, staged)
shots = [("before", src), ("qwen raw", raw), ("palette snap", snap), ("noobai restyle", full)]
sheet = Image.new("RGB", (w // 2 * 4, h // 2 + 24))
for i, (n, p) in enumerate(shots):
    sheet.paste(p.resize((src.width // 2, src.height // 2)), (i * src.width // 2, 24))
    ImageDraw.Draw(sheet).text((i * src.width // 2 + 6, 5), n, fill="white")
sheet.save(out)
# a close-up of the changed area, 2x, to judge the pixel grid
box = hard.getbbox()
if box:
    cw = min(300, box[2] - box[0]); ch = min(300, box[3] - box[1])
    crop = (box[0], box[1], box[0] + cw, box[1] + ch)
    z = Image.new("RGB", (cw * 2 * 3, ch * 2))
    for i, p in enumerate([raw, snap, full]):
        z.paste(p.crop(crop).resize((cw * 2, ch * 2), Image.NEAREST), (i * cw * 2, 0))
    z.save(out.with_name(out.stem + "_zoom.png"))
