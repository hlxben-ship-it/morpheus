"""One zoomed head edit, pasted back with the soft mask (old) and with only the clear changes (new)."""
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus, soulforge

whole = Image.open(sys.argv[1]).convert("RGB"); instr = sys.argv[2]
box = soulforge.focus_box(whole, "head"); src = whole.crop(box)
staged = morpheus.COMFY_INPUT / "morpheus" / "ab_mask.png"; src.save(staged)
w, h = morpheus.edit_size(*src.size); m = morpheus.EDIT_MODELS["qwen-image-edit-2511"]
edited = morpheus.generate(m["workflow"](instr, "morpheus/ab_mask.png", w, h, random.randrange(2**31), 4), ghost=True).convert("RGB").resize(src.size, Image.LANCZOS)
clear = soulforge.change_mask(src, edited, 40)
out = []
for mask in (soulforge.change_mask(src, edited), clear):
    mixed = Image.composite(edited, src, mask)
    out.append(Image.composite(mixed.quantize(palette=soulforge.edit_palette(src, mixed, clear), dither=Image.Dither.NONE).convert("RGB"), mixed, clear))
sheet = Image.new("RGB", (src.width * 3, src.height + 20))
for i, (n, p) in enumerate([("before", src), ("old: soft mask", out[0]), ("new: clear changes only", out[1])]):
    sheet.paste(p, (i * src.width, 20)); ImageDraw.Draw(sheet).text((i * src.width + 6, 4), n, fill="white")
sheet.resize((sheet.width * 2 // 3, sheet.height * 2 // 3)).save(sys.argv[3])
