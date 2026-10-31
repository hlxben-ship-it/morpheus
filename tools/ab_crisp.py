"""Head edit on the current focus crop (lanczos to 1 MP) vs a 512 crop doubled with nearest (exactly 1 MP, crisp)."""
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import morpheus, soulforge
m = morpheus.EDIT_MODELS["qwen-image-edit-2511"]

def qwen(pic, w, h):
    staged = morpheus.COMFY_INPUT / "morpheus" / "ab_crisp.png"; pic.save(staged)
    return morpheus.generate(m["workflow"](instr, "morpheus/ab_crisp.png", w, h, random.randrange(2**31), 4), ghost=True).convert("RGB")

def finish(src, edited):
    clear = soulforge.change_mask(src, edited, 40)
    mixed = Image.composite(edited, src, clear)
    return Image.composite(mixed.quantize(palette=soulforge.edit_palette(src, mixed, clear), dither=Image.Dither.NONE).convert("RGB"), mixed, clear)

whole = Image.open(sys.argv[1]).convert("RGB"); instr = sys.argv[2]
a = whole.copy(); box = soulforge.focus_box(whole, "head"); crop = whole.crop(box)
w, h = morpheus.edit_size(*crop.size)
a.paste(finish(crop, qwen(crop, w, h).resize(crop.size, Image.LANCZOS)), box[:2])
b = whole.copy(); top = box[1]; cx = whole.width // 2
box2 = (cx - 256, top, cx + 256, top + 512); crop2 = whole.crop(box2)
b.paste(finish(crop2, qwen(crop2.resize((1024, 1024), Image.NEAREST), 1024, 1024).resize((512, 512), Image.LANCZOS)), box2[:2])
H = 700
sheet = Image.new("RGB", (whole.width * 3 // 2, H // 2 * 3 // 2 + 20))
for i, (n, p) in enumerate([("before", whole), ("focus crop (lanczos)", a), ("512 crop x2 nearest", b)]):
    sheet.paste(p.crop((0, 0, whole.width, H)).resize((whole.width // 2, H * 3 // 4 // 1 // 2 * 2 // 1 // 1 // 1 // 1 if False else H // 2 * 3 // 2 // 1 // 1 // 1)), (i * whole.width // 2, 20))
    ImageDraw.Draw(sheet).text((i * whole.width // 2 + 6, 4), n, fill="white")
sheet.save(sys.argv[3])
