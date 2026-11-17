"""Side-by-side sheet of H3-Image bench pictures: python sheet_png.py out.jpg variant... (each labelled with its warm time)."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
R = Path("C:/dev/morpheus/output/bench/h3tune"); out, names = sys.argv[1], sys.argv[2:]
tiles = []
for n in names:
    im = Image.open(R / n / f"{n}.png").convert("RGB"); im.thumbnail((640, 960))
    res = json.loads((R / n / "result.json").read_text()); j = res["jobs"]
    lab = f"{n}  cold {j[0]['total']}s  warm {j[-1]['total']}s  ssim {res['quality'].get('ssim','-')}"
    d = ImageDraw.Draw(im); d.rectangle((0, 0, im.width, 30), fill=(0, 0, 0))
    d.text((8, 6), lab, fill=(255, 255, 255), font=ImageFont.load_default(size=18)); tiles.append(im)
W = sum(t.width for t in tiles); H = max(t.height for t in tiles); s = Image.new("RGB", (W, H)); x = 0
for t in tiles: s.paste(t, (x, 0)); x += t.width
s.save(out, quality=90); print(out)
