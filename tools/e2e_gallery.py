"""All E2E agents on one sheet: the locked-in base and every edit after it, with the request and the judge's scores.

    .venv/Scripts/python.exe tools/e2e_gallery.py out.png
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import morpheus  # noqa: E402

TW, TH = 208, 304  # a quarter of the 832x1216 portrait
try:
    font = ImageFont.truetype("segoeui.ttf", 13)
    bold = ImageFont.truetype("segoeuib.ttf", 15)
except OSError:
    font = bold = ImageFont.load_default()

runs = [json.loads(l) for l in (ROOT / "tools" / "e2e_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = []
for r in runs:
    agent = json.loads((morpheus.OUTPUT / "soulforge" / r["slug"] / "agent.json").read_text(encoding="utf-8"))
    edits = agent.get("edits", [])
    shots = [(edits[0]["before"]["file"] if edits else agent["base"]["file"], "locked in", "")]
    scored = {e.get("after"): e for e in r["edits"] if e.get("after")}
    for e in edits:
        s = (scored.get(e["file"]) or {}).get("score")
        shots.append((e["file"], e["instruction"][:60], f"{s['followed']}{s['identity']}{s['style']}{s['artifacts']}" if s else ""))
    rows.append((r, shots))

cols = max(len(s) for _, s in rows)
LABEL = 230
sheet = Image.new("RGB", (LABEL + cols * TW, len(rows) * (TH + 34)), (18, 16, 26))
d = ImageDraw.Draw(sheet)
for k, (r, shots) in enumerate(rows):
    y = k * (TH + 34)
    d.text((10, y + 10), r["name"], fill="white", font=bold)
    words, line, ly = r["description"].split(), "", y + 34
    for w in words:  # the description, wrapped
        if d.textlength(line + " " + w, font=font) > LABEL - 20:
            d.text((10, ly), line.strip(), fill=(190, 190, 205), font=font)
            line, ly = "", ly + 17
        line += " " + w
    d.text((10, ly), line.strip(), fill=(190, 190, 205), font=font)
    d.text((10, y + TH - 10), f"{r['total'] / 60:.1f} min" + ("  ·  alive" if r.get("frames") else ""), fill=(150, 200, 150), font=font)
    for i, (file, caption, score) in enumerate(shots):
        x = LABEL + i * TW
        sheet.paste(Image.open(morpheus.OUTPUT / file).convert("RGB").resize((TW, TH), Image.LANCZOS), (x, y))
        d.text((x + 4, y + TH + 2), caption[:30], fill=(220, 220, 230), font=font)
        if score:
            d.text((x + 4, y + TH + 17), f"judge {score[0]}/{score[1]}/{score[2]}/{score[3]}", fill=(240, 200, 120), font=font)
sheet.save(sys.argv[1])
print(len(rows), "agents")
