"""Soulforge: agent avatars. A short description becomes a character (name, soul, look), the look is rendered
as a 3x3 grid of candidates in one fixed style, and the user picks one in the chat. A pick is either locked in
as the base image or fine-tuned: then it goes into the middle of the next grid with eight variations around it.

The style is the winner of the 2026-10-30 blind race: NoobAI-XL, pixel-art tags, 832x1216 portrait, waist-up,
unfiltered. Agents live in output/soulforge/<slug>/agent.json, next to their candidates, outside the gallery.
"""
import json
import random
import re
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

import morpheus

ROOT = morpheus.OUTPUT / "soulforge"
WIDTH, HEIGHT = 832, 1216
GRID = 9

STYLE = ("pixel art, 16-bit, retro game sprite, {look}, solo, cowboy shot, standing, from waist up, arms at sides, "
         "straight-on, head in upper third, whole head visible, top of head visible, hair fully visible, wide shot, "
         "centered, facing viewer, looking at viewer, visible mouth, lips, fully clothed, limited palette, simple background")
# exclusions only work here: "halo-free" in the positive prompt painted halos (race round 3)
NEGATIVE = ("worst quality, low quality, halo, multiple views, 2girls, 2boys, multiple persons, clone, reference sheet, "
            "character sheet, multiple heads, inset, cropped head, cropped hair, close-up, arms up, hand on head, "
            "text, logo, letters, " + morpheus.SDXL_NEGATIVE)
workflow = morpheus.sdxl_workflow("NoobAI-XL-v1.1.safetensors", "euler_ancestral", "normal", 5.0, NEGATIVE,
                                  prefix="masterpiece, best quality, absurdres, ")


def prompt(look: str) -> str:
    return STYLE.format(look=look.strip().strip(","))


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "agent"


def folder(agent_slug: str) -> Path:
    path = (ROOT / agent_slug).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("Unknown agent.")
    return path


def load(agent_slug: str) -> dict:
    f = folder(agent_slug) / "agent.json"
    if not f.is_file():
        raise ValueError("Unknown agent.")
    return json.loads(f.read_text(encoding="utf-8"))


def save(agent: dict) -> dict:
    agent["updated"] = time.time()
    path = folder(agent["slug"])
    path.mkdir(parents=True, exist_ok=True)
    (path / "agent.json").write_text(json.dumps(agent, ensure_ascii=False, indent=1), encoding="utf-8")
    return agent


def agents() -> list[dict]:
    found = []
    for f in ROOT.glob("*/agent.json"):
        try:
            found.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda a: a.get("updated", 0), reverse=True)


def create(name: str, soul: str, look: str, description: str) -> dict:
    base, n = slug(name), 1
    agent_slug = base
    while (ROOT / agent_slug).exists():  # two agents may share a name
        n += 1
        agent_slug = f"{base}-{n}"
    # pick: the candidate being refined (the middle of the next grid); base: the one the user settled on
    return save({"slug": agent_slug, "name": name, "soul": soul, "look": look, "description": description,
                 "grids": [], "pick": None, "base": None, "messages": [], "created": time.time()})


def render_grid(agent: dict, look: str, progress=None, live=None) -> dict:
    """Render GRID candidates of one look with fresh seeds; each lands in the agent as soon as it exists."""
    grid = {"look": look, "prompt": prompt(look), "candidates": []}
    agent["grids"].append(grid)
    number = len(agent["grids"])
    for i in range(GRID):
        grid["slot"] = i  # the slot being drawn, where the page shows the live preview
        if progress:
            progress(i, grid)
        seed = random.randrange(2**31)
        t = time.time()
        picture = morpheus.generate(workflow(grid["prompt"], WIDTH, HEIGHT, seed, 28), ghost=True, progress=live)
        file = f"g{number}_{i + 1}.png"
        picture.save(folder(agent["slug"]) / file)
        grid["candidates"].append({"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "look": look,
                                   "seconds": round(time.time() - t, 1)})
        save(agent)
    return grid


# img2img denoise: close variations, a visible change, or room to try new ideas. At 0.6 the source's big shapes
# (silhouette, hair outline, cape) still won: eight looks with a bob, a ponytail, a braid all came out long-haired.
STRENGTH = {"slight": 0.5, "strong": 0.75, "bold": 0.85}


def variation_workflow(prompt_text: str, image: str, seed: int, denoise: float) -> dict:
    """The text-to-image graph, started from a picture in ComfyUI/input instead of empty noise."""
    graph = workflow(prompt_text, WIDTH, HEIGHT, seed, 28)
    del graph["7"]
    graph["20"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    graph["21"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["20", 0], "vae": ["1", 2]}}
    graph["8"]["inputs"].update(latent_image=["21", 0], denoise=denoise)
    return graph


def render_variations(agent: dict, center: dict, look: str, denoise: float, progress=None,
                      looks: list[str] | None = None, live=None) -> dict:
    """A grid around one candidate: it stays in the middle (slot 5), eight variations of it fill the rest.

    looks, if given, holds one look per variation, so each tries its own idea instead of only differing by noise.
    """
    looks = [l for l in (looks or []) if l.strip()][:GRID - 1]
    grid = {"look": look, "prompt": prompt(look), "center": center["file"], "denoise": denoise,
            "candidates": [None] * GRID}
    grid["candidates"][GRID // 2] = {"file": center["file"], "seed": center["seed"], "original": True,
                                     "look": center.get("look", look)}
    agent["grids"].append(grid)
    number = len(agent["grids"])
    # ComfyUI's LoadImage only reads its own input folder
    staged = morpheus.COMFY_INPUT / "morpheus" / f"soul_{agent['slug']}.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    Image.open(morpheus.OUTPUT / center["file"]).convert("RGB").save(staged)
    slots = [i for i in range(GRID) if i != GRID // 2]
    try:
        for k, i in enumerate(slots):
            grid["slot"] = i
            if progress:
                progress(k, grid)
            seed = random.randrange(2**31)
            own = looks[k % len(looks)] if looks else look
            t = time.time()
            picture = morpheus.generate(variation_workflow(prompt(own), f"morpheus/{staged.name}", seed, denoise), ghost=True,
                                        progress=live)
            file = f"g{number}_{i + 1}.png"
            picture.save(folder(agent["slug"]) / file)
            grid["candidates"][i] = {"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "look": own,
                                     "seconds": round(time.time() - t, 1)}
            save(agent)
    finally:
        staged.unlink(missing_ok=True)
    return grid


EDIT_MODEL = "flux2-klein-4b-edit"


def polish(agent: dict, instruction: str, live=None) -> dict:
    """One edit of the base image by instruction (FLUX.2 Klein), kept only where the picture really changed.

    Klein isn't a pixel-art model and shifts the whole picture a little; pasting back only the changed area keeps
    everything else pixel-identical to the base. The result becomes the new base; the edit remembers the one before.
    """
    before = agent["base"]
    source = Image.open(morpheus.OUTPUT / before["file"]).convert("RGB")
    staged = morpheus.COMFY_INPUT / "morpheus" / f"soul_{agent['slug']}_polish.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    source.save(staged)
    seed = random.randrange(2**31)
    w, h = morpheus.edit_size(*source.size)
    m = morpheus.EDIT_MODELS[EDIT_MODEL]
    t = time.time()
    try:
        edited = morpheus.generate(m["workflow"](instruction, f"morpheus/{staged.name}", w, h, seed, m["steps"]),
                                   ghost=True, progress=live).convert("RGB").resize(source.size, Image.LANCZOS)
    finally:
        staged.unlink(missing_ok=True)
    mixed = Image.composite(edited, source, change_mask(source, edited))
    # the restyle only touches what clearly changed: Klein's drift around it passes the soft mask but must stay as is
    result = restyle(agent, mixed, source, change_mask(source, edited, 40), staged, live)
    n = len(agent.get("edits", [])) + 1
    file = f"polish_{n}_{seed}.png"  # the seed keeps an undone and redone edit from hitting the browser's cache
    result.save(folder(agent["slug"]) / file)
    edit = {"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "instruction": instruction,
            "before": before, "seconds": round(time.time() - t, 1)}
    agent.setdefault("edits", []).append(edit)
    agent["base"] = {"file": edit["file"], "seed": seed, "look": before.get("look"), "edit": n}
    save(agent)
    return edit


# Klein paints smooth and glossy; the pixel-art model inpaints the changed area in the style, keeping its shapes.
# A full-picture pass at 0.45 left the gloss and greyed the face around it (2026-10-30), hence a masked inpaint.
RESTYLE = 0.65


def restyle(agent: dict, picture: Image.Image, source: Image.Image, mask: Image.Image, staged: Path, live=None) -> Image.Image:
    """The clearly changed area inpainted by NoobAI and snapped to the base's palette; everything else stays as is."""
    if not mask.getbbox():
        return picture
    picture.save(staged)
    staged_mask = staged.with_name(staged.stem + "_mask.png")
    mask.save(staged_mask)
    look = (agent.get("base") or {}).get("look") or agent["look"]
    graph = variation_workflow(prompt(look), f"morpheus/{staged.name}", random.randrange(2**31), RESTYLE)
    graph["22"] = {"class_type": "LoadImageMask", "inputs": {"image": f"morpheus/{staged_mask.name}", "channel": "red"}}
    graph["23"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["21", 0], "mask": ["22", 0]}}
    graph["8"]["inputs"]["latent_image"] = ["23", 0]
    try:
        styled = morpheus.generate(graph, ghost=True, progress=live).convert("RGB")
    finally:
        staged.unlink(missing_ok=True)
        staged_mask.unlink(missing_ok=True)
    styled = styled.resize(source.size, Image.NEAREST)
    # only the base's colours: kills Klein's gradients and gloss, which no 16-bit palette has
    palette = source.quantize(colors=48, method=Image.Quantize.MEDIANCUT)
    styled = styled.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
    return Image.composite(styled, picture, mask)


def change_mask(a: Image.Image, b: Image.Image, threshold: int = 18) -> Image.Image:
    """Where b really differs from a, as a soft mask: the small drift everywhere drops out, real changes stay."""
    from PIL import ImageChops, ImageFilter
    diff = ImageChops.difference(a, b).convert("L").filter(ImageFilter.GaussianBlur(6))
    mask = diff.point(lambda v: 255 if v > threshold else 0)
    # close small holes inside a changed area, then a soft edge so the seam doesn't show
    mask = mask.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.MinFilter(9)).filter(ImageFilter.GaussianBlur(4))
    return mask


def undo(agent: dict) -> dict | None:
    """Drop the newest edit: the base goes back to what it was before it."""
    if not agent.get("edits"):
        return None
    edit = agent["edits"].pop()
    agent["base"] = edit["before"]  # its file stays: the chat still shows it as an earlier before/after
    save(agent)
    return edit


def restore(agent: dict) -> None:
    """Back to the locked-in image: every edit is dropped."""
    while agent.get("edits"):
        undo(agent)


# Idle animation: expression overlays of the base. Each holds only its feature's changed pixels, so the page can
# stack any of them on the base (a blush while blinking) without a seam. Klein only finds the eyes and the mouth;
# the expressions are drawn from her own pixels, the way a pixel artist would: every Klein blink and smirk, even on
# a close-up, changed her (eyeshadow lids, a grin with other lips) or didn't line up.
KEEP = (" Change nothing else: same face shape, same hair, same lip colour, same pose, same colours, same pixel art "
        "style, same framing.")
EXPRESSIONS = {  # the whole picture: where it changes most is where her eyes are
    "closed": "Close both of her eyes completely, eyelids shut with long dark lashes, a calm resting face." + KEEP,
}
# a close-up of the face, blown up to the model's size with its big pixels intact: where it changes is her mouth
FIND_MOUTH = ("A faint smirk: one corner of her mouth lifts slightly. Her lips keep exactly the same size, shape and "
              "colour; no teeth, no wider mouth. This is a close-up of a 16-bit pixel art face: keep the big square "
              "pixels, the flat colours and the exact palette. Change nothing else.")


# The body rig: Klein paints a region pure green and the green is keyed out. A segmentation model (BiRefNet) took
# the whole picture for foreground, swirls and all; asking the edit model works on any busy background.
GREEN = " flat solid pure green (#00FF00)"
KEYS = {
    "key_figure": "Replace the entire background behind the character with" + GREEN + ": everything that is not the "
                  "character becomes pure green, including any moon, halo, rings, stars, swirls or decorations behind "
                  "her. Keep the character, hair, clothes exactly the same.",
    "key_hair": "Paint all of her hair, every strand and every coloured streak," + GREEN + ". Change nothing else.",
    "key_cloak": "Paint her whole cloak or cape, with its hood, collar and lining," + GREEN + ". Change nothing else.",
    "plate": "Remove the woman completely, with her hair, cloak and body: nothing of her remains, no silhouette, no "
             "shadow. Continue the background pattern seamlessly through the space where she stood, same colours, "
             "same pixel art style.",
}
BLOCK = 6  # the art's pixel size at 832 wide (measured on the race winners: the step autocorrelation peaks at 6)
TRIES = 3  # Klein is random: one key took the moon for part of her, one plate left a black silhouette
IDLE_STEPS = ["find the eyes", "find the mouth", *[f"{k} {n + 1}" for k in KEYS for n in range(TRIES)],
              "her motif", "her world", "rig"]

# Her scene: the page puts her, cut out, on a shared parallax backdrop; between the two floats her own motif
# (Morpheus's moon), drawn by FLUX.2 Klein on green. NoobAI painted whole green-tinted scenes for "isolated", and
# Klein given her look drew her instead of an emblem: the motif is its own one-line description of one object.
MOTIF = ("16-bit SNES pixel art, retro video game art, clean flat colors, crisp dark outlines, limited palette. "
         "{motif} "
         "The object is isolated in the middle of the picture, fully visible with empty space around it, on a flat "
         "solid pure green (#00FF00) background. Nothing else, no ground, no scenery, no text.")


def keyed(picture: Image.Image) -> Image.Image:
    """Where the picture is the key green, as a mask."""
    from PIL import ImageFilter
    hsv = picture.convert("HSV")
    h, s, v = (c.point(f) for c, f in zip(hsv.split(), (lambda x: 255 if 60 < x < 110 else 0,
                                                         lambda x: 255 if x > 120 else 0,
                                                         lambda x: 255 if x > 110 else 0)))
    from PIL import ImageChops
    return ImageChops.darker(ImageChops.darker(h, s), v).filter(ImageFilter.MedianFilter(5))


def vote(pictures: list[Image.Image]) -> Image.Image:
    """The key mask where most tries agree: a random extra in one try drops out."""
    import numpy as np
    stack = np.stack([np.asarray(keyed(p)) > 128 for p in pictures])
    return Image.fromarray(((stack.sum(0) * 2 > len(pictures)) * 255).astype(np.uint8))


def best_plate(plates: list[Image.Image], figure: Image.Image) -> Image.Image:
    """The plate whose former figure area looks most like the background around it: a plate that left a dark
    silhouette, or kept half of her, differs there."""
    import numpy as np
    from PIL import ImageFilter
    inside = np.asarray(figure) > 128
    ring = (np.asarray(figure.filter(ImageFilter.MaxFilter(41))) > 128) & ~inside

    def miss(p):
        a = np.asarray(p.convert("RGB"), dtype=np.float32)
        return float(np.abs(a[inside].mean(0) - a[ring].mean(0)).sum() + abs(a[inside].std() - a[ring].std()))
    return min(plates, key=miss)


def rig(raw: dict, source: Image.Image, eyes, mouth) -> Image.Image:
    """The motion weights as one RGB picture: R the character, G how far the hair sways (more toward the tips),
    B how far the cloak sways (more toward the hem). The page warps the pixels by them."""
    import numpy as np
    W, H = source.size
    figure = 255 - np.asarray(vote(raw["key_figure"]), dtype=np.float32)
    hair = np.minimum(np.asarray(vote(raw["key_hair"]), dtype=np.float32), figure)
    cloak = np.minimum(np.asarray(vote(raw["key_cloak"]), dtype=np.float32), figure)
    hair = np.minimum(hair, 255 - cloak)  # Klein tends to green the cloak along with the hair
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # the face never sways with the hair (Klein greens the eyes too)
    ew = eyes[2] - eyes[0]
    cx, cy = (eyes[0] + eyes[2]) / 2, (eyes[1] + mouth[3]) / 2
    face = ((xx - cx) / (ew * 0.62)) ** 2 + ((yy - cy) / ((mouth[3] - eyes[1]) * 0.75 + 8)) ** 2 < 1
    hair[face] = 0
    hair[yy > mouth[3] + ew * 1.6] = 0  # side strands reach the chest; below that it's stray cloak edges
    rows = np.nonzero(hair.max(1) > 128)[0]
    if len(rows):  # crown still, tips free
        top, bottom = rows[0], rows[-1]
        hair *= np.clip((yy - top) / max(1, bottom - top), 0, 1) ** 0.8
    rows = np.nonzero(cloak.max(1) > 128)[0]
    if len(rows):  # the shoulders hold it, the hem swings
        top, bottom = rows[0], rows[-1]
        cloak *= np.clip((yy - top - (bottom - top) * 0.25) / max(1, (bottom - top) * 0.75), 0, 1) ** 1.3
    return Image.fromarray(np.dstack([figure, hair, cloak]).clip(0, 255).astype(np.uint8))


def klein(agent: dict, instruction: str, picture: Image.Image, live=None) -> Image.Image:
    """One raw Klein edit of a picture, back at its size."""
    staged = morpheus.COMFY_INPUT / "morpheus" / f"soul_{agent['slug']}_idle.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    picture.save(staged)
    w, h = morpheus.edit_size(*picture.size)
    m = morpheus.EDIT_MODELS[EDIT_MODEL]
    try:
        return morpheus.generate(m["workflow"](instruction, f"morpheus/{staged.name}", w, h, random.randrange(2**31),
                                               m["steps"]),
                                 ghost=True, progress=live).convert("RGB").resize(picture.size, Image.LANCZOS)
    finally:
        staged.unlink(missing_ok=True)


def expression(agent: dict, name: str, source: Image.Image, live=None) -> Image.Image:
    """One raw Klein edit of the base (the eye finder, a key or the plate)."""
    return klein(agent, {**EXPRESSIONS, **KEYS}[name], source, live)


def close_up(agent: dict, instruction: str, source: Image.Image, box: tuple[int, int, int, int], live=None) -> Image.Image:
    """A close-up edit: the box blown up (nearest, so its pixels stay square), edited, shrunk back into place."""
    crop = source.crop(box)
    edited = klein(agent, instruction, crop.resize((1024, 1024), Image.NEAREST), live)
    out = source.copy()
    out.paste(edited.resize(crop.size, Image.BOX), box[:2])
    return out


def skin_tone(source: Image.Image, eyes) -> tuple:
    """Her skin: the light colour that fills the cheeks right below the eyes."""
    import numpy as np
    ew = eyes[2] - eyes[0]
    band = np.asarray(source.crop((eyes[0] + ew // 6, eyes[3] + 4, eyes[2] - ew // 6, eyes[3] + 4 + ew // 6)))
    px = band.reshape(-1, 3)
    light = px[px.astype(int).sum(1) > np.percentile(px.astype(int).sum(1), 40)]
    return tuple(int(v) for v in np.median(light, 0))


def eye_pair(source: Image.Image, closed: Image.Image, eyes) -> list[tuple[int, int, int, int]]:
    """The two open eyes, as boxes: where Klein's closed-eyes edit changed the most, one on each side of the face's
    middle. Finding them by colour failed on red hair, bright eyeshadow and dark skin; the edit's change is the
    opening itself, whatever its colours."""
    import cv2
    import numpy as np
    ew = eyes[2] - eyes[0]
    x0, y0, x1, y1 = eyes[0] - ew // 6, eyes[1] - ew // 6, eyes[2] + ew // 6, eyes[3] + ew // 6
    d = np.asarray(ImageChops.difference(source, closed).convert("L").crop((x0, y0, x1, y1))).astype(int)
    strong = (d > max(40, np.percentile(d, 97) * 0.5)).astype(np.uint8)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((BLOCK, BLOCK), np.uint8))
    strong = cv2.morphologyEx(strong, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mid = strong.shape[1] // 2
    out = []
    for half, off in ((strong[:, :mid], 0), (strong[:, mid:], mid)):
        n, _, st, _ = cv2.connectedComponentsWithStats(half, connectivity=8)
        if n < 2:
            raise ValueError("Couldn't find two open eyes to animate.")
        big = [j for j in range(1, n) if st[j, 4] >= st[1:, 4].max() * 0.15]  # lashes and lid split into parts
        x, y = min(st[j, 0] for j in big), min(st[j, 1] for j in big)
        w, h = max(st[j, 0] + st[j, 2] for j in big) - x, max(st[j, 1] + st[j, 3] for j in big) - y
        out.append((int(x0 + off + x), int(y0 + y), int(x0 + off + x + w), int(y0 + y + h)))
    return out


def eye_frame(source: Image.Image, shut: Image.Image, pair, closed: bool) -> Image.Image:
    """A closed or half-closed blink: Klein's closed eyes (lashes, lid and eyeshadow in her own style), taken only
    around each eye and only where they changed, so its drift elsewhere stays out. Drawing the lid with skin
    came out as flat patches, and a half lid cut at the eye's middle looked pasted on; the half frame is a blend
    toward the closed eyes; pixel_eyes then redraws both on the art's grid."""
    import cv2
    import numpy as np
    src = np.asarray(source).astype(float)
    lid = np.asarray(shut.convert("RGB")).astype(float)
    out = src.copy()
    for ex0, ey0, ex1, ey1 in pair:
        x0, y0, x1, y1 = ex0 - BLOCK, ey0 - 2 * BLOCK, ex1 + BLOCK, ey1 + BLOCK
        changed = (np.abs(lid[y0:y1, x0:x1] - src[y0:y1, x0:x1]).sum(2) > 30).astype(np.uint8)
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((BLOCK, BLOCK), np.uint8)).astype(bool)
        t = 1.0 if closed else 0.55
        region = out[y0:y1, x0:x1]
        region[changed] = src[y0:y1, x0:x1][changed] * (1 - t) + lid[y0:y1, x0:x1][changed] * t
    return Image.fromarray(out.round().astype(np.uint8))


def pixel_eyes(source: Image.Image, closed: Image.Image, eyes) -> dict:
    """The eye frames on the art's own grid and palette, as overlays: Klein's closed eyes snapped cell by cell (a
    lash line wins its cell), the half blink as the closed lid's upper half with a lash line at its edge, and the
    glance drawn as a pixel artist would, each eye's inside shifted one art pixel aside. Left as Klein drew them
    they were fine, smooth eyes on chunky pixel art, and its half blink, a blend, was a double exposure; Klein's
    sideways eyes, snapped, lost their irises."""
    import numpy as np
    W, H = source.size
    grid = W // BLOCK
    gh = round(grid * H / W)
    cx, cy = np.arange(W) * grid // W, np.arange(H) * gh // H
    span = lambda c, g: (np.searchsorted(c, g), np.searchsorted(c, g, "right"))
    lum = lambda a: a[..., 0] * .3 + a[..., 1] * .59 + a[..., 2] * .11
    small = source.resize((grid, gh), Image.Resampling.BOX)
    pal = np.unique(np.asarray(small.quantize(colors=48, method=Image.Quantize.FASTOCTREE).convert("RGB")).reshape(-1, 3),
                    axis=0).astype(float)

    def snap(picture, gxs, gys):
        a, out = np.asarray(picture.convert("RGB")), {}
        for gx in gxs:
            for gy in gys:
                (x0, x1), (y0, y1) = span(cx, gx), span(cy, gy)
                px = a[y0:y1, x0:x1].reshape(-1, 3).astype(float)
                L = lum(px)
                dark = L < L.mean() * .6
                c = px[dark].mean(0) if dark.mean() > .18 else px.mean(0)
                out[(gx, gy)] = pal[np.argmin(((pal - c) ** 2).sum(1))]
        return out

    x0, y0, x1, y1 = eyes
    gx0, gx1 = cx[max(0, x0 - 2 * BLOCK)], cx[min(W - 1, x1 + 2 * BLOCK)]
    gy0, gy1 = cy[max(0, y0 - 3 * BLOCK)], cy[min(H - 1, y1 + 2 * BLOCK)]
    gxs, gys = range(gx0, gx1 + 1), range(gy0, gy1 + 1)
    base = snap(source, gxs, gys)
    shut = Image.alpha_composite(source.convert("RGBA"), closed).convert("RGB")
    cl = {k: v for k, v in snap(shut, gxs, gys).items() if np.abs(v - base[k]).sum() > 24}
    dark = pal[np.argmin(lum(pal))]
    top, bot = cy[y0], cy[min(y1, H - 1)]
    mid = top + (bot - top) * .5
    half = {k: v for k, v in cl.items() if k[1] < mid}
    for gx in {k[0] for k in half}:
        half[(gx, max(gy for g, gy in half if g == gx) + 1)] = dark
    glances = {1: {}, -1: {}}  # to her left and to her right
    ex0, ex1, m = cx[x0], cx[min(x1, W - 1)], (cx[x0] + cx[min(x1, W - 1)]) // 2
    for a, b in ((ex0, m), (m + 1, ex1)):
        rows = list(range(top, bot + 1))
        L = {gy: np.mean([lum(base[(gx, gy)]) for gx in range(a, b + 1)]) for gy in rows}
        lash = min(rows[:len(rows) // 2 + 1], key=L.get)
        lower = [r for r in rows if r > lash + 1]
        lid = min(lower, key=L.get) if lower else bot
        for d, glance in glances.items():
            for gy in range(lash + 1, lid):
                for gx in range(a, b + 1):
                    if a <= gx - d <= b and np.abs(base[(gx - d, gy)] - base[(gx, gy)]).sum() > 24:
                        glance[(gx, gy)] = base[(gx - d, gy)]

    def paint(cellmap):
        rgba = np.zeros((H, W, 4), np.uint8)
        for (gx, gy), col in cellmap.items():
            (x0, x1), (y0, y1) = span(cx, gx), span(cy, gy)
            rgba[y0:y1, x0:x1] = (*col, 255)
        return Image.fromarray(rgba, "RGBA")
    return {"closed": paint(cl), "half": paint(half), "glance": paint(glances[1]), "glance_l": paint(glances[-1])}


def lips(source: Image.Image, mouth, eyes) -> tuple[int, int, int, int]:
    """Her lips inside the mouth box (which spans Klein's whole change): the blob of pixels clearly darker than her
    skin but not near-black (the hair at her jaw), nearest the box's middle (the chin's shadow below is bigger)."""
    import cv2
    import numpy as np
    a = np.asarray(source.crop(mouth)).astype(int)
    tone = sum(skin_tone(source, eyes))  # relative to her skin: fixed levels found no lips on dark skin
    dark = ((a.sum(2) < tone * 0.8) & (a.sum(2) > tone * 0.3)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, centers = cv2.connectedComponentsWithStats(dark, connectivity=4)
    if n < 2:
        return mouth
    mid = np.array([dark.shape[1] / 2, dark.shape[0] / 2])
    k = 1 + int(np.argmin([np.linalg.norm(centers[j] - mid) if stats[j, 4] >= 50 else 1e9 for j in range(1, n)]))
    x, y, w, h = (int(v) for v in stats[k, :4])
    return (int(mouth[0]) + x, int(mouth[1]) + y, int(mouth[0]) + x + w, int(mouth[1]) + y + h)


def smirk_lift(source: Image.Image, mouth) -> Image.Image:
    """The smirk, drawn: her own lips with one corner lifted by one art pixel, rising from the middle of the mouth
    toward the corner. Only the lips' rows move; with more, the jaw line below got a step."""
    import numpy as np
    a = np.asarray(source)
    out = a.copy()
    x0, y0, x1, y1 = mouth
    mid = (x0 + x1) // 2
    for x in range(mid, min(a.shape[1], x1 + BLOCK)):
        lift = round(min(1.0, (x - mid) / max(1, (x1 - mid))) * BLOCK)
        if lift:
            ys = slice(max(0, y0 - BLOCK), min(a.shape[0] - lift, y1 + BLOCK // 2))
            out[ys, x] = a[ys.start + lift:ys.stop + lift, x]
    return Image.fromarray(out)


def blob(source: Image.Image, edited: Image.Image, window: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The box of the strongest change inside window. Klein drifts everywhere (swirls, moon, outlines), but the
    feature it was asked to change is the biggest, densest change, so a heavy blur makes it the peak."""
    from PIL import ImageChops, ImageFilter
    diff = ImageChops.difference(source, edited).convert("L").crop(window)
    heat = diff.filter(ImageFilter.GaussianBlur(10))
    peak = heat.getextrema()[1]
    if peak < 4:
        raise ValueError("The expression edit changed nothing; try again.")
    px, py = next((i % heat.width, i // heat.width) for i, v in enumerate(heat.get_flattened_data()) if v == peak)
    # the feature: the strong change around the peak, not drift elsewhere in the window
    near = (max(0, px - 140), max(0, py - 70), min(heat.width, px + 140), min(heat.height, py + 70))
    x0, y0, x1, y1 = heat.crop(near).point(lambda v: 255 if v > peak * 0.35 else 0).getbbox()
    return (window[0] + near[0] + x0, window[1] + near[1] + y0, window[0] + near[0] + x1, window[1] + near[1] + y1)


def soft_box(size: tuple[int, int], box: tuple[int, int, int, int], pad: int = 8) -> Image.Image:
    from PIL import ImageFilter
    mask = Image.new("L", size, 0)
    mask.paste(255, (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
    return mask.filter(ImageFilter.GaussianBlur(3))


def blush(source: Image.Image, eyes: tuple[int, int, int, int]) -> Image.Image:
    """A blush painted in, not asked of Klein (which barely showed one), drawn on the art's pixel grid: a soft pink
    oval on each cheek. Hatch strokes at this size came out as a checkerboard."""
    x0, y0, x1, y1 = eyes
    w = x1 - x0
    small = Image.new("L", (source.width // BLOCK, source.height // BLOCK), 0)
    d = ImageDraw.Draw(small)
    for ex in (x0 + w * 0.2, x1 - w * 0.2):  # a light outer oval, a denser core: two flat steps, no gradient
        cx, cy = ex / BLOCK, (y1 + w * 0.13) / BLOCK
        for rx, ry, fill in ((w * 0.17, w * 0.075, 80), (w * 0.1, w * 0.04, 135)):
            rx, ry = rx / BLOCK, ry / BLOCK
            d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)
    layer = small.resize(source.size, Image.NEAREST)
    # on skin only (the ovals reach the hair at the face's edge): near her skin colour, which works for dark skin
    # too, where a brightness test found none
    import numpy as np
    from PIL import ImageChops
    near = np.abs(np.asarray(source).astype(int) - np.array(skin_tone(source, eyes))).sum(2) < 110
    layer = ImageChops.darker(layer, Image.fromarray((near * 255).astype(np.uint8)))
    pink = Image.new("RGBA", source.size, (236, 104, 160, 0))
    pink.putalpha(layer)
    return pink


def key_out(picture: Image.Image) -> Image.Image:
    """The green keyed out to transparency; green spill on the edges pulled back."""
    import numpy as np
    rgb = picture.convert("RGB")
    a = np.asarray(rgb).astype(int)
    a[..., 1] = np.minimum(a[..., 1], np.maximum(a[..., 0], a[..., 2]) + 12)
    return Image.fromarray(np.dstack([a, 255 - np.asarray(keyed(rgb))]).astype(np.uint8), "RGBA")


def motif(agent: dict, live=None) -> Image.Image:
    """Her motif, keyed out and cropped: of three tries the one that is most one clean object of a sensible size."""
    import cv2
    import numpy as np
    m = morpheus.MODELS["flux2-klein-4b"]
    best, score = None, -1.0
    for _ in range(TRIES):
        img = morpheus.generate(m["workflow"](MOTIF.format(motif=agent["motif"]), 832, 832, random.randrange(2**31),
                                              m["steps"]), ghost=True, progress=live).convert("RGB")
        cut = key_out(morpheus.pixelate(img, 832 // BLOCK, 48))
        alpha = (np.asarray(cut.split()[3]) > 128).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
        if n < 2:
            continue
        cover = alpha.mean()
        whole = st[1:, 4].max() / alpha.sum()  # the main object's share: specks and a busy scene score low
        s = whole * (1.0 if 0.06 < cover < 0.5 else 0.3)
        if s > score:
            best, score = cut, s
    if best is None:
        raise ValueError("Couldn't draw her motif.")
    return best.crop(best.getbbox())


# Her world: a backdrop of her own besides the shared themes, three layers Klein paints from Claude's three lines
# (agent["world"]), and the light the page casts on her taken from its sky
WORLD_PX = "16-bit SNES pixel art, retro video game art, clean flat colors, crisp dark outlines, limited palette. "
WORLD_GREEN = " Everything above them is a flat solid pure green (#00FF00) background. No text, no people."
WORLD_WHERE = {"sky": " A wide sky panorama, no ground.", "far": " Along the bottom third of the picture.",
               "near": " Nearly black silhouettes, only along the very bottom and the two side edges of the picture."}


def world_light(sky: Image.Image) -> dict:
    """The light her world casts, read off its sky: the key is its brightest colour, on the brighter side; the fill
    the key's complement; the ambient a pale tint of its average hue; the fog its horizon."""
    import colorsys
    import numpy as np
    a = np.asarray(sky.convert("RGB")).astype(float)
    lum = a @ [.3, .59, .11]
    bright = a[lum >= np.percentile(lum, 90)]
    sat = bright.max(1) - bright.min(1)
    top = bright[sat >= np.percentile(sat, 75)].mean(0)  # the brightest colour, not the brightest grey
    h, _, sat = colorsys.rgb_to_hls(*(top / 255))  # its hue, as a light: saturated and bright
    key = np.array(colorsys.hls_to_rgb(h, .74, max(.75, sat))) * 255
    fill = np.array(colorsys.hls_to_rgb((h + .5) % 1, .64, .8)) * 255
    mean = a.reshape(-1, 3).mean(0)
    amb = 255 - (255 - mean / mean.max() * 255) * .35  # a tint, never a colour cast
    horizon = a[int(a.shape[0] * .7):].reshape(-1, 3).mean(0)
    w = a.shape[1] // 2
    side = 1 if lum[:, :w].mean() >= lum[:, w:].mean() else -1
    day = lum.mean() > 140
    rgb = lambda c: [int(round(v)) for v in np.clip(c, 0, 255)]
    pale = rgb(key + (255 - key) * .5)
    return {"key": rgb(key), "fill": rgb(fill), "side": side, "rim": .58 if day else .7, "amb": [*rgb(amb), .12],
            "fog": "rgba({}, {}, {}, .45)".format(*rgb(horizon * .35 + np.array([10, 8, 22]) * .5)), "glow": "#{:02x}{:02x}{:02x}".format(*rgb(key)),
            "day": int(day), "motes": {"n": 22, "col": ["#{:02x}{:02x}{:02x}".format(*pale), "#ffffff"], "vy": -1.2,
                                       "vx": .8, "twinkle": 1}}


def layer_score(layer_img: Image.Image, layer: str) -> float:
    """How well a keyed layer does its job: standing on the bottom edge (no floating islands cut off below), a
    sensible share of the picture (a far layer low, a near one a frame, neither a wall), the top kept clear."""
    import numpy as np
    a = np.asarray(layer_img.split()[3]) > 128
    cover, bottom, top = a.mean(), a[-a.shape[0] // 12:].mean(), a[:a.shape[0] // 3].mean()
    lo, hi = (.08, .45) if layer == "far" else (.12, .6)
    return bottom * (1.0 if lo < cover < hi else .4) * (1 - top)


def world(agent: dict, live=None) -> dict:
    """Her world's three layers, painted and keyed, saved next to her; with the light the page needs."""
    m = morpheus.MODELS["flux2-klein-4b"]
    stamp, out = int(time.time()), {}
    for layer in ("sky", "far", "near"):
        text = WORLD_PX + agent["world"][layer] + WORLD_WHERE[layer] + ("" if layer == "sky" else WORLD_GREEN)
        best, score = None, -1.0
        for _ in range(1 if layer == "sky" else 2):  # a keyed layer can go wrong: two tries, the better kept
            img = morpheus.generate(m["workflow"](text, 1536, 896, random.randrange(2**31), m["steps"]), ghost=True,
                                    progress=live).convert("RGB")
            img = morpheus.pixelate(img, 1536 // BLOCK, 48)
            if layer != "sky":
                img = key_out(img)
            sc = 1.0 if layer == "sky" else layer_score(img, layer)
            if sc > score:
                best, score = img, sc
        img = best
        file = f"scene_world_{layer}_{stamp}.png"
        img.save(folder(agent["slug"]) / file)
        out[layer] = f"soulforge/{agent['slug']}/{file}"
        if layer == "sky":
            out["light"] = world_light(img)
    return out


def idle(agent: dict, progress=None, live=None) -> dict:
    """The idle set of the base image: blink frames (half and closed), a smirk and a blush as overlays, and the
    body rig with its clean background plate."""
    from PIL import ImageChops
    source = Image.open(morpheus.OUTPUT / agent["base"]["file"]).convert("RGB")
    W, H = source.size
    if progress:
        progress(0, "find the eyes")
    shut = expression(agent, "closed", source, live)
    eyes = blob(source, shut, (0, 0, W, H // 2))  # the head is in the upper third
    pair = eye_pair(source, shut, eyes)
    ew = eyes[2] - eyes[0]
    mouth_window = (max(0, eyes[0] - ew // 4), eyes[3], min(W, eyes[2] + ew // 4), min(H, eyes[3] + ew))
    cx, cy, r = (eyes[0] + eyes[2]) // 2, (eyes[1] + eyes[3]) // 2 + ew // 4, int(ew * 1.1)
    face = (max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r))
    if progress:
        progress(1, "find the mouth")
    mouth = lips(source, blob(source, close_up(agent, FIND_MOUTH, source, face, live), mouth_window), eyes)
    raw, i = {}, 2
    for name in KEYS:
        raw[name] = []
        for n in range(TRIES):
            if progress:
                progress(i, f"{name} {n + 1}")
            raw[name].append(expression(agent, name, source, live))
            i += 1

    def overlay(picture):  # only what the drawing changed
        out = picture.convert("RGBA")
        out.putalpha(ImageChops.difference(source, picture).convert("L").point(lambda v: 255 if v else 0))
        return out
    pictures = {**pixel_eyes(source, overlay(eye_frame(source, shut, pair, True)), eyes),
                "smirk": overlay(smirk_lift(source, mouth)), "blush": blush(source, eyes)}
    stamp = int(time.time())
    frames = {}
    for name, picture in pictures.items():
        file = f"idle_{name}_{stamp}.png"
        picture.save(folder(agent["slug"]) / file)
        frames[name] = f"soulforge/{agent['slug']}/{file}"
    scene = agent.get("scene") or {}
    if not scene.get("motif"):  # the motif is about her, not this picture: it outlives a new look
        if progress:
            progress(len(IDLE_STEPS) - 3, "her motif")
        motif(agent, live).save(folder(agent["slug"]) / f"scene_motif_{stamp}.png")
        scene["motif"] = f"soulforge/{agent['slug']}/scene_motif_{stamp}.png"
    if agent.get("world") and not scene.get("world"):  # like the motif, her world outlives a new look
        if progress:
            progress(len(IDLE_STEPS) - 2, "her world")
        scene["world"] = world(agent, live)
    if progress:
        progress(len(IDLE_STEPS) - 1, "rig")
    figure = rig(raw, source, eyes, mouth)
    figure.save(folder(agent["slug"]) / f"idle_rig_{stamp}.png")
    cut = source.convert("RGBA")
    cut.putalpha(figure.split()[0].point(lambda v: 255 if v > 128 else 0))
    cut.save(folder(agent["slug"]) / f"scene_cut_{stamp}.png")
    if scene.get("cut"):
        (morpheus.OUTPUT / scene["cut"]).unlink(missing_ok=True)
    agent["scene"] = {**scene, "base": agent["base"]["file"], "cut": f"soulforge/{agent['slug']}/scene_cut_{stamp}.png"}
    best_plate(raw["plate"], vote(raw["key_figure"]).point(lambda v: 255 - v)).save(
        folder(agent["slug"]) / f"idle_plate_{stamp}.png")
    old = agent.get("idle") or {}
    for f in [*old.get("frames", {}).values(), old.get("rig"), old.get("plate")]:
        if f:  # only the idle set refers to these; the chat doesn't show them
            (morpheus.OUTPUT / f).unlink(missing_ok=True)
    # breathing: the upper body rises, full above the chest, fading out toward the waist
    agent["idle"] = {"base": agent["base"]["file"], "eyes": list(eyes), "mouth": list(mouth), "frames": frames,
                     "rig": f"soulforge/{agent['slug']}/idle_rig_{stamp}.png",
                     "plate": f"soulforge/{agent['slug']}/idle_plate_{stamp}.png", "block": BLOCK,
                     "chest": mouth[3] + int(ew * 1.4), "waist": mouth[3] + int(ew * 3.2)}
    save(agent)
    return agent["idle"]


def sheet(grid: dict, dest: Path) -> None:
    """The grid as one numbered contact sheet, for Claude to look at."""
    w, h = 277, 405
    out = Image.new("RGB", (w * 3, h * 3), (20, 18, 30))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arialbd.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    for i, c in enumerate(grid["candidates"]):
        if not c:
            continue
        x, y = (i % 3) * w, (i // 3) * h
        out.paste(Image.open(morpheus.OUTPUT / c["file"]).convert("RGB").resize((w, h)), (x, y))
        draw.rectangle((x, y, x + 52, y + 52), fill=(0, 0, 0))
        draw.text((x + 14, y + 4), str(i + 1), fill=(255, 255, 255), font=font)
    out.save(dest)
