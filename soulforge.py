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
STEPS = 20  # same-seed test at 28/20/16: 20 holds the quality of 28 at 9.6 s instead of 14 s; 16 got muddy

STYLE = ("pixel art, 16-bit, retro game sprite, {look}, solo, cowboy shot, standing, from waist up, arms at sides, "
         "straight-on, head in upper third, whole head visible, top of head visible, hair fully visible, wide shot, "
         "centered, facing viewer, looking at viewer, visible mouth, lips, fully clothed, limited palette, simple background")
# exclusions only work here: "halo-free" in the positive prompt painted halos (race round 3)
NEGATIVE = ("worst quality, low quality, halo, multiple views, 2girls, 2boys, multiple persons, clone, reference sheet, "
            "character sheet, multiple heads, inset, cropped head, cropped hair, close-up, arms up, hand on head, "
            "text, logo, letters, name tag, english text, child, loli, shota, aged down, " + morpheus.SDXL_NEGATIVE)
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


def figure_top(picture: Image.Image) -> float:
    """Where she starts, as a share of the height: everything unlike the border's colour counts as her.

    NoobAI draws about one in three "wide shot"s as a tiny figure in a sea of backdrop (her top at 0.37-0.49 of
    the height); the usable ones start in the top tenth.
    """
    from PIL import ImageChops, ImageFilter
    small = picture.convert("RGB").resize((picture.width // 8, picture.height // 8))
    w, h = small.size
    border = [small.getpixel((x, 0)) for x in range(w)] + [small.getpixel((x, y)) for y in range(h) for x in (0, w - 1)]
    bg = tuple(sorted(c[i] for c in border)[len(border) // 2] for i in range(3))
    mask = ImageChops.difference(small, Image.new("RGB", small.size, bg)).convert("L")
    box = mask.point(lambda v: 255 if v > 38 else 0).filter(ImageFilter.MinFilter(3)).getbbox()
    return box[1] / h if box else 1.0


SMALL = 0.25  # a figure starting lower than this is drawn again
REROLLS = 2


def render_grid(agent: dict, look: str, progress=None, live=None, looks: list[str] | None = None) -> dict:
    """Render GRID candidates with fresh seeds; each lands in the agent as soon as it exists.

    looks, if given, are different takes on the agent drawn in turn (with three: one per column), so a first grid
    offers a real choice instead of nine draws of one look.
    """
    looks = [l for l in (looks or []) if l.strip()] or [look]
    grid = {"look": look, "prompt": prompt(look), "candidates": []}
    agent["grids"].append(grid)
    number = len(agent["grids"])
    for i in range(GRID):
        grid["slot"] = i  # the slot being drawn, where the page shows the live preview
        if progress:
            progress(i, grid)
        t = time.time()
        own = looks[i % len(looks)]
        for _ in range(REROLLS + 1):
            seed = random.randrange(2**31)
            picture = morpheus.generate(workflow(prompt(own), WIDTH, HEIGHT, seed, STEPS), ghost=True, progress=live)
            if figure_top(picture) < SMALL:
                break
        file = f"g{number}_{i + 1}.png"
        picture.save(folder(agent["slug"]) / file)
        grid["candidates"].append({"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "look": own,
                                   "seconds": round(time.time() - t, 1)})
        save(agent)
    return grid


# img2img denoise: close variations, a visible change, or room to try new ideas. At 0.6 the source's big shapes
# (silhouette, hair outline, cape) still won: eight looks with a bob, a ponytail, a braid all came out long-haired.
STRENGTH = {"slight": 0.5, "strong": 0.75, "bold": 0.85}


def variation_workflow(prompt_text: str, image: str, seed: int, denoise: float) -> dict:
    """The text-to-image graph, started from a picture in ComfyUI/input instead of empty noise."""
    graph = workflow(prompt_text, WIDTH, HEIGHT, seed, STEPS)
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
# Qwen-Image-Edit reads the picture with a vision model before it edits: outfit changes keep her face, pose and
# the clothing's shape where Klein repainted the whole torso; Klein stays for when it isn't installed
EDIT_MODELS = ["qwen-image-edit-2511", EDIT_MODEL]


def edit_model() -> str:
    return next(m for m in EDIT_MODELS if morpheus.installed(m))


UNCHANGED = 0.002  # an edit touching less than this share of the picture counts as not done


def changed_share(mask: Image.Image) -> float:
    return sum(mask.point(lambda v: 1 if v > 128 else 0).getdata()) / (mask.width * mask.height)


# Small additions get a zoomed crop: at the whole picture's scale headphones around the neck are a few dozen pixels
# and came out as two brown blobs; on the crop (about 2x) Qwen drew the orange pair with its band. The windows are
# share of width and of height, from where the figure starts (a waist-up portrait: the head in the upper third).
FOCUS = {"head": (0.1, -0.12, 0.9, 0.42), "neck": (0.12, 0.08, 0.88, 0.55)}  # generous: dark hair on a dark
# backdrop starts up to a tenth higher than figure_top() sees


def focus_box(source: Image.Image, focus: str) -> tuple[int, int, int, int] | None:
    if focus not in FOCUS:
        return None
    x0, y0, x1, y1 = FOCUS[focus]
    top = figure_top(source)
    W, H = source.size
    return (int(x0 * W), max(0, int((top + y0) * H)), int(x1 * W), min(H, int((top + y1) * H)))


def behind_mask(agent: dict, source: Image.Image, live=None) -> Image.Image:
    """Where the background is, from two Klein key edits that agree, pulled in a little from her edge."""
    from PIL import ImageFilter
    import numpy as np
    keys = [klein(agent, KEYS["key_figure"], source, live) for _ in range(2)]
    stack = np.stack([painted(k, source) for k in keys])
    mask = Image.fromarray((stack.all(0) * 255).astype(np.uint8))
    return mask.filter(ImageFilter.MinFilter(7))


def extend_sky(source: Image.Image, behind: Image.Image) -> Image.Image:
    """Flat near-black background (a band the art model left unpainted) filled with the sky above it, tiled with a
    sideways shift per tile so the stars don't repeat in columns."""
    import numpy as np
    a = np.asarray(source.convert("RGB")).copy()
    H, W, _ = a.shape
    dark = (a.sum(2) < 45) & (np.asarray(behind) > 128)
    rows = np.nonzero(dark.mean(1) > 0.12)[0]
    if not len(rows):
        return source
    top = int(rows[0])
    back = np.asarray(behind) > 128
    sky = a[:top].copy()
    bare = np.median(sky[back[:top] & (sky.sum(2) < 200)], axis=0).astype(np.uint8)  # the sky's own colour
    sky[~back[:top]] = bare  # her hair and face are no sky: never copied out
    seam = max(0, top - 8 * BLOCK)  # the art model's dotted edge along the old boundary goes too
    fill = dark.copy()
    fill[seam:top + 2 * BLOCK] = back[seam:top + 2 * BLOCK]
    for y in range(seam, H):
        k, r = divmod(y - seam, seam or 1)
        row = np.roll(sky[r % top], (k + 1) * 97 % W, axis=0)
        a[y][fill[y]] = row[fill[y]]
    return Image.fromarray(a)


RETOUCH = {"extend_sky": "the flat black background filled with her starry sky"}


def retouch(agent: dict, op: str, live=None) -> dict:
    """A pixel operation on the base, recorded like an edit (undo works): for what edit models won't do reliably."""
    before = agent["base"]
    source = Image.open(morpheus.OUTPUT / before["file"]).convert("RGB")
    t = time.time()
    if op == "extend_sky":
        from PIL import ImageFilter
        result = extend_sky(source, behind_mask(agent, source, live).filter(ImageFilter.MaxFilter(7)))
    else:
        raise ValueError(f"Unknown retouch: {op}")
    n = len(agent.get("edits", [])) + 1
    seed = random.randrange(2**31)
    file = f"retouch_{n}_{seed}.png"
    result.save(folder(agent["slug"]) / file)
    edit = {"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "instruction": RETOUCH[op], "before": before,
            "seconds": round(time.time() - t, 1), "model": "retouch", "focus": "background"}
    agent.setdefault("edits", []).append(edit)
    agent["base"] = {"file": edit["file"], "seed": seed, "look": before.get("look"), "edit": n}
    save(agent)
    return edit


def polish(agent: dict, instruction: str, live=None, focus: str = "whole") -> dict:
    """One edit of the base image by instruction (Qwen-Image-Edit, else FLUX.2 Klein), kept only where the picture
    really changed.

    The edit model isn't a pixel-art model and shifts the whole picture a little; pasting back only the changed area keeps
    everything else pixel-identical to the base. The result becomes the new base; the edit remembers the one before.
    """
    before = agent["base"]
    whole = Image.open(morpheus.OUTPUT / before["file"]).convert("RGB")
    box = focus_box(whole, focus)
    source = whole.crop(box) if box else whole
    staged = morpheus.COMFY_INPUT / "morpheus" / f"soul_{agent['slug']}_polish.png"
    staged.parent.mkdir(parents=True, exist_ok=True)
    source.save(staged)
    w, h = morpheus.edit_size(*source.size)
    model = edit_model()
    m = morpheus.EDIT_MODELS[model]
    t = time.time()
    try:
        # an edit model now and then hands the picture back all but unchanged; a second seed usually does it
        for _ in range(2):
            seed = random.randrange(2**31)
            edited = morpheus.generate(m["workflow"](instruction, f"morpheus/{staged.name}", w, h, seed, m["steps"]),
                                       ghost=True, progress=live).convert("RGB").resize(source.size, Image.LANCZOS)
            changed = changed_share(change_mask(source, edited, 40))
            if changed >= UNCHANGED:
                break
    finally:
        staged.unlink(missing_ok=True)
    # the finish only touches what clearly changed: the drift around it passes the soft mask but must stay as is
    clear = change_mask(source, edited, 40)
    if focus == "background":  # only behind her: a sky fix also flattened Glia's eyes (2026-11-01)
        behind = behind_mask(agent, source, live)
        from PIL import ImageChops
        clear = ImageChops.darker(clear, behind)
        edited = Image.composite(edited, source, behind)
    # on a zoomed face the model also drifts skin tone and redraws eyes and mouth a little (a scar came with a darker
    # face): there only what clearly changed is taken over at all
    mixed = Image.composite(edited, source, clear if box else change_mask(source, edited))
    if m.get("pixel_art"):  # its own pixels are clean art already: a restyle pass only mottled flat colours
        result = Image.composite(mixed.quantize(palette=edit_palette(source, mixed, clear), dither=Image.Dither.NONE)
                                 .convert("RGB"), mixed, clear)
    else:
        result = restyle(agent, mixed, source, clear, staged, live)
    if box:
        whole.paste(result, box[:2])
        result = whole
    n = len(agent.get("edits", [])) + 1
    file = f"polish_{n}_{seed}.png"  # the seed keeps an undone and redone edit from hitting the browser's cache
    result.save(folder(agent["slug"]) / file)
    edit = {"file": f"soulforge/{agent['slug']}/{file}", "seed": seed, "instruction": instruction,
            "before": before, "seconds": round(time.time() - t, 1), "model": model, "changed": round(changed, 4), "focus": focus}
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
    # a limited palette kills the edit model's gradients and gloss, which no 16-bit art has. It is the base's
    # colours plus the edit's own: from the base's alone a yellow raincoat on a girl in green came out mud brown
    styled = styled.quantize(palette=edit_palette(source, picture, mask), dither=Image.Dither.NONE).convert("RGB")
    return Image.composite(styled, picture, mask)


def edit_palette(source: Image.Image, edited: Image.Image, mask: Image.Image) -> Image.Image:
    """48 colours of the base and 16 of the changed area, in one palette."""
    box = mask.point(lambda v: 255 if v > 128 else 0).getbbox() or (0, 0, *source.size)
    base = source.quantize(colors=48, method=Image.Quantize.MEDIANCUT).getpalette()[:48 * 3]
    new = edited.crop(box).quantize(colors=16, method=Image.Quantize.MEDIANCUT).getpalette()[:16 * 3]
    palette = Image.new("P", (1, 1))
    palette.putpalette(base + new + base[:3] * (256 - 64))
    return palette


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


def painted(picture: Image.Image, source: Image.Image | None) -> "np.ndarray":
    """Where a key edit really painted the key green. Green she wears herself (an apron, a green coat) is green in the
    key picture too but unchanged from her, so it's not keyed: it punched holes into Fern's apron (2026-11-01)."""
    import numpy as np
    mask = np.asarray(keyed(picture)) > 128
    if source is None:
        return mask
    from PIL import ImageFilter
    a = np.asarray(picture.convert("RGB")).astype(int)
    b = np.asarray(source.convert("RGB").resize(picture.size)).astype(int)
    changed = Image.fromarray(((np.abs(a - b).sum(2) > 70) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    own = np.asarray(keyed(source.convert("RGB").resize(picture.size))) > 128
    return mask & (~own | (np.asarray(changed) > 128))


def vote(pictures: list[Image.Image], source: Image.Image | None = None) -> Image.Image:
    """The key mask where most tries agree: a random extra in one try drops out."""
    import numpy as np
    stack = np.stack([painted(p, source) for p in pictures])
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
    figure = 255 - np.asarray(vote(raw["key_figure"], source), dtype=np.float32)
    hair = np.minimum(np.asarray(vote(raw["key_hair"], source), dtype=np.float32), figure)
    cloak = np.minimum(np.asarray(vote(raw["key_cloak"], source), dtype=np.float32), figure)
    # Klein greens one along with the other: the hair key took Morpheus's cloak, the cloak key took Glia's hair
    # (read as a hood). The smaller key is the one that stayed on its part, so it keeps the overlap.
    if (hair > 128).sum() <= (cloak > 128).sum():
        cloak = np.minimum(cloak, 255 - hair)
    else:
        hair = np.minimum(hair, 255 - cloak)
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


# the lids drawn on a close-up of the eyes: on the whole picture the edit model gave them eyeshadow and blobby lashes
LIDS = ("Close both of her eyes gently: each closed eye is one clean thin dark curved lash line on her skin, curving "
        "slightly downward like a calm sleeping anime girl. No eyeshadow, no makeup, no coloured lids, the skin above "
        "the lash line is her normal skin colour. Her eyebrows, hair, cheeks and mouth stay exactly the same.")


def lids(agent: dict, source: Image.Image, eyes, live=None) -> Image.Image:
    """Her closed eyes, from an edit of a close-up around them (see LIDS)."""
    ew = eyes[2] - eyes[0]
    cx, cy, r = (eyes[0] + eyes[2]) // 2, (eyes[1] + eyes[3]) // 2, int(ew * .75)
    W, H = source.size
    return close_up(agent, LIDS, source, (max(0, cx - r), max(0, cy - r), min(W, cx + r), min(H, cy + r)), live)


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
    # only the eye itself: above its top the edit model repainted hair and brows as forehead (orange patches on Glia)
    cl = {k: v for k, v in snap(shut, gxs, gys).items() if np.abs(v - base[k]).sum() > 24 and k[1] >= cy[y0] - 1}
    dark = pal[np.argmin(lum(pal))]
    top, bot = cy[y0], cy[min(y1, H - 1)]
    # a closed eye shows no iris or white: what the lid edit left of them below the lash line becomes skin
    skin = pal[np.argmin(((pal - np.array(skin_tone(source, eyes), float)) ** 2).sum(1))]
    shut_at = lambda k: cl.get(k, base[k])
    src_a = np.asarray(source.convert("RGB"))
    for gx in gxs:
        for gy in range(top, bot + 1):
            col = shut_at((gx, gy))
            if lum(col) > 150 and np.abs(col - skin).sum() > 70:
                cl[(gx, gy)] = skin
            elif lum(col) > 60:  # not lash: any bright speck of iris or white in the cell (the art is a little off the
                (x0_, x1_), (y0_, y1_) = span(cx, gx), span(cy, gy)  # grid) is painted over with a whole skin cell
                px = src_a[y0_:y1_, x0_:x1_].reshape(-1, 3).astype(float)
                if ((lum(px) > 120) & (np.abs(px - skin).sum(1) > 70)).any() and np.abs(col - skin).sum() <= 70:
                    cl[(gx, gy)] = skin
    # half: the closed lid's upper half over the open eye, its edge a dark line where the eye itself is
    mid = top + round((bot - top) * .6)  # the lashes fill the top third already: lower, or it barely looks closing
    half = {(gx, gy): shut_at((gx, gy)) for gx in gxs for gy in range(top, mid) if (gx, gy) in cl}
    for gx in gxs:
        if np.abs(base[(gx, mid)] - skin).sum() > 70 and lum(base[(gx, mid)]) > 60:
            half[(gx, mid)] = dark
    glances = {1: {}, -1: {}}  # to her left and to her right
    ex0, ex1, m = cx[x0], cx[min(x1, W - 1)], (cx[x0] + cx[min(x1, W - 1)]) // 2
    for a, b in ((ex0, m), (m + 1, ex1)):
        rows = list(range(top, bot + 1))
        L = {gy: np.mean([lum(base[(gx, gy)]) for gx in range(a, b + 1)]) for gy in rows}
        lash = min(rows[:len(rows) // 2 + 1], key=L.get)
        lower = [r for r in rows if r > lash + 1]
        lid = min(lower, key=L.get) if lower else bot
        for d, glance in glances.items():
            for gy in range(lash + 2, lid):  # the lash band stays put: shifted, it broke into specks (Glia)
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


def smirk_lift(source: Image.Image, mouth, lift_px: int = 1) -> Image.Image:
    """The smirk, drawn: her own lips with one corner lifted by one art pixel, rising from the middle of the mouth
    toward the corner. Only the lips' rows move; with more, the jaw line below got a step."""
    import numpy as np
    a = np.asarray(source)
    out = a.copy()
    x0, y0, x1, y1 = mouth
    mid = (x0 + x1) // 2
    for x in range(mid, min(a.shape[1], x1 + BLOCK)):
        lift = round(min(1.0, (x - mid) / max(1, (x1 - mid))) * BLOCK * lift_px)
        if lift:
            ys = slice(max(0, y0 - BLOCK), min(a.shape[0] - lift, y1 + BLOCK // 2))
            out[ys, x] = a[ys.start + lift:ys.stop + lift, x]
    return Image.fromarray(out)


def mouth_shapes(source: Image.Image, mouth, seam: int | None = None) -> dict:
    """Her mouth in more shapes, drawn from her own pixels on the art's grid, like the smirk: a smile and a frown
    (both corners lifted or dropped by one art pixel), and the mouth open one, two and three art pixels for talking
    and laughing, the lower lip and the chin under it moved down as an oval, the gap her own darkest lip colour,
    deepened, with a row of teeth in the widest one."""
    import numpy as np
    a = np.asarray(source)
    H, W = a.shape[:2]
    x0, y0, x1, y1 = mouth
    mid, hw = (x0 + x1) / 2, max(BLOCK, (x1 - x0) / 2)
    lum = a[y0:y1 + 1, x0:x1 + 1].astype(float) @ [.3, .59, .11]
    if seam is None:  # the line between the lips: their darkest row (a grin with teeth names its own, under them)
        seam = y0 + int(np.argmin(lum.mean(1)))
    lip = a[y0:y1 + 1, x0:x1 + 1].reshape(-1, 3)
    inside = (lip[np.argmin(lip.astype(float) @ [.3, .59, .11])] * .45).astype(np.uint8)
    teeth = np.array([236, 230, 226], np.uint8)
    shapes = {}
    for name, lift in (("smile", 1), ("frown", -1)):
        out = a.copy()
        for x in range(int(x0 - BLOCK), int(x1 + BLOCK)):
            if not 0 <= x < W:
                continue
            d = round(lift * min(1.0, abs(x - mid) / hw) ** 2 * BLOCK)  # the corners move, the middle stays
            if d:
                ys = np.arange(max(0, y0 - BLOCK), min(H, y1 + BLOCK))
                src = np.clip(ys + d, 0, H - 1)
                out[ys, x] = a[src, x]
        shapes[name] = out
    # rounded: the gap tapers toward the corners, decided per art pixel; a tongue tone along its floor once it's
    # wide; drawn teeth only for a mouth that shows none already (a grin's teeth stay hers, the gap opens under them)
    tongue = (np.median(lip, 0) * .62).astype(np.uint8)
    has_teeth = ((lip.astype(float) @ [.3, .59, .11]) > 200).mean() > .04
    for n in (1, 2, 3):
        out = a.copy()
        for x in range(int(x0 - BLOCK // 2), int(x1 + BLOCK // 2)):
            if not 0 <= x < W:
                continue
            u = ((x // BLOCK) * BLOCK + BLOCK / 2 - mid) / hw
            if abs(u) >= 1:
                continue
            drop = int(n * (1 - u * u) ** .5 + .35) * BLOCK
            if not drop:
                continue
            below = np.arange(seam + 1, min(H - drop, y1 + 2 * BLOCK + 1))  # the lower lip and some chin move down
            out[below + drop, x] = a[below, x]
            out[seam + 1:seam + 1 + drop, x] = inside
            if n >= 2 and drop >= 2 * BLOCK and abs(u) < .55:
                out[seam + 1 + drop - BLOCK:seam + 1 + drop, x] = tongue
            if n == 3 and not has_teeth and abs(u) < .7:
                out[seam + 1:seam + 1 + BLOCK, x] = teeth
        shapes[f"open{n}"] = out
    return {k: Image.fromarray(v) for k, v in shapes.items()}


# Arm poses: Klein draws her with a raised arm; only the arm's region is kept, on the art's grid. Opaque pixels
# are her posed arm, half-transparent ones are where the posed picture shows background (where her arm hung before,
# now empty): the page hides her there, so on a scene the backdrop shows through.
# each arm comes up in front of her body, its forearm in her own sleeve: raised from out of frame, the hand floated
# on a cuff (Glia, 2026-11-01). One hand only: asked to wave, the edit model often raised both.
POSES = {
    "wave": "She waves hello with her right hand: her elbow is bent sharply at a right angle and points out to the "
            "side at chest height, the forearm in her own sleeve and cuff goes straight up from the elbow, open hand "
            "beside her face with all five fingers spread, palm toward the viewer. Only one hand is raised; the "
            "other arm stays down at her side.",
    "chin": "She thinks: her right elbow is bent sharply and rests at chest height in front of her body, the forearm "
            "in her own sleeve and cuff goes up to her chin, her curled index finger rests against her chin, "
            "pondering. Her other arm stays down at her side.",
    "heart": "She lays her right hand flat over her heart: her elbow is bent sharply and points out to the side at "
             "chest height, the forearm in her own sleeve and cuff runs level across her chest, a tender, touched "
             "gesture. Her other arm stays down at her side.",
}
POSE_TRIES = 4  # the best of four by pose_score: the edit model often raises both hands or swings the elbow out
IDLE_STEPS += [f"{k} {n + 1}" for k in POSES for n in range(POSE_TRIES)]


def pose_score(overlay: Image.Image | None, eyes) -> float:
    """Lower is better: holes opened in the middle of her torso (where no arm hangs) mean Klein redrew her there;
    a second raised hand counts from arm_pose (Glia's wave). An elbow out beside her is the bend, not a fault."""
    import numpy as np
    if overlay is None:
        return float("inf")
    a = np.asarray(overlay)[..., 3]
    ew = eyes[2] - eyes[0]
    mid = (eyes[0] + eyes[2]) // 2
    band = a[:, max(0, mid - int(ew * .8)):mid + int(ew * .8)]
    score = float((band == 128).sum()) / max(1, (a == 255).sum())
    return score + max(0, overlay.info.get("hands", 1) - 1)


# where each pose's hand belongs, from the face: its top in eye widths below the eyes' bottom. A drawing with the
# hand elsewhere is another gesture (Glia's thinking came out as arms crossed at her waist)
POSE_AIM = {"wave": (-.6, .3),  # higher, the fingers left the frame
            "chin": (.1, .9), "heart": (.7, 1.8)}


def pose_aim(overlay: Image.Image | None, piece: str, eyes) -> float:
    """0 when the hand is where the pose puts it, else 1 (added to pose_score)."""
    if overlay is None or piece not in POSE_AIM:
        return 0.0
    top = overlay.info.get("hand_top")
    if top is None:
        return 1.0
    lo, hi = POSE_AIM[piece]
    u = (top - eyes[3]) / (eyes[2] - eyes[0])
    return 0.0 if lo <= u <= hi else 1.0


def arm_pose(source: Image.Image, posed: Image.Image, plate: Image.Image, eyes, mouth,
             figure: Image.Image | None = None) -> Image.Image | None:
    """The posed arm as an overlay (see POSES), or None when Klein's change isn't one clear arm."""
    import cv2
    import numpy as np
    W, H = source.size
    grid = W // BLOCK
    gh = round(grid * H / W)
    raw = posed
    posed = posed.resize((grid, gh), Image.Resampling.BOX).resize((W, H), Image.Resampling.NEAREST)
    a, b = np.asarray(source).astype(int), np.asarray(posed).astype(int)
    diff = (np.abs(a - b).sum(2) > 90).astype(np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    diff = cv2.dilate(diff, np.ones((15, 15), np.uint8))
    ew = eyes[2] - eyes[0]
    face = np.zeros_like(diff)  # the face stays hers: Klein redraws its smile too
    cv2.ellipse(face, ((eyes[0] + eyes[2]) // 2, (eyes[1] + mouth[3]) // 2),
                (int(ew * .62), int((mouth[3] - eyes[1]) * .8 + 10)), 0, 0, 360, 1, -1)
    face[mouth[3] + BLOCK:] = 0  # below the lips is the hand's (on the chin), not the face's
    # the ellipse reaches over hair and earrings, where a waving hand's fingers come (Glia's wave lost them): new
    # skin where she had none is the hand's
    cy, cx = (eyes[3] + mouth[1]) // 2, (eyes[0] + eyes[2]) // 2
    skin0 = np.median(a[cy - 6:cy + 6, cx - 10:cx + 10].reshape(-1, 3), axis=0)
    finger = (np.abs(b - skin0).sum(2) < 90) & (np.abs(a - skin0).sum(2) >= 90)
    face[cv2.morphologyEx(finger.astype(np.uint8), cv2.MORPH_OPEN, np.ones((BLOCK, BLOCK), np.uint8)) > 0] = 0
    diff[face > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, 4]))
    if not .015 < stats[k, 4] / (W * H) < .25:
        return None
    main = (lab == k).astype(np.uint8)
    # the arm: everything of Klein's on its side of a seam. Grown from the change, the sleeve came out ragged or
    # ended below the cuff (Glia, 2026-11-01): a dark sleeve over a dark coat or a night sky barely differs. The
    # seam runs down from the hand where Klein's picture and hers agree best, so it shows nowhere.
    cell = lambda im: np.asarray(im.convert("RGB").resize((grid, gh), Image.Resampling.BOX)).astype(int)
    # Klein's outlines are thinner than an art pixel: averaged into a cell they faded into the coat (Glia's
    # sleeve and fingers). A cell a quarter line-dark is line.
    fine = np.asarray(raw.convert("RGB").resize((grid * BLOCK, gh * BLOCK), Image.Resampling.LANCZOS)).astype(int)
    fine = fine.reshape(gh, BLOCK, grid, BLOCK, 3).transpose(0, 2, 1, 3, 4).reshape(gh, grid, BLOCK * BLOCK, 3)
    lum = fine.sum(3)
    dark = lum < 110
    mean = fine.mean(2)
    line = np.where(dark[..., None], fine, 0).sum(2) / np.maximum(1, dark.sum(2))[..., None]
    cellc = np.where((dark.mean(2) >= .25)[..., None], line, mean).clip(0, 255).astype(np.uint8)
    pc, sc = cellc.astype(int), cell(source)
    d = np.abs(pc - sc).sum(2)
    facec = np.asarray(Image.fromarray(face * 255).resize((grid, gh), Image.Resampling.NEAREST)) > 0
    core = cv2.resize(main, (grid, gh), interpolation=cv2.INTER_NEAREST) > 0
    mid = (eyes[0] + eyes[2]) / 2 * grid / W
    flip = np.nonzero(core)[1].mean() > mid  # the arm on her left: mirrored, worked out as a right one
    fl = (lambda m: m[:, ::-1]) if flip else (lambda m: m)
    d, core, facec = fl(d), fl(core), fl(facec)
    mid = grid - mid if flip else mid
    top = int(np.nonzero(core)[0].min())
    hi = min(grid - 1, int(mid + ew * grid / W * .9))
    acc = np.full((gh, grid), np.inf)
    back = np.zeros((gh, grid), int)
    for r in range(top, gh):
        xs_ = np.nonzero(core[r])[0]
        lo = int(xs_.max()) + 1 if len(xs_) else 0
        row = np.full(grid, np.inf)
        # the + 4: a straight seam among equal ones; and each cell away from the arm costs, or the seam wandered
        # over to her hair and took her whole head along (Glia's raised wave)
        row[lo:hi + 1] = d[r, lo:hi + 1] + 4 + 3 * (np.arange(lo, hi + 1) - lo)
        if r == top:
            acc[r] = row
            continue
        prev = acc[r - 1]
        cand = np.stack([np.roll(prev, 1), prev, np.roll(prev, -1)])
        cand[0, 0] = cand[2, -1] = np.inf
        k = cand.argmin(0)
        acc[r] = row + cand[k, np.arange(grid)]
        back[r] = np.arange(grid) + k - 1
    regc = np.zeros((gh, grid), bool)
    c = int(acc[gh - 1].argmin())
    for r in range(gh - 1, top - 1, -1):
        regc[r, :c] = True
        c = back[r, c] if r > top else c
    regc &= ~facec
    # no elbow hangs lower than a forearm below the hand and cuff: further down Klein only reshaded her skirt
    # no seam below: her old arm hung to the frame's bottom, and anything of it left under the new elbow read as an
    # arm running straight down (Glia, 2026-11-01); Klein's coat replaces all of it
    regc = fl(regc)
    # the backdrop: sky-like in Klein's picture and open to the sky at the frame's edge. A sleeve as dark as the
    # sky is walled off by its own outline and stays arm
    skyish = np.abs(pc - cell(plate)).sum(2) < 36
    reach = np.pad(skyish.astype(np.uint8), 1, constant_values=1)
    cv2.floodFill(reach, None, (0, 0), 2)
    opened = (reach[1:-1, 1:-1] == 2) & skyish
    # the cells along the old arm's edge mix sky with its outline: still backdrop, or they stayed as a dotted ghost
    # of the arm in her scene
    edge_ = cv2.dilate(opened.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    opened |= edge_ & (np.abs(pc - cell(plate)).sum(2) < 110) & (np.abs(pc - sc).sum(2) > 30)
    up = lambda m: cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    # where Klein's picture is hers anyway she keeps her own pixels: requantized, her coat came out grainy
    # all of Klein's pixels on the arm's side: her old forearm hung there, dark like the coat Klein drew over it,
    # and kept as "unchanged" it ran on below the bent elbow (Glia's wave and chin looked straight)
    region = up(regc) & (face == 0)
    empty = up(opened)
    out = np.zeros((H, W, 4), np.uint8)
    # her own palette: the edit model paints smooth shading, a hand of 30 skin tones on flat pixel art (Glia's wave)
    pal = source.resize((grid, gh), Image.Resampling.BOX).quantize(colors=48, method=Image.Quantize.FASTOCTREE)
    q = np.array(Image.fromarray(cellc, "RGB").quantize(palette=pal, dither=Image.Dither.NONE))
    # Klein's soft shading lands on her flat palette as lone specks (Glia's heart sleeve): a cell matching none of
    # its four neighbours takes the commonest colour around it
    pad = np.pad(q, 1, mode="edge")
    alone = np.ones(q.shape, bool)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        alone &= pad[dy:dy + gh, dx:dx + grid] != q
    lut = np.array(pal.getpalette()[:768]).reshape(-1, 3)
    for y, x in zip(*np.nonzero(alone)):
        around = pad[y:y + 3, x:x + 3].ravel()
        m = np.bincount(np.delete(around, 4)).argmax()
        if np.abs(lut[m] - lut[q[y, x]]).sum() < 70:  # a shade off; a diagonal outline is a line, not a speck
            q[y, x] = m
    # and soft shading lands as grain over larger areas (her skirt under the new elbow): a cell a shade off the
    # commonest colour around it takes that colour, twice over
    for _ in range(2):
        pad = np.pad(q, 1, mode="edge")
        win = np.stack([pad[dy:dy + gh, dx:dx + grid] for dy in range(3) for dx in range(3)], -1)
        mode = np.array([np.bincount(w).argmax() for w in win.reshape(-1, 9)]).reshape(gh, grid)
        near_ = np.abs(lut[mode] - lut[q]).sum(2) < 45
        q = np.where(near_, mode, q).astype(q.dtype)
    cq = Image.frombytes("P", (grid, gh), np.ascontiguousarray(q).tobytes()); cq.putpalette(pal.getpalette())
    out[..., :3] = np.asarray(cq.convert("RGB").resize((W, H), Image.Resampling.NEAREST))
    if figure is not None:
        # open backdrop beside her was backdrop before too: nothing to draw there. Inside her outline it's where
        # her arm hung, and the plate shows through
        body = np.asarray(figure.convert("RGB").split()[0]) > 128
        region &= ~(empty & ~body)
        raised = np.nonzero(main)[0].min() < eyes[1] - ew * .5  # a hand above her head; one beside her face lies over her hair
        # a hand raised beside her head (the wave): the arm is outside her old outline, and inside it she keeps
        # her own hair and coat, only her old hanging arm opening to the sky. The seam took her whole head along
        if raised:
            region &= ~body | empty
        # beside her Klein's sky is a shade off the plate's: still sky, unless walled in by the arm's outline
        loose = np.pad((np.abs(pc - cell(plate)).sum(2) < 60).astype(np.uint8), 1, constant_values=1)
        cv2.floodFill(loose, None, (0, 0), 2)
        if not raised:  # a raised sleeve against the night sky is within 60 of it, and was lost
            region &= ~(up(loose[1:-1, 1:-1] == 2) & ~body)
        empty &= body
    out[..., 3] = np.where(region, np.where(empty, 128, 255), 0)
    # hands: skin-coloured blobs of the arm. Asked to wave, the edit model often raises both (pose_score)
    cy, cx = (eyes[3] + mouth[1]) // 2, (eyes[0] + eyes[2]) // 2
    skin = np.median(a[cy - 6:cy + 6, cx - 10:cx + 10].reshape(-1, 3), axis=0)
    sk = ((np.abs(out[..., :3].astype(int) - skin).sum(2) < 70) & (out[..., 3] == 255)
          & (np.abs(a - skin).sum(2) >= 70)).astype(np.uint8)  # new skin: her neck was there before
    sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, np.ones((BLOCK * 2, BLOCK * 2), np.uint8))
    sk = cv2.morphologyEx(sk, cv2.MORPH_CLOSE, np.ones((int(ew * .3), int(ew * .3)), np.uint8))  # one hand's fingers
    ns, _, st, _ = cv2.connectedComponentsWithStats(sk, connectivity=8)
    im = Image.fromarray(out, "RGBA")
    im.info["hands"] = int((st[1:, 4] > (ew * .22) ** 2).sum())
    if ns > 1:  # the biggest new skin: the hand, its top for pose_aim
        im.info["hand_top"] = int(st[1 + int(np.argmax(st[1:, 4])), 1])
    # touching hands merge into one blob (Glia's chin, the other hand on her forearm); each brings its own cuff
    o3 = out[..., :3].astype(int)
    cuff = ((o3.min(2) > 190) & (o3.max(2) - o3.min(2) < 28) & (out[..., 3] == 255) & (a.min(2) <= 190)).astype(np.uint8)
    cuff = cv2.morphologyEx(cuff, cv2.MORPH_OPEN, np.ones((BLOCK * 2, BLOCK * 2), np.uint8))
    nc, _, stc, _ = cv2.connectedComponentsWithStats(cuff, connectivity=8)
    im.info["cuffs"] = int((stc[1:, 4] > (ew * .14) ** 2).sum())
    im.info["hands"] = max(im.info["hands"], im.info["cuffs"])
    return im


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


def find_eyes(source: Image.Image, shut: Image.Image, window: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Both eyes from the closed-eyes edit: the strongest change, then a second one of eye size level with it. blob()
    alone took the stronger eye plus an earring on Glia, whose eyes sit far apart (2026-11-01): one eye blinked, the
    blush landed on her nose."""
    import numpy as np
    from PIL import ImageChops, ImageFilter
    heat = np.asarray(ImageChops.difference(source, shut).convert("L").crop(window).filter(ImageFilter.GaussianBlur(8)),
                      dtype=np.float32)
    boxes = []
    for _ in range(6):
        if len(boxes) == 2:
            break
        peak = heat.max()
        if peak < 4 or (boxes and peak < boxes[0][4] * 0.3):
            break
        py, px = np.unravel_index(int(heat.argmax()), heat.shape)
        y0, y1, x0, x1 = max(0, py - 45), py + 45, max(0, px - 75), px + 75
        hot = np.argwhere(heat[y0:y1, x0:x1] > peak * 0.4)
        bx0, bx1 = x0 + hot[:, 1].min(), x0 + hot[:, 1].max()
        # the closed lid only changes the upper eye: its height comes from the picture, the rows that aren't skin
        rgb = np.asarray(source.crop(window).convert("RGB"), dtype=np.int32)
        skin = np.median(rgb[min(rgb.shape[0] - 1, py + 70), bx0:bx1 + 1], axis=0)
        span = rgb[max(0, py - 60):py + 70, bx0:bx1 + 1]
        eye = (np.abs(span - skin).sum(2) > 90).mean(1) > 0.3  # per row from py - 60: not skin, so still the eye
        by0, by1 = y0 + int(hot[:, 0].min()), y0 + int(hot[:, 0].max())  # the lid's top is right: above it, hair
        k = py - max(0, py - 60)
        while k < len(eye) and eye[k]:  # downward from the lid until the cheek
            k += 1
        by1 = max(by1, max(0, py - 60) + k)
        box = (bx0, by0, bx1, by1, peak)
        heat[max(0, py - 60):py + 60, max(0, px - 85):px + 85] = 0
        if boxes:  # the other eye: level with the first and a face's width at most away, else it's drift
            if abs((box[1] + box[3]) / 2 - (boxes[0][1] + boxes[0][3]) / 2) > 40 or not 50 < abs(px - (boxes[0][0] + boxes[0][2]) / 2) < 300:
                heat[max(0, py - 60):py + 60, max(0, px - 85):px + 85] = 0
                continue
        boxes.append(box)
    if len(boxes) < 2:
        return blob(source, shut, window)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return (window[0] + int(x0), window[1] + int(y0), window[0] + int(x1), window[1] + int(y1))


def soft_box(size: tuple[int, int], box: tuple[int, int, int, int], pad: int = 8) -> Image.Image:
    from PIL import ImageFilter
    mask = Image.new("L", size, 0)
    mask.paste(255, (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
    return mask.filter(ImageFilter.GaussianBlur(3))


def blush(source: Image.Image, eyes: tuple[int, int, int, int], strength: float = 1.0) -> Image.Image:
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
            d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=min(255, round(fill * strength)))
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


def changed_overlay(source: Image.Image, picture: Image.Image) -> Image.Image:
    """Only what the drawing changed, as an overlay."""
    from PIL import ImageChops
    out = picture.convert("RGBA")
    out.putalpha(ImageChops.difference(source, picture).convert("L").point(lambda v: 255 if v else 0))
    return out


# the pieces of the idle set that can be drawn again alone; the rig and plate can't (everything is placed on them)
PIECES = [*POSES, "blink", "blush", "smirk", "motion"]
LEVELS = {"blush": {"less": 0.6, "same": 1.0, "more": 1.45}, "smirk": {"less": 1, "same": 1, "more": 2}}


def redo_piece(agent: dict, piece: str, hint: str = "", level: str = "same", live=None) -> list[str]:
    """One piece of the idle set drawn again (an arm pose, the blink) or set anew (blush strength, smirk lift); the
    rest stays as it is. Returns the frame names that changed."""
    idle = agent.get("idle") or {}
    if piece not in PIECES:
        raise ValueError(f"Unknown piece: {piece}")
    if not idle.get("frames") or idle.get("base") != agent["base"]["file"]:
        raise ValueError("Her frames belong to an older picture: wake up the new look first.")
    source = Image.open(morpheus.OUTPUT / agent["base"]["file"]).convert("RGB")
    eyes, mouth = tuple(idle["eyes"]), tuple(idle["mouth"])
    if piece in POSES:
        plate = Image.open(morpheus.OUTPUT / idle["plate"])
        instruction = POSES[piece] + (f" {hint.strip()}" if hint.strip() else "") + KEEP
        figure = Image.open(morpheus.OUTPUT / idle["rig"])
        tries = [arm_pose(source, klein(agent, instruction, source, live), plate, eyes, mouth, figure) for _ in range(POSE_TRIES)]
        best = min(tries, key=lambda o: pose_score(o, eyes) + pose_aim(o, piece, eyes))
        if best is None:
            raise ValueError(f"The {piece} pose didn't come out as one clear arm; try again.")
        pictures = {piece: best}
    elif piece == "motion":  # breathing, hair and cloak: the rig and her cut-out, from new keys; the plate stays
        raw = {k: [expression(agent, k, source, live) for _ in range(TRIES)] for k in ("key_figure", "key_hair", "key_cloak")}
        stamp = int(time.time())
        figure = rig(raw, source, eyes, mouth)
        figure.save(folder(agent["slug"]) / f"idle_rig_{stamp}.png")
        cut = source.convert("RGBA")
        cut.putalpha(figure.split()[0].point(lambda v: 255 if v > 128 else 0))
        cut.save(folder(agent["slug"]) / f"scene_cut_{stamp}.png")
        scene = agent.get("scene") or {}
        for old in (idle.get("rig"), scene.get("cut")):
            if old:
                (morpheus.OUTPUT / old).unlink(missing_ok=True)
        idle["rig"] = f"soulforge/{agent['slug']}/idle_rig_{stamp}.png"
        if scene:
            scene["cut"] = f"soulforge/{agent['slug']}/scene_cut_{stamp}.png"
        save(agent)
        return ["rig"]
    elif piece == "blink":
        shut = expression(agent, "closed", source, live)
        pictures = pixel_eyes(source, changed_overlay(source, eye_frame(source, lids(agent, source, eyes, live), eye_pair(source, shut, eyes), True)), eyes)
    elif piece == "blush":
        idle["blush"] = min(3.0, max(0.3, LEVELS["blush"].get(level, 1.0) * idle.get("blush", 1.0)))
        pictures = {"blush": blush(source, eyes, idle["blush"])}
    else:
        idle["smirk"] = LEVELS["smirk"].get(level, 1)
        pictures = {"smirk": changed_overlay(source, smirk_lift(source, mouth, idle["smirk"]))}
    stamp = int(time.time())
    for name, picture in pictures.items():
        file = f"idle_{name}_{stamp}.png"
        picture.save(folder(agent["slug"]) / file)
        old = idle["frames"].get(name)
        idle["frames"][name] = f"soulforge/{agent['slug']}/{file}"
        if old and old != idle["frames"][name]:
            (morpheus.OUTPUT / old).unlink(missing_ok=True)
    save(agent)
    return list(pictures)


def idle(agent: dict, progress=None, live=None) -> dict:
    """The idle set of the base image: blink frames (half and closed), a smirk and a blush as overlays, and the
    body rig with its clean background plate."""
    from PIL import ImageChops
    source = Image.open(morpheus.OUTPUT / agent["base"]["file"]).convert("RGB")
    W, H = source.size
    if progress:
        progress(0, "find the eyes")
    shut = expression(agent, "closed", source, live)
    eyes = find_eyes(source, shut, (0, 0, W, H // 2))  # the head is in the upper third
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

    overlay = lambda picture: changed_overlay(source, picture)
    pictures = {**pixel_eyes(source, overlay(eye_frame(source, lids(agent, source, eyes, live), pair, True)), eyes),
                "smirk": overlay(smirk_lift(source, mouth)), "blush": blush(source, eyes),
                **{k: overlay(v) for k, v in mouth_shapes(source, mouth, (agent.get("idle") or {}).get("seam")).items()}}
    stamp = int(time.time())
    frames = {}
    for name, picture in pictures.items():
        file = f"idle_{name}_{stamp}.png"
        picture.save(folder(agent["slug"]) / file)
        frames[name] = f"soulforge/{agent['slug']}/{file}"
    scene = agent.get("scene") or {}
    if not scene.get("motif"):  # the motif is about her, not this picture: it outlives a new look
        if progress:
            progress(IDLE_STEPS.index("her motif"), "her motif")
        motif(agent, live).save(folder(agent["slug"]) / f"scene_motif_{stamp}.png")
        scene["motif"] = f"soulforge/{agent['slug']}/scene_motif_{stamp}.png"
    if agent.get("world") and not scene.get("world"):  # like the motif, her world outlives a new look
        if progress:
            progress(IDLE_STEPS.index("her world"), "her world")
        scene["world"] = world(agent, live)
    if progress:
        progress(IDLE_STEPS.index("rig"), "rig")
    figure = rig(raw, source, eyes, mouth)
    figure.save(folder(agent["slug"]) / f"idle_rig_{stamp}.png")
    cut = source.convert("RGBA")
    cut.putalpha(figure.split()[0].point(lambda v: 255 if v > 128 else 0))
    cut.save(folder(agent["slug"]) / f"scene_cut_{stamp}.png")
    if scene.get("cut"):
        (morpheus.OUTPUT / scene["cut"]).unlink(missing_ok=True)
    agent["scene"] = {**scene, "base": agent["base"]["file"], "cut": f"soulforge/{agent['slug']}/scene_cut_{stamp}.png"}
    best_plate(raw["plate"], vote(raw["key_figure"], source).point(lambda v: 255 - v)).save(
        folder(agent["slug"]) / f"idle_plate_{stamp}.png")
    plate = Image.open(folder(agent["slug"]) / f"idle_plate_{stamp}.png")
    for name, instruction in POSES.items():
        tries = []
        for n in range(POSE_TRIES):
            if progress:
                progress(IDLE_STEPS.index(f"{name} {n + 1}"), f"{name} {n + 1}")
            tries.append(arm_pose(source, klein(agent, instruction + KEEP, source, live), plate, eyes, mouth, figure))
        best = min(tries, key=lambda o: pose_score(o, eyes) + pose_aim(o, name, eyes))
        if best is not None:
            best.save(folder(agent["slug"]) / f"idle_{name}_{stamp}.png")
            frames[name] = f"soulforge/{agent['slug']}/idle_{name}_{stamp}.png"
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
