# H3-Image reference cap (Glia seed h3_image_edit_speed). WanGP resizes every reference to the output's pixel area
# (fit_into_canvas_image_refs 0), so each ref costs as many tokens as the output. This scales refs down afterwards to a short
# edge of var["ref_cap"] (sides multiples of 32); var["ref_cap_skip_first"] keeps picture 1 (the edited image) as it was.
import math
from PIL import Image as _Image

_cap = int(var.get("ref_cap", 1024)); _skip = int(bool(var.get("ref_cap_skip_first")))
_orig_rrb = wgp.resize_and_remove_background


def _capped(img):
    if not isinstance(img, _Image.Image) or min(img.size) <= _cap:
        return img
    s = _cap / min(img.size)
    w, h = (max(32, round(v * s / 32) * 32) for v in img.size)
    return img.resize((w, h), _Image.Resampling.LANCZOS)


def _rrb(img_list, *a, **k):
    out = _orig_rrb(img_list, *a, **k)
    imgs, masks = out if isinstance(out, tuple) else (out, None)
    if len(img_list) > 1 or not _skip:  # the K path calls this with picture 1 alone: keep it when skip_first
        imgs = [im if i < _skip else _capped(im) for i, im in enumerate(imgs)]
    print("## REFCAP", [getattr(im, "size", None) for im in imgs], flush=True)
    return (imgs, masks) if isinstance(out, tuple) else imgs


wgp.resize_and_remove_background = _rrb
