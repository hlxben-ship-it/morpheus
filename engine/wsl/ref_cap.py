"""H3-Image reference size cap (Glia seed h3_image_edit_speed). Pure PIL, testable on Windows.

WanGP scales every H3-Image reference to the output's pixel area (model def fit_into_canvas_image_refs 0), so each reference costs as
many transformer tokens as the output. With a cap the worker scales them down afterwards to that short edge (sides multiples of 32).
Bench 1664x2496, preload on: 2 refs 130 s -> 80 s warm, 1 ref 76 s -> 55 s; the picture changes (less reference detail), so it is opt-in.
"""
from PIL import Image

CHOICES = {"full": 0, "1024": 1024, "1280": 1280}  # the Playground param "ref_size" -> short edge (0 = as WanGP sizes it)


def capped(img, cap: int):
    """img scaled down so its short edge is ``cap`` (sides rounded to 32); unchanged when cap is 0, img is no PIL image or already small."""
    if not cap or not isinstance(img, Image.Image) or min(img.size) <= cap:
        return img
    s = cap / min(img.size)
    w, h = (max(32, round(v * s / 32) * 32) for v in img.size)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def wrap(resize_fn, get_cap):
    """resize_fn (wgp.resize_and_remove_background) with every returned image capped at get_cap() at call time."""
    def wrapped(*a, **k):
        out = resize_fn(*a, **k)
        cap = get_cap()
        if not cap:
            return out
        if isinstance(out, tuple):
            return ([capped(im, cap) for im in out[0]],) + tuple(out[1:])
        return [capped(im, cap) for im in out]
    return wrapped
