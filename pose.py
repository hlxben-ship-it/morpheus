"""Pose of the people in a picture: an OpenPose skeleton image for the ControlNet, and their face and hand boxes.

Runs in .venv-faceswap (rtmlib + onnxruntime, on the CPU: one picture takes a few seconds), not in the ComfyUI venv;
yumeko.py starts it as a subprocess per picture.

    pose.py IMAGE SKELETON.png            every person, where they stand (keep the pose)
    pose.py IMAGE SKELETON.png --fit x0,y0,x1,y1 --canvas W,H [--onto PANEL]
                                          the largest person only, on a W x H canvas, in that box: with --onto at the size
                                          and neck of the character standing there in PANEL, else scaled to fill the box
                                          (another picture's pose, put where the masked character stands)

The last line printed is JSON (rtmlib prints its own lines before it):{"people": [{"box": [x0,y0,x1,y1], "face": [...] | null, "hands": [[...], ...]}]}, in the
coordinates of the skeleton image.
"""
import argparse
import json
import sys

import cv2
import numpy as np
import onnxruntime
from rtmlib import Wholebody, draw_skeleton

onnxruntime.set_default_logger_severity(3)  # the models' "unused initializer" warnings are noise

THR = 0.4  # a keypoint below this is a guess; drawn, it would bend a limb to nowhere
BODY, FACE, LEFT, RIGHT = slice(0, 18), slice(24, 92), slice(92, 113), slice(113, 134)  # OpenPose-134 layout


def box_of(points: np.ndarray, scores: np.ndarray, least: int, pad: float, size) -> list | None:
    """The box around the confident points, padded by a share of its longer side; None when too few are sure."""
    ok = points[scores > THR]
    if len(ok) < least:
        return None
    x0, y0, x1, y1 = ok[:, 0].min(), ok[:, 1].min(), ok[:, 0].max(), ok[:, 1].max()
    p = pad * max(x1 - x0, y1 - y0, 8)
    return [int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(size[0], x1 + p)), int(min(size[1], y1 + p))]


def torso(k: np.ndarray, s: np.ndarray):
    """(neck point, a body length) of one person: neck to mid-hip, or the shoulder width when the hips are unsure. None without a neck."""
    if s[1] <= THR:
        return None
    hips = [k[i] for i in (8, 11) if s[i] > THR]
    if hips:
        return k[1], max(float(np.linalg.norm(np.mean(hips, axis=0) - k[1])), 1.0)
    if s[2] > THR and s[5] > THR:
        return k[1], max(float(np.linalg.norm(k[2] - k[5])) * 1.5, 1.0)  # a torso is about one and a half shoulder widths
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("skeleton")
    ap.add_argument("--fit")
    ap.add_argument("--canvas")
    ap.add_argument("--onto")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        print(json.dumps({"error": "That is no picture."}))
        return 1
    model = Wholebody(mode="balanced", to_openpose=True, backend="onnxruntime", device="cpu")
    kp, sc = model(img)
    keep = []
    for i in range(len(kp)):
        if (sc[i][BODY] > THR).sum() < 4:  # a detection without a body is no person
            continue
        # the detector sometimes finds one person twice: the same body points again are dropped
        if any(np.abs(kp[i][BODY] - kp[j][BODY]).mean() < 0.03 * max(img.shape[:2]) for j in keep):
            continue
        keep.append(i)
    kp, sc = kp[keep], sc[keep]
    H, W = img.shape[:2]
    if a.fit and len(kp):
        # the largest person, by the box of their sure body points
        def area(i):
            ok = kp[i][BODY][sc[i][BODY] > THR]
            return (ok[:, 0].max() - ok[:, 0].min()) * (ok[:, 1].max() - ok[:, 1].min())
        i = max(range(len(kp)), key=area)
        kp, sc = kp[i:i + 1].copy(), sc[i:i + 1]
        ok = kp[0][sc[0] > THR]
        sx0, sy0, sx1, sy1 = ok[:, 0].min(), ok[:, 1].min(), ok[:, 0].max(), ok[:, 1].max()
        x0, y0, x1, y1 = (float(v) for v in a.fit.split(","))
        mine, hers = torso(kp[0], sc[0]), None
        if a.onto and mine:
            # the character who stands in the box now: the new pose gets her size and hangs from her neck, so she
            # doesn't grow to fill the mask (a pose with a hidden leg stretched to the mask's height made a giant)
            okp, osc = model(cv2.imread(a.onto))
            inside = [j for j in range(len(okp)) if torso(okp[j], osc[j]) and x0 <= torso(okp[j], osc[j])[0][0] <= x1 and y0 <= torso(okp[j], osc[j])[0][1] <= y1]
            hers = max((torso(okp[j], osc[j]) for j in inside), key=lambda t: t[1], default=None)
        if hers:
            s = hers[1] / mine[1]
            kp[0] = (kp[0] - mine[0]) * s + hers[0]
        else:
            # centred in the box: the figure keeps its proportions, so a sitting pose in a standing mask stays a sitting pose
            s = min((x1 - x0) / max(sx1 - sx0, 1), (y1 - y0) / max(sy1 - sy0, 1))
            kp[0, :, 0] = (kp[0, :, 0] - (sx0 + sx1) / 2) * s + (x0 + x1) / 2
            kp[0, :, 1] = (kp[0, :, 1] - (sy0 + sy1) / 2) * s + (y0 + y1) / 2
    if a.canvas:
        W, H = (int(v) for v in a.canvas.split(","))
    sc_draw = sc.copy()
    sc_draw[:, FACE] = 0  # no face dots: they pin the expression, and anime faces put them in the wrong places
    canvas = np.zeros((H, W, 3), np.uint8)
    if len(kp):
        canvas = draw_skeleton(canvas, kp, sc_draw, openpose_skeleton=True, kpt_thr=THR, radius=4, line_width=4)
    cv2.imwrite(a.skeleton, canvas)
    people = []
    for k, s in zip(kp, sc):
        people.append({"box": box_of(k, s, 4, 0.05, (W, H)), "face": box_of(k[FACE], s[FACE], 20, 0.45, (W, H)),
                       "hands": [b for b in (box_of(k[LEFT], s[LEFT], 8, 0.4, (W, H)), box_of(k[RIGHT], s[RIGHT], 8, 0.4, (W, H))) if b]})
    print(json.dumps({"people": people}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
