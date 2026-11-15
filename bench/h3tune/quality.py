"""Quality gate of the H3 bench: a frame strip to look at, and numbers against a reference clip (same clip, same seed).
ssim: structure vs the reference (1 = same clip; sparse attention changes motion, so well below 1 is normal there).
sharp / sharp_ratio: Laplacian variance (detail); motion / motion_ratio: mean frame-to-frame change. A drop in either is the alarm."""
import cv2, numpy as np
from skimage.metrics import structural_similarity


def frames(path, n=12):
    cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); out = []
    for i in np.linspace(1, total - 2, n).astype(int):
        pair = []
        for j in (i, i + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(j)); ok, f = cap.read()
            if ok: pair.append(f)
        if len(pair) == 2: out.append(pair)
    cap.release(); return out


def stats(fr):
    g = [[cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY) for f in p] for p in fr]
    sharp = float(np.mean([cv2.Laplacian(a, cv2.CV_64F).var() for a, _ in g]))
    motion = float(np.mean([np.abs(a.astype(np.int16) - b).mean() for a, b in g]))
    return g, sharp, motion


def check(mp4, strip, ref=None):
    fr = frames(mp4); g, sharp, motion = stats(fr)
    pick = [fr[i][0] for i in np.linspace(0, len(fr) - 1, 6).astype(int)]
    h = 270; cv2.imwrite(str(strip), np.hstack([cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in pick]))
    q = {"sharp": round(sharp, 1), "motion": round(motion, 2)}
    if ref:
        rg, rsharp, rmotion = stats(frames(ref))
        q.update(ssim=round(float(np.mean([structural_similarity(a[0], b[0]) for a, b in zip(g, rg)])), 3),
                 sharp_ratio=round(sharp / rsharp, 2), motion_ratio=round(motion / max(rmotion, 1e-6), 2))
    return q
