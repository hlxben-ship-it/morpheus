"""H3 quality decay (Glia seed h3_quality_decay): how a clip loses detail and the face from its first eighth to its last.

Per eighth of the clip: face sharpness (Laplacian variance of the face crop, scaled to 256 px), face match (insightface
cosine to the face in the clip's first frame), whole-frame sharpness (640 px wide, as in bench/h3tune/quality.py) and
motion (mean frame-to-frame change). Main number: last eighth vs first.

Run in .venv-faceswap (insightface):  .venv-faceswap\\Scripts\\python.exe -m bench.h3decay.metrics out.csv clip.mp4 [...]
"""
import csv
import os
import sys

import cv2
import numpy as np

PARTS = 8
PER_PART = 4  # frames sampled per eighth
FACE_PX = 64  # small on purpose: a far face is not blown up (reads blurry); residual size bias ~0.65 worst case (pure noise), see test


def eighths(total: int, parts: int = PARTS, per_part: int = PER_PART) -> list[list[int]]:
    """Frame indices to sample, grouped per part; every part gets evenly spread frames inside its own span."""
    if total < parts:
        raise ValueError(f"clip has {total} frames, fewer than {parts} parts")
    edges = np.linspace(0, total, parts + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = int(a), max(int(a), int(b) - 2)  # keep i+1 inside the part for the motion pair
        out.append(sorted(set(np.linspace(lo, hi, per_part).astype(int).tolist())))
    return out


def sharpness(img) -> float:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def frame_sharpness(img) -> float:
    h, w = img.shape[:2]
    return sharpness(cv2.resize(img, (640, max(1, round(h * 640 / w)))))


def face_crop(img, box):
    """The face box padded by 20 %, clipped to the frame, scaled to FACE_PX square; None when nothing is left."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (FACE_PX, FACE_PX), interpolation=cv2.INTER_AREA)  # AREA: bilinear aliases on big shrinks


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def biggest(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) if faces else None


def pick(faces, ref):
    """The tracked face: the one closest to the reference identity; the biggest while there is no reference yet."""
    if not faces:
        return None
    return biggest(faces) if ref is None else max(faces, key=lambda f: cosine(ref, f.normed_embedding))


def mean_or_none(xs):
    return round(float(np.mean(xs)), 3) if xs else None


def measure(mp4: str, app=None) -> list[dict]:
    """One row per eighth. app = an insightface FaceAnalysis (detection + recognition); None skips the face numbers."""
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    read = lambda i: (cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)), cap.read())[1]
    ref = None  # identity of the first sampled face; every later frame is matched against it
    rows = []
    for n, idx in enumerate(eighths(total)):
        fs, ms, face_s, face_m = [], [], [], []
        for i in idx:
            ok, a = read(i)
            ok2, b = read(i + 1)
            if not (ok and ok2):
                continue
            fs.append(frame_sharpness(a))
            ga, gb = (cv2.cvtColor(cv2.resize(x, (640, 360)), cv2.COLOR_BGR2GRAY) for x in (a, b))
            ms.append(float(np.abs(ga.astype(np.int16) - gb).mean()))
            if app is not None:
                f = pick(app.get(a), ref)
                if f is not None:
                    if ref is None:
                        ref = f.normed_embedding
                    crop = face_crop(a, f.bbox)
                    if crop is not None:
                        face_s.append(sharpness(crop))
                    if ref is not None:
                        face_m.append(cosine(ref, f.normed_embedding))
        rows.append({"clip": os.path.basename(str(mp4)), "part": n + 1, "frames": total,
                     "frame_sharp": mean_or_none(fs), "motion": mean_or_none(ms),
                     "face_sharp": mean_or_none(face_s), "face_match": mean_or_none(face_m),
                     "faces_found": f"{len(face_s)}/{len(idx)}"})
    cap.release()
    return rows


def decay(rows: list[dict], key: str):
    """Last part vs first part as a ratio (0.6 = lost 40 %); None when either end has no value."""
    a, b = rows[0].get(key), rows[-1].get(key)
    return round(b / a, 3) if a and b is not None else None


def face_valid(rows: list[dict]) -> bool:
    """Face numbers mean something only when the face is there from the start (>= 3 of 4 frames of the first eighth);
    a person walking in later would make the start match a stranger at the door."""
    found, of = (int(x) for x in rows[0]["faces_found"].split("/"))
    return found >= 3 and of > 0


def face_app():
    from insightface.app import FaceAnalysis
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "faceswap")
    app = FaceAnalysis(name="buffalo_l", root=root, providers=["CPUExecutionProvider"],  # CPU: never competes with a render on the GPU
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def main(out_csv: str, clips: list[str]):
    app = face_app()
    rows = []
    for c in clips:
        r = measure(c, app)
        rows += r
        ok = face_valid(r)
        print(os.path.basename(c), "face_match", decay(r, "face_match") if ok else "n/a", "face_sharp",
              decay(r, "face_sharp") if ok else "n/a", "frame_sharp", decay(r, "frame_sharp"), flush=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
