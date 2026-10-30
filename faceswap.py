"""Face swap for videos: puts the face from one photo onto the largest face in every frame of a clip.

Runs in .venv-faceswap (insightface + onnxruntime-directml), not in the ComfyUI venv; server.py starts it as a
subprocess and reads the progress lines from stdout.

    python faceswap.py <photo> <video> <out.mp4> [--quality fast|high|premium] [--no-enhance] [--hair] [--upscale 2|4]
                       [--start s] [--end s]

fast: inswapper 128 px, GFPGAN over a soft rectangle.
high: HyperSwap 256 px, pasted through an occlusion mask (XSeg: hands, hair in front) and a face-region mask
      (BiSeNet: skin, brows, eyes, nose, mouth), GPEN 512 blended at 70 % so skin texture survives,
      and landmarks smoothed over neighbouring frames against jitter.
premium: as high, but restored with GPEN 1024: twice the detail in skin, eyes and the hairline.
--upscale: Real-ESRGAN x2 / x4 over every finished frame, in tiles.
"""
import os
import subprocess
import sys
import tempfile
import time

import cv2
import imageio_ffmpeg
import insightface
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "faceswap")
PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]

# 5-landmark alignment templates (eyes, nose, mouth corners), as fractions of the crop size
FFHQ = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                 [0.39308822, 0.72541100], [0.61150205, 0.72490465]], dtype=np.float32)
ARCFACE = np.array([[0.36167656, 0.40387734], [0.63696719, 0.40235469], [0.50019687, 0.56044219],
                    [0.38710391, 0.72160547], [0.61507734, 0.72034453]], dtype=np.float32)
FFHQ_512 = FFHQ * 512

# BiSeNet face-parsing classes that belong to the swapped face: skin, brows, eyes, nose, mouth, lips
FACE_REGIONS = [1, 2, 3, 4, 5, 10, 11, 12, 13]
ENHANCE_BLEND = 0.7
# share of the last frame's mask in this frame's, against flicker: face masks (face-aligned) and the hair mask (in the frame)
MASK_KEEP, HAIR_KEEP = 0.5, 0.6

# feathered mask so the restored crop blends into the frame
MASK = np.zeros((512, 512), np.float32)
cv2.rectangle(MASK, (32, 32), (480, 480), 1.0, -1)
MASK = cv2.GaussianBlur(MASK, (0, 0), 16)


def area(face):
    return (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])


def session(name):
    s = ort.InferenceSession(os.path.join(MODELS, name), providers=PROVIDERS)
    return s, s.get_inputs()[0].name


def align(img, kps, template, size):
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), template * size, method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE), M


def paste(img, crop, M, mask):
    IM = cv2.invertAffineTransform(M)
    h, w = img.shape[:2]
    back = cv2.warpAffine(crop, IM, (w, h), borderMode=cv2.BORDER_REPLICATE)
    m = cv2.warpAffine(mask, IM, (w, h))[:, :, None]
    return (back * m + img * (1 - m)).clip(0, 255).astype(np.uint8)


def to_net(crop):
    """BGR uint8 crop -> RGB NCHW in [-1, 1]."""
    return ((crop[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]


def from_net(y):
    """RGB CHW in [-1, 1] -> BGR float crop in [0, 255]."""
    return ((np.clip(y, -1, 1) + 1) / 2 * 255).transpose(1, 2, 0)[:, :, ::-1]


def box_mask(size):
    m = np.zeros((size, size), np.float32)
    pad = int(size * 0.06)
    cv2.rectangle(m, (pad, pad), (size - pad, size - pad), 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), size * 0.04)


def soften(mask, size):
    """Resize a 0..1 mask to the crop and feather its edge inwards, so it never bleeds past the boundary."""
    mask = cv2.resize(mask.astype(np.float32), (size, size))
    return (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2


class Enhancer:
    """GFPGAN 1.4 on the aligned face crop, pasted back over the swapped frame."""

    def __init__(self):
        self.session, self.input = session("gfpgan_1.4.onnx")

    def __call__(self, img, kps):
        crop, M = align(img, kps, FFHQ, 512)
        y = from_net(self.session.run(None, {self.input: to_net(crop)})[0][0])
        return paste(img, y, M, MASK)


class Masks:
    """Which pixels of an aligned crop belong to the visible face."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self):
        self.xseg, self.xseg_in = session("xseg_2.onnx")
        self.parser, self.parser_in = session("bisenet_resnet_34.onnx")
        self.boxes = {}

    def occlusion(self, crop):
        x = cv2.resize(crop, (256, 256)).astype(np.float32)[None] / 255.0
        return soften(self.xseg.run(None, {self.xseg_in: x})[0][0, :, :, 0], crop.shape[0])

    def region(self, crop):
        x = (cv2.resize(crop, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0 - self.MEAN) / self.STD
        classes = self.parser.run(None, {self.parser_in: x.transpose(2, 0, 1)[None]})[0][0].argmax(0)
        return soften(np.isin(classes, FACE_REGIONS), crop.shape[0])

    def box(self, size):
        if size not in self.boxes:
            self.boxes[size] = box_mask(size)
        return self.boxes[size]


class HighQuality:
    def __init__(self, source, enhance, restore=512):
        self.swapper, _ = session("hyperswap_1a_256.onnx")
        self.embedding = source.normed_embedding.reshape(1, -1).astype(np.float32)
        self.masks = Masks()
        self.enhancer = session(f"gpen_bfr_{restore}.onnx") if enhance else None
        self.restore = restore
        self.prev = {}

    def steady(self, key, mask):
        """Mask averaged with the last frame's. The crops are face-aligned, so this follows the head instead of
        trailing behind it, and the segmentation's per-frame noise at the edges stops flickering."""
        prev = self.prev.get(key)
        if prev is not None:
            mask = prev * MASK_KEEP + mask * (1 - MASK_KEEP)
        self.prev[key] = mask
        return mask

    def __call__(self, frame, kps):
        crop, M = align(frame, kps, ARCFACE, 256)
        out = self.swapper.run(None, {"source": self.embedding, "target": to_net(crop)})[0][0]
        mask = self.steady("swap", np.minimum.reduce([self.masks.box(256), self.masks.occlusion(crop), self.masks.region(crop)]))
        frame = paste(frame, from_net(out), M, mask)

        if self.enhancer:
            enhancer, name = self.enhancer
            crop, M = align(frame, kps, FFHQ, self.restore)
            restored = from_net(enhancer.run(None, {name: to_net(crop)})[0][0])
            restored = restored * ENHANCE_BLEND + crop.astype(np.float32) * (1 - ENHANCE_BLEND)
            mask = self.steady("restore", np.minimum(self.masks.box(self.restore), self.masks.occlusion(crop)))
            frame = paste(frame, restored, M, mask)
        return frame


class HairColor:
    """Recolours the hair in the video to the hair colour of the photo; cut and length stay as they are.

    BiSeNet finds the hair next to the face in a zoomed-out crop (the seed) and k-means learns its colours.
    Long hair runs far outside that crop, so the mask then grows over pixels of those colours, inside the
    person (u2net human segmentation, so the wall never takes part) and away from the face. When the face is
    covered and there is no seed, the colours of the last good frame carry on, so the hair doesn't flicker back.
    The colour moves in Lab: lightness and tint shift to the photo's, the strands' own shading stays.
    """

    HAIR, SKIN = 17, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # BiSeNet: hair; face, ears, neck
    WIDE = (FFHQ - 0.5) * 0.55 + 0.5  # the face takes about half the crop, so the hair around it is in view
    SCALE = 0.5  # the mask is found at half resolution; only the recolouring runs at full size
    REFIT = 12

    def __init__(self, masks, photo, face):
        self.masks = masks
        self.person, self.person_in = session("u2net_human_seg.onnx")
        hair, _ = self.parse(photo, face.kps)
        lab = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)[hair]
        if len(lab) < 300:
            sys.exit("error: no hair found in the photo, so the hair colour can't be taken over.")
        self.src_mean, self.src_std = lab.mean(0), lab.std(0) + 1e-3
        self.mean = self.std = self.centers = None
        self.alpha, self.n, self.blur = None, 0, 1.5  # hair alpha of the last frame (full size), frame count, edge blur

    def body(self, img):
        x = cv2.resize(img, (320, 320))[:, :, ::-1].astype(np.float32)
        x = (x / max(x.max(), 1e-6) - Masks.MEAN) / Masks.STD
        y = self.person.run(None, {self.person_in: x.transpose(2, 0, 1)[None]})[0][0, 0]
        y = (y - y.min()) / max(y.max() - y.min(), 1e-6)
        return cv2.resize(y, (img.shape[1], img.shape[0])) > 0.5

    def parse(self, img, kps):
        """Hair and skin masks in frame coordinates, from BiSeNet on a zoomed-out aligned crop."""
        crop, M = align(img, kps, self.WIDE, 512)
        x = (crop[:, :, ::-1].astype(np.float32) / 255.0 - Masks.MEAN) / Masks.STD
        classes = self.masks.parser.run(None, {self.masks.parser_in: x.transpose(2, 0, 1)[None]})[0][0].argmax(0)
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        back = lambda m: cv2.warpAffine(m.astype(np.uint8), IM, (w, h), flags=cv2.INTER_NEAREST) > 0
        return back(classes == self.HAIR), back(np.isin(classes, self.SKIN))

    def segment(self, frame, face):
        """Hair mask at half resolution inside a box around the head: (box, hair, skin, lab of the box) or None."""
        seed, skin = self.parse(frame, face.kps)
        # work in a box around the head that reaches down to the waist, where long hair ends
        fw, fh = face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]
        cx, cy = (face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2
        x1, x2 = int(max(0, cx - 3 * fw)), int(min(frame.shape[1], cx + 3 * fw))
        y1, y2 = int(max(0, cy - 2.5 * fh)), int(min(frame.shape[0], cy + 6 * fh))
        roi = frame[y1:y2, x1:x2]
        size = (max(1, int(roi.shape[1] * self.SCALE)), max(1, int(roi.shape[0] * self.SCALE)))
        small = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
        shrink = lambda m: cv2.resize(m[y1:y2, x1:x2].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        person = self.body(small)
        seed, skin = shrink(seed) & person, shrink(skin)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)

        # learn the hair colours from the seed, every REFIT frames: k-means each frame gives slightly different centres
        # and the mask flickers with them; without a seed (face covered) keep the last ones
        if seed.sum() >= 150 and (self.centers is None or self.n % self.REFIT == 0):
            samples = lab[seed]
            samples = samples[np.random.default_rng(0).choice(len(samples), min(len(samples), 3000), replace=False)]
            _, _, self.centers = cv2.kmeans(samples, 3, None,
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                            2, cv2.KMEANS_PP_CENTERS)
            # a highlight or a prop in the seed can give a skin-like centre, which would then grow over the arms
            if skin.sum() >= 150:
                tone = lab[skin][:, None, :]  # a centre that already explains a good part of the skin is not hair
                near = (np.linalg.norm(tone - self.centers[None], axis=2) < 22).mean(0) > 0.15
                if not near.all():
                    self.centers = self.centers[~near]
        if self.centers is None:
            return None

        # grow: pixels close to one of the hair colours, inside the person, not skin; connected to the seed if any.
        # Dyed tips sit a bit further from the root colours, and strands in front of the body are cut off from the
        # head by thin gaps, hence the looser distance and the wider closing before the connection test.
        dist = np.min(np.linalg.norm(lab[:, :, None, :] - self.centers[None, None], axis=3), axis=2)
        skin = cv2.dilate(skin.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        cand = ((dist < 18) | seed) & person & ~skin
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if seed.sum() >= 150:
            _, labels = cv2.connectedComponents(cand)
            keep = np.unique(labels[seed & (cand > 0)])
            hair = np.isin(labels, keep[keep > 0])
        else:
            hair = cand > 0
        if hair.sum() < 100:
            return None
        return (x1, y1, x2, y2), hair, skin, lab

    def __call__(self, frame, face):
        self.n += 1
        found = self.segment(frame, face)
        alpha = np.zeros(frame.shape[:2], np.float32)
        if found is not None:
            (x1, y1, x2, y2), hair, _, lab = found
            self.blur = max(1.5, (face.bbox[2] - face.bbox[0]) * 0.01)
            a = cv2.resize(hair.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            alpha[y1:y2, x1:x2] = cv2.GaussianBlur(a, (0, 0), self.blur)
            # hair colour of this frame, smoothed over time so the shift doesn't flicker
            mean, std = lab[hair].mean(0), lab[hair].std(0) + 1e-3
            if self.mean is not None:
                mean, std = self.mean * 0.8 + mean * 0.2, self.std * 0.8 + std * 0.2
            self.mean, self.std = mean, std
        # the mask carries part of the last frame's: edges stop jumping, and a frame where nothing is found fades the
        # colour out over a few frames instead of popping back to the original
        if self.alpha is not None and self.alpha.shape == alpha.shape:
            alpha = alpha * (1 - HAIR_KEEP) + self.alpha * HAIR_KEEP
        self.alpha = alpha
        if self.mean is None:
            return frame
        ys, xs = np.nonzero(alpha > 0.02)
        if not len(ys):
            return frame
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi, a = frame[y1:y2, x1:x2], alpha[y1:y2, x1:x2, None]

        # tint (a, b) may shrink a lot: dyed streaks of another colour should end up in the photo's colour too
        ratio = np.clip(self.src_std / self.std, [0.5, 0.1, 0.1], [2.5, 1.5, 1.5])
        full = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        recoloured = (full - self.mean) * ratio + self.src_mean
        recoloured = cv2.cvtColor(recoloured.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = frame.copy()
        out[y1:y2, x1:x2] = (recoloured * a + roi * (1 - a)).astype(np.uint8)
        return out


class Upscaler:
    """Real-ESRGAN over the whole frame, in overlapping tiles so a 1080p frame fits the card."""

    TILE, PAD = 384, 16

    def __init__(self, factor):
        self.session, self.input = session(f"real_esrgan_x{factor}.onnx")
        self.factor = factor

    def __call__(self, frame):
        h, w = frame.shape[:2]
        f, t, p = self.factor, self.TILE, self.PAD
        out = np.zeros((h * f, w * f, 3), np.uint8)
        for y in range(0, h, t):
            for x in range(0, w, t):
                # each tile with a margin so the seams have context; only its inner part is kept
                y1, x1, y2, x2 = max(0, y - p), max(0, x - p), min(h, y + t + p), min(w, x + t + p)
                tile = frame[y1:y2, x1:x2, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                up = self.session.run(None, {self.input: tile})[0][0].clip(0, 1).transpose(1, 2, 0)[:, :, ::-1] * 255
                iy, ix = (y - y1) * f, (x - x1) * f
                th, tw = (min(h, y + t) - y) * f, (min(w, x + t) - x) * f
                out[y * f:y * f + th, x * f:x * f + tw] = up[iy:iy + th, ix:ix + tw].round().astype(np.uint8)
        return out


class Smoother:
    """Exponential smoothing of the 5 landmarks; resets on a jump (new shot, fast turn) so it never lags far behind."""

    def __init__(self, keep=0.5):
        self.keep, self.last = keep, None

    def __call__(self, face):
        kps = face.kps.astype(np.float32)
        size = max(face.bbox[2] - face.bbox[0], 1)
        if self.last is not None and np.abs(kps - self.last).mean() < size * 0.08:
            kps = self.last * self.keep + kps * (1 - self.keep)
        self.last = kps
        return kps


def main(photo, video, out, quality="fast", enhance=True, hair=False, start=0.0, end=None, upscale=1):
    app = FaceAnalysis(name="buffalo_l", root=MODELS, providers=PROVIDERS, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    src = cv2.imread(photo)
    if src is None:
        sys.exit("error: the photo can't be read.")
    faces = app.get(src)
    if not faces:
        sys.exit("error: no face found in the photo.")
    source = max(faces, key=area)

    if quality in ("high", "premium"):
        high, smooth = HighQuality(source, enhance, 1024 if quality == "premium" else 512), Smoother()
    else:
        swapper = insightface.model_zoo.get_model(os.path.join(MODELS, "inswapper_128.onnx"), providers=PROVIDERS)
        enhancer = Enhancer() if enhance else None
    hair_color = HairColor(high.masks if quality != "fast" else Masks(), src, source) if hair else None

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("error: the video can't be read.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = min(count, int(round(start * fps)))
    stop = count if end is None else min(count, int(round(end * fps)))
    total = max(0, stop - first)
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # at most 4K on the long side: x4 is for small clips, a 1080p clip gets x2, a 4K clip none
    while upscale > 1 and max(w, h) * upscale > 3840:
        upscale //= 2
    upscaler = Upscaler(upscale) if upscale > 1 else None
    silent = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * upscale, h * upscale))

    i, missed, t0, last = 0, 0, time.time(), None
    while i < total:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        if not faces and hair_color and last is not None:
            frame = hair_color(frame, last)  # face hidden for a moment: the hair keeps its new colour anyway
        if faces:
            last = max(faces, key=area)
            target = max(faces, key=area)
            if quality != "fast":
                frame = high(frame, smooth(target))
            else:
                frame = swapper.get(frame, target, source, paste_back=True)
                if enhancer:
                    frame = enhancer(frame, target.kps)
            if hair_color:
                frame = hair_color(frame, target)
        else:
            missed += 1
        writer.write(upscaler(frame) if upscaler else frame)
        i += 1
        print(f"progress {i} {total}", flush=True)
    cap.release()
    writer.release()

    # re-encode to H.264 for the browser and carry the original audio over
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent,
                    "-ss", f"{first / fps:.3f}", "-t", f"{i / fps:.3f}", "-i", video,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", out], check=True)
    os.remove(silent)
    print(f"done {i} {missed} {time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    opt = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
    values = {opt(n, None) for n in ("--quality", "--start", "--end", "--upscale")}
    args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in values]
    end = opt("--end", None)
    main(*args[:3], quality=opt("--quality", "fast"), enhance="--no-enhance" not in sys.argv, hair="--hair" in sys.argv,
         start=float(opt("--start", 0)), end=float(end) if end is not None else None,
         upscale=int(opt("--upscale", 1)))
