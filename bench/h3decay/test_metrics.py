"""Tests of the decay metrics (no GPU, no insightface): python -m unittest bench.h3decay.test_metrics (Morpheus root)."""
import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from bench.h3decay import metrics as m


def clip(path, frames, blur_from=None):
    """A sharp checkerboard clip; from frame blur_from on, frames are blurred."""
    board = (np.indices((360, 640)).sum(axis=0) // 8 % 2 * 255).astype(np.uint8)
    img = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 24, (640, 360))
    for i in range(frames):
        w.write(cv2.GaussianBlur(img, (15, 15), 5) if blur_from is not None and i >= blur_from else img)
    w.release()


class FakeApp:
    """Stands in for insightface: one face at a fixed box, or none."""
    def __init__(self, face=True):
        self.face = face

    def get(self, img):
        return [SimpleNamespace(bbox=np.array([200, 100, 300, 220]), normed_embedding=np.ones(4) / 2)] if self.face else []


class Eighths(unittest.TestCase):
    def test_eight_parts_inside_the_clip(self):
        parts = m.eighths(209)
        self.assertEqual(len(parts), 8)
        self.assertTrue(all(0 <= i and i + 1 < 209 for p in parts for i in p))
        self.assertTrue(all(max(a) < min(b) for a, b in zip(parts, parts[1:])))

    def test_too_short_clip_is_refused(self):
        with self.assertRaises(ValueError):
            m.eighths(5)


class FaceCrop(unittest.TestCase):
    def test_crop_is_square_and_clipped(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(m.face_crop(img, (80, 80, 120, 130)).shape, (m.FACE_PX, m.FACE_PX, 3))

    def test_size_bias_stays_bounded(self):
        """A push-in: the same detailed face seen far (100 px) and near (500 px) should read about equally sharp.
        Pure pixel noise is the worst case for resampling; 64 px keeps the ratio ~0.65, the first 256 px setting gave 0.10."""
        rng = np.random.default_rng(1)
        near = cv2.GaussianBlur(rng.integers(0, 255, (800, 800), dtype=np.uint8), (3, 3), 0.8)
        far = cv2.resize(near, (160, 160), interpolation=cv2.INTER_AREA)
        far, near = (cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in (far, near))
        small, large = m.face_crop(far, (30, 30, 130, 130)), m.face_crop(near, (150, 150, 650, 650))
        self.assertGreater(m.sharpness(small) / m.sharpness(large), 0.5)

    def test_tiny_box_gives_none(self):
        self.assertIsNone(m.face_crop(np.zeros((100, 100, 3), np.uint8), (99, 99, 100, 100)))


class Measure(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def test_blur_at_the_end_shows_as_decay(self):
        p = os.path.join(self.dir.name, "decay.avi")
        clip(p, 80, blur_from=60)
        rows = m.measure(p, FakeApp())
        self.assertEqual(len(rows), 8)
        self.assertLess(m.decay(rows, "frame_sharp"), 0.5)
        self.assertLess(m.decay(rows, "face_sharp"), 0.5)
        self.assertEqual(m.decay(rows, "face_match"), 1.0)

    def test_steady_clip_shows_no_decay(self):
        p = os.path.join(self.dir.name, "steady.avi")
        clip(p, 80)
        rows = m.measure(p, FakeApp())
        self.assertAlmostEqual(m.decay(rows, "frame_sharp"), 1.0, delta=0.15)  # MJPG noise alone moves it ~8 %

    def test_no_face_leaves_face_numbers_empty(self):
        p = os.path.join(self.dir.name, "noface.avi")
        clip(p, 80)
        rows = m.measure(p, FakeApp(face=False))
        self.assertIsNone(rows[0]["face_sharp"])
        self.assertIsNone(m.decay(rows, "face_match"))
        self.assertEqual(rows[0]["faces_found"], "0/4")

    def test_face_entering_late_is_flagged(self):
        p = os.path.join(self.dir.name, "late.avi")
        clip(p, 80)
        app = FakeApp()
        calls = {"n": 0}
        real = app.get
        def late(img):  # no face in the first eighth's frames (the first 4 sampled), then one
            calls["n"] += 1
            return [] if calls["n"] <= 4 else real(img)
        app.get = late
        rows = m.measure(p, app)
        self.assertFalse(m.face_valid(rows))
        self.assertIsNone(m.decay(rows, "face_match") if m.face_valid(rows) else None)

    def test_face_from_the_start_is_valid(self):
        p = os.path.join(self.dir.name, "start.avi")
        clip(p, 80)
        self.assertTrue(m.face_valid(m.measure(p, FakeApp())))


class Pick(unittest.TestCase):
    def face(self, emb, size=100):
        return SimpleNamespace(bbox=np.array([0, 0, size, size]), normed_embedding=np.array(emb, float))

    def test_tracks_identity_not_size(self):
        her, other = self.face([1, 0], 50), self.face([0, 1], 300)
        self.assertIs(m.pick([other, her], np.array([1.0, 0])), her)

    def test_biggest_before_a_reference(self):
        small, big = self.face([1, 0], 50), self.face([0, 1], 300)
        self.assertIs(m.pick([small, big], None), big)

    def test_no_faces(self):
        self.assertIsNone(m.pick([], None))


class Decay(unittest.TestCase):
    def test_ratio_and_missing(self):
        self.assertEqual(m.decay([{"x": 200}, {"x": 50}], "x"), 0.25)
        self.assertIsNone(m.decay([{"x": None}, {"x": 50}], "x"))


if __name__ == "__main__":
    unittest.main()
