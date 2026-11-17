"""Reference latent cache (engine/wsl/ref_latents.py). Run: python -m unittest engine.test_ref_latents"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "wsl"))
from ref_latents import LatentCache, wrap  # noqa: E402


class Pipe:
    def __init__(self, h3_image=True):
        self.h3_image, self.calls = h3_image, 0


def encode(self, video, keep_all_latents=False):
    self.calls += 1
    return f"latent({video},{keep_all_latents})"


def make(cache):
    return wrap(encode, lambda self, v, k: (id(self), v, k), lambda self: self.h3_image, cache)


class Cache(unittest.TestCase):
    def test_same_input_encodes_once(self):
        f, p = make(LatentCache()), Pipe()
        self.assertEqual(f(p, "a"), "latent(a,False)")
        self.assertEqual(f(p, "a"), "latent(a,False)")
        self.assertEqual(p.calls, 1)

    def test_other_input_or_flag_encodes_again(self):
        f, p = make(LatentCache()), Pipe()
        f(p, "a"); f(p, "b"); f(p, "a", keep_all_latents=True)
        self.assertEqual(p.calls, 3)

    def test_other_pipeline_does_not_share(self):
        f, p1, p2 = make(LatentCache()), Pipe(), Pipe()
        f(p1, "a"); f(p2, "a")
        self.assertEqual((p1.calls, p2.calls), (1, 1))

    def test_not_h3_image_never_cached(self):
        c = LatentCache(); f, p = make(c), Pipe(h3_image=False)
        f(p, "a"); f(p, "a")
        self.assertEqual(p.calls, 2)
        self.assertEqual(len(c.entries), 0)

    def test_oldest_dropped_beyond_limit(self):
        c = LatentCache(max_entries=2); f, p = make(c), Pipe()
        f(p, "a"); f(p, "b"); f(p, "c")
        self.assertEqual(len(c.entries), 2)
        f(p, "a")                      # dropped: encoded again
        self.assertEqual(p.calls, 4)
        f(p, "c")                      # still there
        self.assertEqual(p.calls, 4)

    def test_copy_keeps_cached_value_safe(self):
        c = LatentCache(); p = Pipe()
        f = wrap(lambda self, v, keep_all_latents=False: [v], lambda s, v, k: v, lambda s: True, c, copy=list)
        first = f(p, "a"); first.append("changed by caller")
        self.assertEqual(f(p, "a"), ["a"])


if __name__ == "__main__":
    unittest.main()
