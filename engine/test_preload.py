"""Per-job VRAM preload of the warm worker (engine/wsl/preload.py). Run: python -m unittest engine.test_preload"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "wsl"))
from preload import preload_for  # noqa: E402


class PreloadFor(unittest.TestCase):
    def test_h3_image_gets_its_preload(self):
        self.assertEqual(preload_for({"kind": "image", "family": "h3_image"}, 0, 14000), 14000)

    def test_everything_else_keeps_the_base(self):
        for job in ({"kind": "render", "settings": {"model_type": "minimax_h3_ref2va_hyperflow"}},
                    {"kind": "render", "family": "h3_image"},   # only an image job is H3-Image
                    {"kind": "image", "family": "sdxl"}, {"kind": "image", "family": "qwen_edit"},
                    {"kind": "image"}, {"kind": "upscale"}):
            self.assertEqual(preload_for(job, 0, 14000), 0, job)

    def test_base_is_passed_through(self):
        self.assertEqual(preload_for({"kind": "render"}, 5000, 14000), 5000)


if __name__ == "__main__":
    unittest.main()
