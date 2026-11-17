"""H3-Image reference cap (engine/wsl/ref_cap.py) and how the Playground asks for it. Run: python -m unittest engine.test_ref_cap"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "wsl"))
import ref_cap  # noqa: E402
import image_models as im  # noqa: E402
import image_playground as ip  # noqa: E402
import playground  # noqa: E402

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class Capped(unittest.TestCase):
    def test_scales_short_edge_to_cap_sides_of_32(self):
        out = ref_cap.capped(Image.new("RGB", (1664, 2496)), 1024)
        self.assertEqual(out.size, (1024, 1536))
        out = ref_cap.capped(Image.new("RGB", (2496, 1664)), 1280)
        self.assertEqual(min(out.size), 1280)
        self.assertEqual(out.size[0] % 32, 0)

    def test_leaves_small_images_and_off_alone(self):
        small = Image.new("RGB", (800, 1200))
        self.assertIs(ref_cap.capped(small, 1024), small)          # never scaled up
        big = Image.new("RGB", (1664, 2496))
        self.assertIs(ref_cap.capped(big, 0), big)                 # 0 = off
        self.assertEqual(ref_cap.capped("not an image", 1024), "not an image")

    def test_wrap_reads_cap_per_call(self):
        cap = [0]
        resize = ref_cap.wrap(lambda imgs, *a, **k: ([i for i in imgs], ["mask"]), lambda: cap[0])
        imgs = [Image.new("RGB", (1664, 2496)), Image.new("RGB", (2048, 2048))]
        out, masks = resize(imgs)
        self.assertEqual([i.size for i in out], [(1664, 2496), (2048, 2048)])  # off: untouched
        cap[0] = 1024
        out, masks = resize(imgs)
        self.assertEqual([i.size for i in out], [(1024, 1536), (1024, 1024)])
        self.assertEqual(masks, ["mask"])                                      # the rest of the tuple passes through

    def test_wrap_plain_list_result(self):
        resize = ref_cap.wrap(lambda imgs: list(imgs), lambda: 1024)
        self.assertEqual(resize([Image.new("RGB", (1664, 2496))])[0].size, (1024, 1536))

    def test_choices_match_schema(self):
        self.assertEqual(set(im.get("h3_image")["params"]["ref_size"]["choices"]), set(ref_cap.CHOICES))
        self.assertEqual(ref_cap.CHOICES["full"], 0)


class WorkerJob(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(playground, "ROOT", self.tmp), mock.patch.object(ip, "ROOT", self.tmp)]
        for x in self.p:
            x.start()
        self.id = ip.new({})["id"]
        ip.save({"id": self.id, "model": "h3_image", "prompt": "Edit <Picture 1>."})
        ip.input_picture({"id": self.id, "slot": 0, "data": PNG})

    def tearDown(self):
        for x in self.p:
            x.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def job(self, **params):
        ip.save({"id": self.id, "params": {**ip.load(self.id)["params"], **params}})
        return ip.worker_job(self.id, ip.snapshot(self.id))["settings"]

    def test_default_is_full_and_sends_nothing(self):
        self.assertEqual(ip.load(self.id)["params"]["ref_size"], "full")
        self.assertNotIn("_h3_ref_size", self.job())

    def test_cap_choice_is_sent(self):
        self.assertEqual(self.job(ref_size="1024")["_h3_ref_size"], "1024")
        self.assertEqual(self.job(ref_size="1280")["_h3_ref_size"], "1280")

    def test_unknown_choice_refused(self):
        with self.assertRaises(ValueError):
            im.resolve(im.get("h3_image"), {"ref_size": "512"}, strict=True)


if __name__ == "__main__":
    unittest.main()
