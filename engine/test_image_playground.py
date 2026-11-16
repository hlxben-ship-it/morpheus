"""Image Playground: schema rules (image_models) and saved state (image_playground). Run: python -m unittest engine.test_image_playground"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import image_models as im
import image_playground as ip
import playground

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class Schema(unittest.TestCase):
    def test_file_passes_check(self):
        self.assertTrue(im.models())

    def test_resolve_strict_refuses(self):
        m = im.get("janku_v5")
        with self.assertRaises(ValueError):
            im.resolve(m, {"steps": 500}, strict=True)
        with self.assertRaises(ValueError):
            im.resolve(m, {"sampler": "magic"}, strict=True)
        with self.assertRaises(ValueError):
            im.resolve(m, {"nope": 1}, strict=True)

    def test_resolve_lenient_repairs(self):
        m = im.get("janku_v5")
        out = im.resolve(m, {"steps": 500, "sampler": "magic", "nope": 1})
        self.assertEqual(out["steps"], 80)
        self.assertEqual(out["sampler"], m["params"]["sampler"]["default"])
        self.assertNotIn("nope", out)

    def test_fixed_param_stays(self):
        self.assertEqual(im.resolve(im.get("h3_image"), {"steps": 20})["steps"], 6)

    def test_size(self):
        m = im.get("janku_v5")
        self.assertEqual(im.resolve_size(m, "1x1"), m["sizes"][0])
        with self.assertRaises(ValueError):
            im.resolve_size(m, "1x1", strict=True)

    def test_broken_schema_says_what(self):
        with self.assertRaises(ValueError):
            im.check({"models": []})


class State(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(playground, "ROOT", self.tmp), mock.patch.object(ip, "ROOT", self.tmp)]
        for x in self.p:
            x.start()
        self.id = ip.new({})["id"]

    def tearDown(self):
        for x in self.p:
            x.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_is_image_with_defaults(self):
        st = ip.load(self.id)
        self.assertEqual(st["kind"], "image")
        self.assertEqual(len(st["inputs"]), ip.INPUTS)
        self.assertEqual(len(st["loras"]), ip.LORAS)

    def test_video_load_refuses_image_playground(self):
        with self.assertRaises(ValueError):
            playground.load(self.id)

    def test_save_checks_count_and_seed(self):
        with self.assertRaises(ValueError):
            ip.save({"id": self.id, "count": 99})
        with self.assertRaises(ValueError):
            ip.save({"id": self.id, "seed": "abc"})
        st = ip.save({"id": self.id, "count": 2, "seed": "12"})
        self.assertEqual((st["count"], st["seed"]), (2, 12))

    def test_model_switch_takes_new_defaults(self):
        st = ip.save({"id": self.id, "model": "h3_image"})
        self.assertEqual(st["params"], im.defaults(im.get("h3_image")))
        self.assertEqual(st["prompt"], im.get("h3_image").get("prompt_default", ""))

    def test_model_switch_keeps_own_prompt(self):
        ip.save({"id": self.id, "prompt": "a fox"})
        self.assertEqual(ip.save({"id": self.id, "model": "h3_image"})["prompt"], "a fox")

    def test_input_slot(self):
        st = ip.input_picture({"id": self.id, "slot": 1, "data": PNG})
        self.assertTrue(st["inputs"][1])
        self.assertTrue((self.tmp / self.id / st["inputs"][1]).is_file())
        with self.assertRaises(ValueError):
            ip.input_picture({"id": self.id, "slot": 9, "data": PNG})
        self.assertIsNone(ip.input_picture({"id": self.id, "slot": 1, "data": None})["inputs"][1])

    def test_unknown_lora_refused(self):
        with self.assertRaises(ValueError):
            ip.save({"id": self.id, "loras": [{"id": "no_such_lora", "strength": 1}] + [{"id": "", "strength": 1}] * (ip.LORAS - 1)})

    def test_bad_id(self):
        with self.assertRaises(ValueError):
            ip.load("../etc")


if __name__ == "__main__":
    unittest.main()
