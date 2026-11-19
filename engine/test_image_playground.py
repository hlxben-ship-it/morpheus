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

    def test_toggle_param(self):
        base = im.get("muse_krea2")   # no live model has a checkbox today: a made-up one on top of a real entry
        m = {**base, "params": {**base["params"], "free_pose": {"label": "x", "toggle": True, "default": True}}}
        im.check({"models": [m]})
        self.assertIs(im.defaults(m)["free_pose"], True)
        self.assertIs(im.resolve(m, {"free_pose": False}, strict=True)["free_pose"], False)
        with self.assertRaises(ValueError):
            im.resolve(m, {"free_pose": 1}, strict=True)          # a number is no tick
        self.assertIs(im.resolve(m, {"free_pose": "no"})["free_pose"], True)  # stored junk: back to the default
        bad = {"models": [{**m, "params": {"free_pose": {"toggle": True, "default": 1}}}]}
        with self.assertRaises(ValueError):
            im.check(bad)

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


class Resharpen(unittest.TestCase):
    """Glia seed muse_resharpen: the helper model's schema, prompt fill, worker job and sidecar."""
    setUp, tearDown = State.setUp, State.tearDown

    def _shelf_picture(self, meta):
        d = self.tmp / self.id / "images"
        d.mkdir(parents=True, exist_ok=True)
        png = d / "20261117_000000_1.png"
        import base64
        png.write_bytes(base64.b64decode(PNG.split(",", 1)[1]))
        if meta is not None:
            png.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return png

    def _use(self, png):
        with mock.patch.object(ip, "image_path", return_value=png):
            return ip.use_as_input({"id": self.id, "slot": 0, "image": "x"})

    def test_schema_strength_range_and_scale(self):
        m = im.get("muse_resharpen")
        self.assertEqual(m["helper"], "resharpen")
        self.assertEqual(im.defaults(m)["strength"], 0.3)
        self.assertEqual(im.defaults(m)["scale"], 1)
        self.assertEqual(im.resolve(m, {"strength": 0.6, "scale": 2}, strict=True)["scale"], 2)
        for bad in ({"strength": 0.7}, {"strength": 0.01}, {"scale": 3}):
            with self.assertRaises(ValueError):
                im.resolve(m, bad, strict=True)

    def test_slot_fills_the_prompt_from_the_pictures_sidecar(self):
        ip.save({"id": self.id, "model": "muse_resharpen"})
        st = self._use(self._shelf_picture({"prompt": "a red fox in snow"}))
        self.assertEqual(st["prompt"], "a red fox in snow")
        st = self._use(self._shelf_picture({"prompt": "DEFAULT, a cat", "studio_prompt": "a cat"}))
        self.assertEqual(st["prompt"], "a cat")  # Studio: the user's own part, not the joined default

    def test_other_models_keep_their_prompt(self):
        ip.save({"id": self.id, "model": "muse_krea2", "prompt": "mine"})
        self.assertEqual(self._use(self._shelf_picture({"prompt": "theirs"}))["prompt"], "mine")

    def test_picture_without_sidecar_keeps_the_prompt(self):
        ip.save({"id": self.id, "model": "muse_resharpen", "prompt": "kept"})
        self.assertEqual(self._use(self._shelf_picture(None))["prompt"], "kept")

    def test_snapshot_needs_the_picture_and_falls_back_to_the_neutral_prompt(self):
        ip.save({"id": self.id, "model": "muse_resharpen", "prompt": ""})
        with self.assertRaises(ValueError):
            ip.snapshot(self.id)
        st = ip.input_picture({"id": self.id, "slot": 0, "data": PNG})
        ip.save({"id": self.id, "prompt": ""})
        snap = ip.snapshot(self.id)
        self.assertEqual(snap["prompt"], "sharp, detailed photo")
        self.assertEqual((snap["helper"], snap["source"]), ("resharpen", st["inputs"][0]))

    def test_worker_job_is_img2img_without_references(self):
        st = ip.input_picture({"id": self.id, "slot": 0, "data": PNG})
        ip.save({"id": self.id, "model": "muse_resharpen", "prompt": "x", "params": {"mode": "cheap", "strength": 0.4, "scale": 2}})
        job = ip.worker_job(self.id, ip.snapshot(self.id))
        self.assertEqual(job["family"], "krea2")
        self.assertNotIn("mode", job["params"])
        self.assertTrue(job["params"]["init"].endswith(st["inputs"][0]))
        self.assertEqual(job["params"]["images"], [])
        self.assertEqual((job["params"]["strength"], job["params"]["scale"]), (0.4, 2))

    def test_studio_snapshot_with_a_picture(self):
        """Regression: in a Studio the default-prompt dict once shadowed the folder -> "dict / str" on hashing the inputs."""
        import studio
        with mock.patch.object(studio, "ROOT", self.tmp, create=True):
            sid = studio.new({})["id"]
        st = ip.input_picture({"id": sid, "slot": 0, "data": PNG})
        ip.save({"id": sid, "model": "muse_resharpen", "prompt": "a fox"})
        snap = ip.snapshot(sid)
        self.assertEqual(len(snap["input_hashes"]), 1)
        self.assertEqual(snap["source"], st["inputs"][0])
        self.assertIn("a fox", snap["prompt"])

    # --- stage 2: modes Max / Cheap / Swap only + optional persona -----------------------------------------------------------
    def _two(self, mode, persona=True, prompt="sharp, detailed photo"):
        st = ip.input_picture({"id": self.id, "slot": 0, "data": PNG})
        if persona:
            st = ip.input_picture({"id": self.id, "slot": 1, "data": PNG})
        ip.save({"id": self.id, "model": "muse_resharpen", "prompt": prompt, "params": {"mode": mode}})
        return st, ip.snapshot(self.id)

    def test_mode_default_is_max_and_unknown_mode_is_refused(self):
        m = im.get("muse_resharpen")
        self.assertEqual(im.defaults(m)["mode"], "max")
        with self.assertRaises(ValueError):
            im.resolve(m, {"mode": "turbo"}, strict=True)

    def test_max_is_an_h3_edit_of_picture_1_only(self):
        """The persona never goes to H3: a whole start frame as <Picture 2> made it copy that scene (2026-11-19 render)."""
        st, snap = self._two("max")
        job = ip.worker_job(self.id, snap)
        s = job["settings"]
        self.assertEqual((job["family"], s["video_prompt_type"], s["resolution"]), ("h3_image", "KI", "2048x2048"))
        self.assertEqual([Path(x).name for x in s["image_refs"]], st["inputs"][:1])
        self.assertNotIn("<Picture 2>", s["prompt"])
        self.assertNotIn("Scene:", s["prompt"])  # the neutral default adds nothing
        self.assertEqual(snap["total_steps"], 6)
        self.assertEqual(snap["persona"], st["inputs"][1])

    def test_max_without_persona_keeps_the_face_and_appends_the_scene(self):
        st, snap = self._two("max", persona=False, prompt="a fox in rain")
        s = ip.worker_job(self.id, snap)["settings"]
        self.assertEqual(len(s["image_refs"]), 1)
        self.assertNotIn("<Picture 2>", s["prompt"])
        self.assertTrue(s["prompt"].endswith("Scene: a fox in rain"))
        self.assertIsNone(snap["persona"])

    def test_cheap_never_sends_the_persona_to_muse(self):
        st, snap = self._two("cheap")
        job = ip.worker_job(self.id, snap)
        self.assertTrue(job["params"]["init"].endswith(st["inputs"][0]))
        self.assertEqual(job["params"]["images"], [])
        self.assertEqual(snap["total_steps"], 12)

    def test_swap_only_has_no_worker_job(self):
        _, snap = self._two("swap")
        self.assertIsNone(ip.worker_job(self.id, snap))
        self.assertEqual(snap["total_steps"], 1)

    def test_other_models_are_not_resharpen(self):
        ip.input_picture({"id": self.id, "slot": 0, "data": PNG})
        ip.save({"id": self.id, "model": "h3_image", "prompt": "x"})
        snap = ip.snapshot(self.id)
        self.assertIsNone(ip.resharpen_mode(snap))
        self.assertNotIn("persona", snap)

    def test_plain_muse_job_has_no_init(self):
        ip.input_picture({"id": self.id, "slot": 0, "data": PNG})
        ip.save({"id": self.id, "model": "muse_krea2", "prompt": "x"})
        job = ip.worker_job(self.id, ip.snapshot(self.id))
        self.assertNotIn("init", job["params"])
        self.assertEqual(len(job["params"]["images"]), 1)


if __name__ == "__main__":
    unittest.main()
