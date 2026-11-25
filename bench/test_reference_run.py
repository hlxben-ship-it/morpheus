"""Tests of the reference shot variants and runner logic (no GPU, no server): python -m unittest bench.test_reference_run (Morpheus root)."""
import json
import tempfile
import unittest
from pathlib import Path

import image_models
import morpheus
import playground_models
from bench import reference_run as rr

REF = rr.load_ref()


class Variants(unittest.TestCase):
    def test_every_t2i_model_has_a_variant(self):
        t2i = set(morpheus.MODELS) | {m["id"] for m in image_models.models() if m["inputs"]["min"] == 0}
        self.assertEqual(t2i - set(REF["image"]), set())

    def test_edit_only_models_are_left_out(self):
        self.assertNotIn("qwen_edit_2511", REF["image"])
        self.assertFalse(set(morpheus.EDIT_MODELS) & set(REF["image"]))

    def test_every_video_model_has_a_variant(self):
        self.assertEqual({m["id"] for m in playground_models.models()}, set(REF["video"]["models"]))

    def test_anime_only_models_are_restyled_realism_stays_photo(self):
        for mid, v in REF["image"].items():
            if v["pipe"] == "comfy":
                want = "photo" if morpheus.MODELS[mid]["category"] == "realism" else ("anime", "cartoon")
                self.assertIn(v["style"], want if isinstance(want, tuple) else (want,), mid)
        jobs = {j["id"]: j for j in rr.image_jobs(REF)}
        for mid, j in jobs.items():
            if j["style"] == "anime":
                self.assertNotIn("Photorealistic", j["prompt"], mid)
            if mid in ("z-image-turbo", "flux2-klein-4b", "h3_image", "muse_krea2"):
                self.assertEqual(j["prompt"], REF["master"]["prompt"], mid)

    def test_all_prompts_resolve_and_sizes_are_portrait_and_offered(self):
        for j in rr.image_jobs(REF):
            self.assertNotIn("@", j["prompt"], j["id"])
            self.assertLess(j["width"], j["height"], j["id"])
            if j["pipe"] == "image":
                self.assertIn(f"{j['width']}x{j['height']}", image_models.get(j["id"])["sizes"], j["id"])
        for j in rr.video_jobs(REF):
            self.assertNotIn("@", j["prompt"])
            self.assertEqual(j["duration"], 5)
            self.assertTrue(Path(j["start_frame"]).is_file())

    def test_h3_prompt_is_structured_so_the_engine_keeps_it(self):
        from engine import jobs
        p = rr.video_jobs(REF)[1]["prompt"]
        self.assertEqual(jobs.h3_prompt(p, 1, None), p.strip())
        self.assertIn("At 00:03.000", p)


class Resolve(unittest.TestCase):
    TABLE = {"a": {"prompt": "tags"}, "b": {"prompt": "pre, @a"}, "x": {"prompt": "@y"}, "y": {"prompt": "@x"}}

    def test_nested_reference(self):
        self.assertEqual(rr.resolve_prompt(REF, "@b", self.TABLE), "pre, tags")
        self.assertEqual(rr.resolve_prompt(REF, "@master", {}), REF["master"]["prompt"])

    def test_unknown_and_loop_are_errors(self):
        with self.assertRaises(ValueError):
            rr.resolve_prompt(REF, "@nope", self.TABLE)
        with self.assertRaises(ValueError):
            rr.resolve_prompt(REF, "@x", self.TABLE)

    def test_negative(self):
        self.assertEqual(rr.resolve_negative(REF, "+nipples", "lowres, "), "lowres, nipples")
        self.assertEqual(rr.resolve_negative(REF, "+nipples", ""), "nipples")
        self.assertEqual(rr.resolve_negative(REF, "@master", "x"), REF["master"]["negative"])
        self.assertEqual(rr.resolve_negative(REF, "own", "x"), "own")
        self.assertIsNone(rr.resolve_negative(REF, None, "x"))
        self.assertIsNone(rr.resolve_negative(REF, "+nipples", None))  # a pipe without negative never gets one

    def test_distilled_models_get_no_negative(self):
        jobs = {j["id"]: j for j in rr.image_jobs(REF)}
        for mid in ("z-image-turbo", "flux2-klein-4b", "h3_image"):
            self.assertIsNone(jobs[mid]["negative"], mid)
        self.assertTrue(jobs["noobai-xl"]["negative"].endswith("nipples, smile, tongue, text, logo"))


class Grid(unittest.TestCase):
    def test_grid_lists_results_errors_and_embeds(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            Image.new("RGB", (70, 90), "red").save(run / "z-image-turbo.png")
            (run / "z-image-turbo.json").write_text(json.dumps({"model": "z-image-turbo", "kind": "image", "style": "photo",
                                                                "file": "z-image-turbo.png", "seconds": 4.2}))
            (run / "video").mkdir()
            (run / "video" / "ltx23.json").write_text(json.dumps({"model": "ltx23", "kind": "video", "file": None,
                                                                  "error": "engine <down>"}))
            index, grid = rr.write_grid(run)
            html = index.read_text(encoding="utf-8")
            self.assertIn('src="z-image-turbo.png"', html)
            self.assertIn("engine &lt;down&gt;", html)
            self.assertLess(html.index("z-image-turbo"), html.index("ltx23"))
            self.assertIn("data:image/jpeg;base64,", grid.read_text(encoding="utf-8"))
            self.assertNotIn('src="z-image-turbo.png"', grid.read_text(encoding="utf-8"))


class Cli(unittest.TestCase):
    def test_unknown_model_is_refused(self):
        self.assertEqual(rr.main(["--model", "nope", "--dry"]), 2)

    def test_dry_single_model(self):
        self.assertEqual(rr.main(["--model", "h3_10eros", "--dry"]), 0)


if __name__ == "__main__":
    unittest.main()
