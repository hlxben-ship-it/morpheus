"""Tests of the snapshot -> WanGP settings mapping (no GPU, no torch): python -m unittest engine.test_jobs (Morpheus root)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import playground
from engine import jobs

DIR = Path("C:/play")


def snap(model="ltx23", steps=8, res="720p", aspect="16:9", start="start_1.png", end=None, loras=(), render=None, prompt="A fox runs.",
         refs=(None,) * 4, main_first=True, last_ref=None):
    r = {"id": res, "label": res, "target": res}
    if render:
        r["render"] = render
    state = {"start": start, "end": end, "refs": list(refs), "main_first": main_first, "last_ref": last_ref}
    return {"model": {"id": model, "label": model}, "prompt": prompt, "params": {"main.steps": steps}, "resolution": r,
            "aspect": aspect, "duration": 3, "start_frame": start, "end_frame": end, "loras": list(loras),
            "pictures": playground.h3_pictures(state)}


class Sizes(unittest.TestCase):
    def test_every_size_fits_ltx(self):
        for res, table in jobs.SIZES.items():
            for aspect, (w, h) in table.items():
                with self.subTest(res=res, aspect=aspect):
                    self.assertEqual((w % 64, h % 64), (0, 0))
                    self.assertLessEqual(w * h, jobs.LTX_BUDGET)
                    a, b = (int(x) for x in aspect.split(":"))
                    self.assertAlmostEqual(w / h, a / b, delta=0.1)

    def test_480p_draft(self):
        self.assertEqual(jobs.size("480p", "16:9"), (896, 512))
        self.assertEqual(jobs.render_settings(snap(res="480p"), DIR)["resolution"], "896x512")
        self.assertEqual(jobs.render_settings(snap(model="h3_minimax", res="480p", aspect="9:16"), DIR)["resolution"], "512x896")

    def test_spike_sizes(self):
        self.assertEqual(jobs.size("720p", "16:9"), (1280, 704))
        self.assertEqual(jobs.size("1080p", "16:9"), (1920, 1088))

    def test_no_2k(self):
        with self.assertRaises(ValueError):
            jobs.size("2k", "16:9")

    def test_frames_are_8n_plus_1(self):
        self.assertEqual([jobs.frames(s) for s in (3, 5, 10)], [73, 121, 241])
        for s in range(1, 11):
            self.assertEqual((jobs.frames(s) - 1) % 8, 0)


class Ltx(unittest.TestCase):
    def test_8_steps_is_the_spike_setup(self):
        s = jobs.render_settings(snap(), DIR, seed=7)
        self.assertEqual(s["model_type"], "ltx2_22B_dev_q6")
        self.assertEqual((s["sample_solver"], s["num_inference_steps"], s["guidance_phases"], s["guidance_scale"]), ("distilled_8_steps", 8, 2, 1.0))
        self.assertEqual((s["resolution"], s["video_length"], s["force_fps"], s["seed"]), ("1280x704", 73, 24, 7))
        self.assertEqual(s["image_prompt_type"], "S")
        self.assertEqual(Path(s["image_start"]), DIR / "start_1.png")
        self.assertNotIn("activated_loras", s)  # WanGP loads the distill LoRA itself at 8
        self.assertEqual(s["prompt"], "A fox runs.")

    def test_more_steps_is_the_dev_schedule_with_the_distill_lora_in_phase_2(self):
        s = jobs.render_settings(snap(steps=20, loras=[{"path": "C:/l/a.safetensors", "strength": 0.8, "name": "a"}]), DIR)
        self.assertEqual((s["sample_solver"], s["num_inference_steps"], s["guidance_scale"]), ("euler", 20, 3.0))
        self.assertEqual(s["activated_loras"], [jobs.DISTILL_LORA, "C:/l/a.safetensors"])
        self.assertEqual(s["loras_multipliers"], "0;1 0.8")

    def test_steps_out_of_range(self):
        for bad in (7, 31):
            with self.assertRaises(ValueError):
                jobs.render_settings(snap(steps=bad), DIR)

    def test_start_and_end(self):
        s = jobs.render_settings(snap(end="end_1.png"), DIR)
        self.assertEqual(s["image_prompt_type"], "SE")
        self.assertEqual(Path(s["image_end"]), DIR / "end_1.png")

    def test_refs_are_h3_only(self):
        s = jobs.render_settings(snap(refs=("r1.png", None, None, None), last_ref=0), DIR)
        self.assertNotIn("image_refs", s)
        self.assertEqual(s["image_prompt_type"], "S")

    def test_text_only(self):
        s = jobs.render_settings(snap(start=None), DIR)
        self.assertEqual(s["image_prompt_type"], "")
        self.assertNotIn("image_start", s)

    def test_loras_and_strengths(self):
        s = jobs.render_settings(snap(loras=[{"path": "C:/l/a.safetensors", "strength": 1.0, "name": "a"},
                                             {"path": "C:/l/b.safetensors", "strength": 0.55, "name": "b"}]), DIR)
        self.assertEqual(s["activated_loras"], ["C:/l/a.safetensors", "C:/l/b.safetensors"])
        self.assertEqual(s["loras_multipliers"], "1 0.55")

    def test_missing_lora_stops_before_loading(self):
        with self.assertRaisesRegex(ValueError, "b.safetensors"):
            jobs.render_settings(snap(loras=[{"path": "C:/l/b.safetensors", "strength": 1, "name": "b.safetensors", "missing": True}]), DIR)

    def test_upscaled_resolution_refused(self):
        with self.assertRaisesRegex(ValueError, "Upscale"):
            jobs.render_settings(snap(res="1080p", render="720p"), DIR)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            jobs.render_settings(snap(model="wan22"), DIR)


class H3(unittest.TestCase):
    def test_spike_setup(self):
        s = jobs.render_settings(snap(model="h3_minimax", steps=4), DIR, seed=1)
        self.assertEqual(s["model_type"], "minimax_h3_ref2va_hyperflow")
        self.assertEqual((s["num_inference_steps"], s["sample_solver"], s["guidance_phases"], s["config"]), (4, "euler", 1, "distilled_4b,lightvae_int8"))
        self.assertEqual(s["image_refs"], [s["image_start"]])
        self.assertEqual(s["video_prompt_type"], "I")
        self.assertTrue(s["prompt"].startswith("For the target video, at 0.00 seconds"))
        self.assertIn("integrated_multimodal_description: [Shot 1] A fox runs.", s["prompt"])

    def test_frames_follow_the_h3_grid_without_the_floor(self):
        self.assertEqual([jobs.h3_frames(s) for s in (2, 3, 5, 10)], [56, 73, 124, 243])
        for s in range(1, 11):
            n = jobs.h3_frames(s)
            self.assertEqual((n - 5) % 17, 0)
            self.assertGreaterEqual(n, jobs.frames(s))
        short = jobs.render_settings(snap(model="h3_minimax"), DIR)  # 3 s: below WanGP's 107, the worker lowers the floor
        self.assertEqual((short["video_length"], short["_frames_minimum"]), (73, 73))
        long = jobs.render_settings(dict(snap(model="h3_minimax"), duration=5), DIR)
        self.assertEqual(long["video_length"], 124)
        self.assertNotIn("_frames_minimum", long)

    def test_steps(self):
        for ok in (4, 6, 8):
            self.assertEqual(jobs.render_settings(snap(model="h3_minimax", steps=ok), DIR)["num_inference_steps"], ok)
        for bad in (5, 10):
            with self.assertRaises(ValueError):
                jobs.render_settings(snap(model="h3_minimax", steps=bad), DIR)

    def test_no_picture_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "main image or a reference"):
            jobs.render_settings(snap(model="h3_minimax", start=None), DIR)

    def test_stored_end_frame_is_not_sent(self):  # H3 has no End slot; an LTX end stays saved, unused
        s = jobs.render_settings(snap(model="h3_minimax", end="end_1.png"), DIR)
        self.assertEqual(s["image_prompt_type"], "S")
        self.assertNotIn("image_end", s)

    def test_seed_example(self):  # main = first frame, Ref 2 = last frame, Ref 1 + 3 filled
        s = jobs.render_settings(snap(model="h3_minimax", refs=("r1.png", "r2.png", "r3.png", None), last_ref=1), DIR)
        self.assertEqual(s["image_prompt_type"], "SE")
        self.assertEqual(Path(s["image_start"]), DIR / "start_1.png")
        self.assertEqual(Path(s["image_end"]), DIR / "r2.png")
        self.assertEqual([Path(x).name for x in s["image_refs"]], ["r1.png", "r3.png"])  # each image once
        self.assertIn("Picture 1 (from Shot 1)", s["prompt"])
        self.assertIn("At the end of the target video, Picture 2", s["prompt"])

    def test_main_as_reference_only(self):
        s = jobs.render_settings(snap(model="h3_minimax", main_first=False, refs=(None, "r2.png", None, None)), DIR)
        self.assertEqual(s["image_prompt_type"], "")
        self.assertNotIn("image_start", s)
        self.assertEqual([Path(x).name for x in s["image_refs"]], ["start_1.png", "r2.png"])
        self.assertTrue(s["prompt"].startswith("integrated_multimodal_description"))

    def test_last_frame_without_main_is_picture_1(self):
        s = jobs.render_settings(snap(model="h3_minimax", start=None, refs=("r1.png", None, None, None), last_ref=0), DIR)
        self.assertEqual(s["image_prompt_type"], "E")
        self.assertEqual(s["image_refs"], [s["image_end"]])  # the only anchor repeated as the reference
        self.assertIn("At the end of the target video, Picture 1", s["prompt"])

    def test_structured_prompt_untouched(self):
        p = "subject_definitions:\n<Subject 1> is ...\ndetailed_description:\n[Shot 1] ..."
        self.assertEqual(jobs.render_settings(snap(model="h3_minimax", prompt=p), DIR)["prompt"], p)


class H3Pictures(unittest.TestCase):
    def names(self, **state):
        return [(p["slot"], p["role"], p["tag"], p["shown"]) for p in playground.h3_pictures(state)]

    def test_seed_example(self):
        self.assertEqual(self.names(start="m.png", refs=["a", "b", "c", None], last_ref=1), [
            ("main", "start", "<Picture 1>", True), ("ref2", "end", "<Picture 2>", True),
            ("ref1", "ref", "<Picture 3>", True), ("ref3", "ref", "<Picture 4>", True)])

    def test_first_frame_off(self):
        self.assertEqual(self.names(start="m.png", main_first=False, refs=[None, "b", None, "d"]), [
            ("main", "ref", "<Picture 1>", True), ("ref2", "ref", "<Picture 2>", True), ("ref4", "ref", "<Picture 3>", True)])

    def test_no_last_frame_refs_with_gaps(self):
        self.assertEqual(self.names(start="m.png", refs=[None, None, "c", None]), [
            ("main", "start", "<Picture 1>", True), ("ref3", "ref", "<Picture 2>", True)])

    def test_only_anchors_repeat_the_first_as_hidden_reference(self):
        self.assertEqual(self.names(start="m.png", refs=[None, "b", None, None], last_ref=1), [
            ("main", "start", "<Picture 1>", True), ("ref2", "end", "<Picture 2>", True), ("main", "ref", "<Picture 3>", False)])

    def test_last_frame_on_an_empty_slot_is_ignored(self):
        self.assertEqual(self.names(start="m.png", refs=["a", None, None, None], last_ref=2), [
            ("main", "start", "<Picture 1>", True), ("ref1", "ref", "<Picture 2>", True)])

    def test_nothing(self):
        self.assertEqual(playground.h3_pictures({"start": None, "refs": [None] * 4}), [])
        self.assertEqual(playground.h3_pictures({}), [])


PNG = "data:image/png;base64,iVBORw0KGgo="


class RefState(unittest.TestCase):
    """The ref slots in a real playground folder (temp): store, toggles, removal, what is saved."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(playground, "ROOT", self.tmp)
        self.patch.start()
        self.id = playground.new({})["id"]

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fresh_state(self):
        st = playground.load(self.id)
        self.assertEqual((st["main_first"], st["refs"], st["last_ref"], st["pictures"]), (True, [None] * 4, None, []))

    def test_ref_last_frame_and_removal(self):
        st = playground.ref({"id": self.id, "slot": 2, "data": PNG})
        self.assertTrue(st["refs"][2].startswith("ref3_"))
        st = playground.save({"id": self.id, "last_ref": 2, "main_first": False})
        self.assertEqual((st["last_ref"], st["main_first"]), (2, False))
        self.assertEqual([(p["slot"], p["role"]) for p in st["pictures"]], [("ref3", "end"), ("ref3", "ref")])
        saved = json.loads((self.tmp / self.id / "playground.json").read_text(encoding="utf-8"))
        self.assertNotIn("pictures", saved)  # derived, never stored
        st = playground.ref({"id": self.id, "slot": 2, "data": None})
        self.assertEqual((st["refs"][2], st["last_ref"]), (None, None))

    def test_bad_input(self):
        for bad in (4, -1, "1", True, None):
            with self.subTest(slot=bad), self.assertRaises(ValueError):
                playground.ref({"id": self.id, "slot": bad, "data": PNG})
        with self.assertRaises(ValueError):
            playground.save({"id": self.id, "last_ref": 5})
        with self.assertRaises(ValueError):
            playground.ref({"id": self.id, "slot": 0, "data": "data:text/plain;base64,eA=="})

    def test_deleted_ref_file_is_gone(self):
        name = playground.ref({"id": self.id, "slot": 0, "data": PNG})["refs"][0]
        playground.save({"id": self.id, "last_ref": 0})
        (self.tmp / self.id / name).unlink()
        st = playground.load(self.id)
        self.assertEqual((st["refs"][0], st["last_ref"]), (None, None))


class Upscale(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(jobs.upscale_request("relayvsr", 1.5), "relayvsr*1.5")
        self.assertEqual(jobs.upscale_request("seedvr2", "2"), "seedvr2*2")

    def test_refused(self):
        for method, factor in (("relayvsr", 1.534), ("relayvsr", 1), ("relayvsr", 4.5), ("flashvsr", 2), ("seedvr2", "x")):
            with self.subTest(method=method, factor=factor), self.assertRaises(ValueError):
                jobs.upscale_request(method, factor)


if __name__ == "__main__":
    unittest.main()
