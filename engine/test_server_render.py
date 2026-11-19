"""server.py's Playground render path without the GPU: the WSL engine runner is replaced by a replay of the progress lines a
real LTX run printed (2026-11-13), ComfyUI stop and the RAM wait are no-ops. python -m unittest engine.test_server_render"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import playground
import server

# what the worker printed in the real LTX 720p 2 s run (main 8 + refine 3), trimmed
REPLAY = [("loading_model", None, None), ("encoding_text", 0, 48), ("encoding_text", 48, 48), ("inference", None, None),
          *[("inference", i, 8) for i in range(8)], ("decoding", 8, 8),
          *[("inference", i, 3) for i in range(3)], ("decoding", 3, 3), ("decoding", 0, 7), ("decoding", 7, 7),
          ("decoding", None, None), ("inference", None, None)]


class RenderPath(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(playground, "ROOT", self.tmp / "playground"),
                        mock.patch.object(playground.morpheus, "OUTPUT", self.tmp),
                        mock.patch.object(server.morpheus, "stop_comfy", lambda: None),
                        mock.patch.object(server.engine_wsl, "wait_for_ram", lambda stopped: None),
                        mock.patch.object(server.engine_wsl, "release", lambda: None)]
        for p in self.patches:
            p.start()
        self.play = playground.new({})["id"]
        playground.save({"id": self.play, "prompt": "A fox runs.", "duration": 2})
        self.handler = server.Handler.__new__(server.Handler)  # only its methods are used

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_run(self, seen: list, stop_at=None):
        def run(work, on_event, stopped, keep_warm=None):
            seen.append(work)
            for n, (phase, step, total) in enumerate(REPLAY):
                if stop_at is not None and n == stop_at:
                    self.job["_stop"] = True
                if stopped():
                    raise ValueError("Stopped.")
                on_event({"event": "progress", "phase": phase, "step": step, "total": total})
                self.trace.append((self.job.get("phase"), self.job.get("step"), self.job.get("total")))
            out = self.tmp / "out.mp4"
            out.write_bytes(b"not really a video")
            return {"event": "done", "file": str(out), "seconds": 188.4}
        return run

    def test_render_lands_in_the_shelf_with_summed_steps(self):
        seen, self.trace, self.job = [], [], {}
        with mock.patch.object(server.engine_wsl, "run", self.fake_run(seen)):
            meta = self.handler.playground_render({"id": self.play, "_job": self.job})
        self.assertEqual(seen[0]["kind"], "render")
        self.assertEqual(seen[0]["settings"]["model_type"], "ltx2_22B_dev_q6")
        self.assertEqual(seen[0]["settings"]["video_length"], 49)
        steps = [s for _, s, _ in self.trace]
        self.assertEqual(max(steps), 10)  # the last refine step reported is 2 of 3 (0-based) → 8 + 2
        self.assertEqual(steps, sorted(steps))  # the bar never runs backwards across main → refine
        self.assertEqual({t for _, _, t in self.trace}, {11})  # LTX: 8 main + 3 refine, as the schema says
        self.assertIn("Rendering", {p for p, _, _ in self.trace})
        self.assertEqual(meta["render_seconds"], 188.4)
        self.assertEqual(meta["seed"], seen[0]["settings"]["seed"])
        clip = playground.folder(self.play) / meta["video"]
        self.assertTrue(clip.is_file() and clip.with_suffix(".json").is_file())
        self.assertEqual(len(playground.clips()), 1)

    def test_stop_ends_the_job_without_a_clip(self):
        seen, self.trace, self.job = [], [], {}
        with mock.patch.object(server.engine_wsl, "run", self.fake_run(seen, stop_at=6)):
            with self.assertRaisesRegex(ValueError, "Stopped"):
                self.handler.playground_render({"id": self.play, "_job": self.job})
        self.assertEqual(playground.clips(), [])

    def test_missing_lora_never_reaches_the_engine(self):
        # a library entry whose file went missing from models/loras/video after it was picked
        import lora_library
        with mock.patch.object(lora_library, "DIR", self.tmp / "loras"):
            lora_library.write_entries([{"id": "gone", "name": "Gone LoRA", "family": "ltx2", "file": "gone.safetensors",
                                         "strength_min": 0.5, "strength_max": 1.0}])
            playground.save({"id": self.play, "loras": [{"id": "gone", "strength": 1}, {"id": "", "strength": 1}]})
            seen, self.trace, self.job = [], [], {}
            with mock.patch.object(server.engine_wsl, "run", self.fake_run(seen)), self.assertRaisesRegex(ValueError, "Gone LoRA"):
                self.handler.playground_render({"id": self.play, "_job": self.job})
        self.assertEqual(seen, [])

    def test_upscale_makes_a_new_clip_linked_to_the_source(self):
        seen, self.trace, self.job = [], [], {}
        with mock.patch.object(server.engine_wsl, "run", self.fake_run(seen)):
            src = self.handler.playground_render({"id": self.play, "_job": self.job})
            video = f"playground/{self.play}/{src['video']}"
            self.job = {}
            up = self.handler.playground_upscale({"id": self.play, "video": video, "method": "relayvsr", "factor": 1.5, "_job": self.job})
        self.assertEqual(seen[-1], {"kind": "upscale", "video": str(playground.folder(self.play) / src["video"]), "upsampling": "relayvsr*1.5"})
        self.assertEqual(up["upscale"]["source"], video)
        self.assertIn("↑ RelayVSR ×1.5", up["resolution"]["label"])
        self.assertEqual(len(playground.clips()), 2)

    def test_upscale_refuses_a_bad_factor(self):
        with self.assertRaises(ValueError):
            self.handler.playground_upscale({"id": self.play, "video": "x.mp4", "method": "relayvsr", "factor": 1.534})




class ResharpenRender(unittest.TestCase):
    """Glia seed muse_resharpen stage 2: server.image_render skips the engine for "Swap only" and swaps the persona's face
    onto every result; without a persona nothing is swapped. Engine and face swap are fakes."""
    PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    def setUp(self):
        import image_playground as ip
        self.ip, self.tmp = ip, Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(playground, "ROOT", self.tmp / "playground"), mock.patch.object(ip, "ROOT", self.tmp / "playground"),
                        mock.patch.object(playground.morpheus, "OUTPUT", self.tmp)]
        for p in self.patches:
            p.start()
        self.play = ip.new({})["id"]
        self.handler = server.Handler.__new__(server.Handler)
        self.swaps, self.engine = [], []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _render(self, mode, persona):
        self.ip.input_picture({"id": self.play, "slot": 0, "data": self.PNG})
        if persona:
            self.ip.input_picture({"id": self.play, "slot": 1, "data": self.PNG})
        self.ip.save({"id": self.play, "model": "muse_resharpen", "prompt": "", "params": {"mode": mode}})

        def engine_job(job, work, steps, engine=None):
            self.engine.append(work)
            out = self.tmp / "h3.png"
            shutil.copyfile(self.ip.playground.folder(self.play) / self.ip.load(self.play)["inputs"][0], out)
            return {"files": [str(out)], "seeds": [7], "seconds": 3}
        with mock.patch.object(self.handler, "engine_job", engine_job, create=True),                 mock.patch.object(server, "swap_picture", lambda per, pic: self.swaps.append((per.name, pic.exists()))):
            return self.handler.image_render({"id": self.play, "_job": {}})

    def test_swap_only_skips_the_engine_and_swaps(self):
        out = self._render("swap", persona=True)
        self.assertEqual(self.engine, [])
        self.assertEqual(len(self.swaps), 1)
        self.assertTrue(self.swaps[0][1])
        self.assertEqual(len(out["images"]), 1)
        meta = out["images"][0]
        self.assertEqual((meta["helper"], meta["persona"]), ("resharpen", self.ip.load(self.play)["inputs"][1]))

    def test_max_renders_then_swaps(self):
        self._render("max", persona=True)
        self.assertEqual(self.engine[0]["family"], "h3_image")
        self.assertEqual(len(self.swaps), 1)

    def test_no_persona_no_swap(self):
        self._render("max", persona=False)
        self.assertEqual(len(self.engine), 1)
        self.assertEqual(self.swaps, [])


if __name__ == "__main__":
    unittest.main()
