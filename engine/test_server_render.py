"""server.py's Playground render path without the GPU: the engine runner is replaced by a replay of the progress lines a
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
                        mock.patch.object(server.engine_runner, "wait_for_ram", lambda stopped: None)]
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
        def run(work, on_event, stopped):
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
        with mock.patch.object(server.engine_runner, "run", self.fake_run(seen)):
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
        with mock.patch.object(server.engine_runner, "run", self.fake_run(seen, stop_at=6)):
            with self.assertRaisesRegex(ValueError, "Stopped"):
                self.handler.playground_render({"id": self.play, "_job": self.job})
        self.assertEqual(playground.clips(), [])

    def test_missing_lora_never_reaches_the_engine(self):
        playground.save({"id": self.play, "loras": [{"path": str(self.tmp / "gone.safetensors"), "strength": 1}, {"path": "", "strength": 1}]})
        seen, self.trace, self.job = [], [], {}
        with mock.patch.object(server.engine_runner, "run", self.fake_run(seen)):
            with self.assertRaisesRegex(ValueError, "gone.safetensors"):
                self.handler.playground_render({"id": self.play, "_job": self.job})
        self.assertEqual(seen, [])

    def test_upscale_makes_a_new_clip_linked_to_the_source(self):
        seen, self.trace, self.job = [], [], {}
        with mock.patch.object(server.engine_runner, "run", self.fake_run(seen)):
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


if __name__ == "__main__":
    unittest.main()
