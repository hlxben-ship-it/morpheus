"""Studio (seed combined_playground): video half at the top level, image half under "image". Run: python -m unittest engine.test_studio"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import image_models as im
import image_playground as ip
import morpheus
import playground
import playground_models as pm
import studio

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class Studio(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p = [mock.patch.object(playground, "ROOT", self.tmp), mock.patch.object(ip, "ROOT", self.tmp),
                  mock.patch.object(morpheus, "OUTPUT", self.tmp.parent)]
        for x in self.p:
            x.start()
        self.id = studio.new({})["id"]

    def tearDown(self):
        for x in self.p:
            x.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def raw(self):
        return json.loads((self.tmp / self.id / "playground.json").read_text(encoding="utf-8"))

    def add_picture(self, name="20260101_000000_1"):
        d = self.tmp / self.id / "images"
        d.mkdir(parents=True, exist_ok=True)
        import base64
        (d / f"{name}.png").write_bytes(base64.b64decode(PNG.split(",")[1]))
        (d / f"{name}.json").write_text(json.dumps({"created": "2026-01-01T00:00:00", "model": {"label": "x"}}), encoding="utf-8")
        return f"{self.tmp.name}/{self.id}/images/{name}.png"

    def test_new_has_both_halves(self):
        st = self.raw()
        self.assertEqual((st["kind"], st["side"]), ("studio", "video"))
        self.assertIn("model", st["image"])
        self.assertEqual(ip.load(self.id)["kind"], "image")  # the image half reads like an image playground
        self.assertEqual(playground.load(self.id)["kind"], "studio")

    def test_halves_do_not_overwrite_each_other(self):
        ip.save({"id": self.id, "prompt": "a picture"})
        playground.save({"id": self.id, "prompt": "a clip"})
        ip.save({"id": self.id, "count": 1})
        st = self.raw()
        self.assertEqual(st["prompt"], "a clip")
        self.assertEqual(st["image"]["prompt"], "a picture")
        self.assertEqual(st["kind"], "studio")

    def test_side_switch_takes_model(self):
        img = im.models()[-1]["id"]
        studio.side({"id": self.id, "side": "image", "model": img})
        self.assertEqual((self.raw()["side"], self.raw()["image"]["model"]), ("image", img))
        vid = pm.models()[-1]["id"]
        studio.side({"id": self.id, "side": "video", "model": vid})
        self.assertEqual((self.raw()["side"], self.raw()["model"]), ("video", vid))

    def test_side_refuses_bad_input(self):
        with self.assertRaises(ValueError):
            studio.side({"id": self.id, "side": "audio"})
        with self.assertRaises(ValueError):
            studio.side({"id": self.id, "side": "image", "model": pm.models()[0]["id"]})  # a video model on the image side
        with self.assertRaises(ValueError):
            studio.side({"id": ip.new({})["id"], "side": "video"})  # not a Studio

    def test_picture_to_start_frame(self):
        pic = self.add_picture()
        st = studio.picture_frame({"id": self.id, "image": pic, "kind": "start"})
        self.assertTrue(st["start"] and (self.tmp / self.id / st["start"]).is_file())
        with self.assertRaises(ValueError):
            studio.picture_frame({"id": self.id, "image": "elsewhere/x.png", "kind": "start"})

    def test_star_picture_keeps_studio(self):
        pic = self.add_picture()
        ip.star_image({"id": self.id, "image": pic, "starred": True})
        st = self.raw()
        self.assertEqual((st["kind"], st["cover"]), ("studio", "images/20260101_000000_1.png"))
        ip.star_image({"id": self.id, "image": pic, "starred": False})
        self.assertNotIn("cover", self.raw())

    def test_gallery_counts_clips_and_pictures(self):
        self.add_picture()
        row = next(r for r in playground.list_all() if r["id"] == self.id)
        self.assertEqual((row["kind"], row["clips"]), ("studio", 1))
        self.assertTrue(row["thumb"].endswith(".png"))

    # ---- stage 2: stateful tiles ----
    def add_clip(self, prompt="a clip", **extra):
        """A rendered clip as add_clip writes it, from the current video state (no real mp4 needed for the settings)."""
        snap = playground.snapshot(self.id)
        snap.update(prompt=prompt, studio_prompt=prompt, **extra)
        src = self.tmp / "x.mp4"
        src.write_bytes(b"\0")
        return playground.add_clip(self.id, src, snap, seed=7)["video"]

    def test_new_studio_is_clean_draft(self):
        i = studio.info({"id": self.id})
        self.assertEqual((i["open"], i["dirty"], i["edited"]), ("new", False, []))

    def test_edit_marks_dirty_remix_does_not(self):
        studio.open_tile({"id": self.id, "tile": self.add_clip()})
        self.assertFalse(studio.info({"id": self.id})["dirty"])
        playground.save({"id": self.id, "remix": []})
        self.assertFalse(studio.info({"id": self.id})["dirty"])
        playground.save({"id": self.id, "prompt": "changed"})
        self.assertTrue(studio.info({"id": self.id})["dirty"])

    def test_open_tile_loads_rendered_and_keeps_draft(self):
        playground.save({"id": self.id, "prompt": "my draft"})
        clip = self.add_clip("rendered prompt")
        studio.open_tile({"id": self.id, "tile": clip})
        self.assertEqual(self.raw()["prompt"], "rendered prompt")
        studio.open_tile({"id": self.id, "tile": "new"})
        self.assertEqual(self.raw()["prompt"], "my draft")

    def test_edited_copy_survives_switch_and_back_resets(self):
        clip = self.add_clip("rendered prompt")
        studio.open_tile({"id": self.id, "tile": clip})
        playground.save({"id": self.id, "prompt": "tweaked"})
        studio.open_tile({"id": self.id, "tile": "new"})
        self.assertEqual(studio.info({"id": self.id})["edited"], [clip])
        studio.open_tile({"id": self.id, "tile": clip})
        self.assertEqual((self.raw()["prompt"], self.raw()["dirty"]), ("tweaked", True))
        studio.back({"id": self.id})
        self.assertEqual((self.raw()["prompt"], self.raw()["dirty"]), ("rendered prompt", False))
        studio.open_tile({"id": self.id, "tile": "new"})
        with self.assertRaises(ValueError):
            studio.back({"id": self.id})  # the draft has no rendered state

    def test_old_clip_with_stale_params_opens(self):
        clip = self.add_clip("old", params={**playground.load(self.id)["params"], "length": 3})
        studio.open_tile({"id": self.id, "tile": clip})
        self.assertEqual(self.raw()["prompt"], "old")
        self.assertNotIn("length", self.raw()["params"])

    def test_rename_tile(self):
        clip = self.add_clip("A")
        self.assertEqual(studio.rename({"id": self.id, "tile": clip, "name": "  Forest   take 1 "})["names"], {clip: "Forest take 1"})
        self.assertFalse(studio.info({"id": self.id})["dirty"])  # a name is no settings edit
        self.assertEqual(studio.rename({"id": self.id, "tile": clip, "name": ""})["names"], {})
        self.assertEqual(studio.rename({"id": self.id, "tile": "new", "name": "my draft"})["names"], {"new": "my draft"})
        with self.assertRaises(ValueError):
            studio.rename({"id": self.id, "tile": "clips/gone.mp4", "name": "x"})

    # ---- stage 3: layout config + default prompt / negative ----
    def test_config_covers_every_model(self):
        import studio_models as sm
        ms = sm.models()
        self.assertEqual(set(ms), {m["id"] for m in pm.models()} | {m["id"] for m in im.models()})
        self.assertEqual(ms["ltx23"]["references"]["slots"], 1)
        self.assertTrue(ms["janku_v5"]["prompt"]["negative"])
        self.assertTrue(ms["janku_v5"]["prompt"]["default"])  # from image_models.json's prompt_default
        self.assertFalse(ms["h3_image"]["prompt"]["negative"])

    def test_config_refuses_bad_values(self):
        import studio_models as sm
        with self.assertRaises(ValueError):
            sm.check({"references": {"slots": 99}}, "x")
        with self.assertRaises(ValueError):
            sm.check({"references": {"toggles": ["middle"]}}, "x")
        with self.assertRaises(ValueError):
            sm.clean_defaults({"prompt": 5})
        with self.assertRaises(ValueError):
            sm.clean_defaults({"other": "x"})

    def test_video_default_prompt_joined_at_render(self):
        playground.save({"id": self.id, "prompt": "she waves", "defaults": {"prompt": "cinematic, 35mm", "negative": None}})
        snap = playground.snapshot(self.id)
        self.assertEqual((snap["prompt"], snap["studio_prompt"]), ("cinematic, 35mm, she waves", "she waves"))
        self.assertEqual(self.raw()["prompt"], "she waves")  # the prompt itself stays clean

    def test_studio_image_half_starts_clean_and_joins(self):
        ip.save({"id": self.id, "model": "janku_v5"})
        st = self.raw()["image"]
        self.assertEqual((st["prompt"], st["negative"]), ("", ""))
        ip.save({"id": self.id, "prompt": "a cat"})
        snap = ip.snapshot(self.id)
        self.assertTrue(snap["prompt"].endswith(", a cat") and snap["prompt"].startswith("masterpiece"))
        self.assertTrue(snap["negative"])  # JANKU's default negative
        self.assertEqual(snap["studio_prompt"], "a cat")

    def test_model_switch_drops_tile_defaults(self):
        playground.save({"id": self.id, "defaults": {"prompt": "x", "negative": None}})
        other = [m["id"] for m in pm.models() if m["id"] != self.raw()["model"]][0]
        playground.save({"id": self.id, "model": other})
        self.assertNotIn("defaults", self.raw())

    def test_reopen_gives_prompt_without_defaults(self):
        playground.save({"id": self.id, "prompt": "she waves", "defaults": {"prompt": "cinematic", "negative": None}})
        snap = playground.snapshot(self.id)
        src = self.tmp / "x.mp4"; src.write_bytes(b"\0")
        clip = playground.add_clip(self.id, src, snap, seed=3)["video"]
        playground.save({"id": self.id, "prompt": "other"})
        studio.open_tile({"id": self.id, "tile": clip})
        st = self.raw()
        self.assertEqual((st["prompt"], st["defaults"]["prompt"]), ("she waves", "cinematic"))

    def test_old_playgrounds_unaffected_by_defaults(self):
        vid = playground.new({})["id"]
        playground.save({"id": vid, "prompt": "plain"})
        self.assertEqual(playground.snapshot(vid)["prompt"], "plain")
        img = ip.new({})["id"]
        self.assertTrue(ip.load(img)["prompt"].startswith("masterpiece"))  # the image playground still prefills its prompt

    # ---- stage 4: stills in the Remix timeline ----
    def test_remix_takes_stills_with_seconds(self):
        st = playground.save({"id": self.id, "remix": ["images/a.png@3", "images/b.jpg@0.5", "clips/x.mp4"]})
        self.assertEqual(st["remix"], ["images/a.png@3", "images/b.jpg@0.5", "clips/x.mp4"])
        for bad in (["images/a.png"], ["images/a.png@0.2"], ["images/a.png@31"], ["images/a.gif@3"], ["../a.png@3"]):
            with self.assertRaises(ValueError):
                playground.save({"id": self.id, "remix": bad})

    def test_remaster_refuses_stills(self):
        playground.save({"id": self.id, "remix": ["images/a.png@3"]})
        plan = playground.remaster_plan({"id": self.id})
        self.assertFalse(plan["ok"])
        self.assertIn("still", plan["reason"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not on PATH")
    def test_export_with_stills(self):
        import subprocess
        from PIL import Image
        d = playground.folder(self.id)
        (d / "clips").mkdir(exist_ok=True); (d / "images").mkdir(exist_ok=True)
        subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=24:d=1",
                        "-pix_fmt", "yuv420p", str(d / "clips" / "c.mp4")], check=True)
        Image.new("RGB", (200, 300), "blue").save(d / "images" / "p.png")
        playground.save({"id": self.id, "remix": ["clips/c.mp4", "images/p.png@2"]})
        out = playground.export_remix({"id": self.id})
        self.assertFalse(out["copied"])
        self.assertAlmostEqual(out["seconds"], 3, delta=0.2)
        info = playground.video_info(morpheus.OUTPUT / out["video"])
        self.assertEqual((info["w"], info["h"]), (320, 240))
        self.assertAlmostEqual(info["seconds"], 3, delta=0.2)
        # stills only: the first picture's frame at 24 fps
        playground.save({"id": self.id, "remix": ["images/p.png@1.5"]})
        info = playground.video_info(morpheus.OUTPUT / playground.export_remix({"id": self.id})["video"])
        self.assertEqual((info["w"], info["h"], round(info["fps"])), (200, 300, 24))

    def test_unknown_tile_refused(self):
        for bad in ("clips/nope.mp4", "../x.mp4", "images/x.jpg"):
            with self.assertRaises(ValueError):
                studio.open_tile({"id": self.id, "tile": bad})

    def test_render_from_draft_opens_new_tile_and_empties_draft(self):
        playground.save({"id": self.id, "prompt": "run me"})
        src = studio.run_source(self.id)
        clip = self.add_clip("run me")
        studio.rendered(self.id, src, [clip])
        i = studio.info({"id": self.id})
        self.assertEqual((i["open"], i["dirty"]), (clip, False))
        studio.open_tile({"id": self.id, "tile": "new"})
        self.assertEqual(self.raw()["prompt"], "")

    def test_render_from_tile_resets_source(self):
        a = self.add_clip("A")
        studio.open_tile({"id": self.id, "tile": a})
        playground.save({"id": self.id, "prompt": "A edited"})
        src = studio.run_source(self.id)
        b = self.add_clip("A edited")
        studio.rendered(self.id, src, [b])
        self.assertEqual(studio.info({"id": self.id})["edited"], [])
        studio.open_tile({"id": self.id, "tile": a})
        self.assertEqual(self.raw()["prompt"], "A")

    def test_render_elsewhere_leaves_editor(self):
        a = self.add_clip("A")
        src = studio.run_source(self.id)  # draft
        studio.open_tile({"id": self.id, "tile": a})  # user moves on while it renders
        b = self.add_clip("draft run")
        studio.rendered(self.id, src, [b])
        self.assertEqual(studio.info({"id": self.id})["open"], a)

    def test_banner_and_deleted_open_tile(self):
        a = self.add_clip("A")
        loopy = self.add_clip("A", loop={"source": "clips/" + a.split("/")[-1]})
        studio.open_tile({"id": self.id, "tile": loopy})
        self.assertTrue(studio.info({"id": self.id})["banner"].startswith("Loop of "))
        (self.tmp / self.id / loopy).unlink()
        self.assertEqual(studio.info({"id": self.id})["open"], "new")

    def test_open_picture_tile_switches_to_image_side(self):
        ip.save({"id": self.id, "prompt": "a picture prompt"})
        src = self.tmp / "p.png"
        import base64
        src.write_bytes(base64.b64decode(PNG.split(",")[1]))
        ref = ip.add_images(self.id, [str(src)], [5], ip.snapshot(self.id))[0]["image"]
        ip.save({"id": self.id, "prompt": "changed since"})
        studio.open_tile({"id": self.id, "tile": ref})
        st = self.raw()
        self.assertEqual((st["side"], st["image"]["prompt"], st["open"], st["dirty"]), ("image", "a picture prompt", ref, False))

    def test_run_source_none_for_other_kinds(self):
        self.assertIsNone(studio.run_source(ip.new({})["id"]))
        self.assertIsNone(studio.run_source(playground.new({})["id"]))

    def test_image_playground_unchanged(self):
        other = ip.new({})["id"]
        ip.save({"id": other, "prompt": "x"})
        st = json.loads((self.tmp / other / "playground.json").read_text(encoding="utf-8"))
        self.assertEqual((st["kind"], st["prompt"]), ("image", "x"))
        self.assertNotIn("image", st)


if __name__ == "__main__":
    unittest.main()
