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

    def test_new_studio_opens_one_empty_draft(self):
        i = studio.info({"id": self.id})
        self.assertEqual((i["open"], i["readonly"]), ("draft:1", False))
        self.assertEqual(i["drafts"], [{"ref": "draft:1", "label": "Draft", "from": None}])

    def test_rendered_tile_opens_read_only_with_its_settings(self):
        playground.save({"id": self.id, "prompt": "my draft"})
        clip = self.add_clip("rendered prompt")
        i = studio.open_tile({"id": self.id, "tile": clip})
        self.assertEqual((i["open"], i["readonly"], self.raw()["prompt"]), (clip, True, "rendered prompt"))
        studio.open_tile({"id": self.id, "tile": "draft:1"})
        self.assertEqual(self.raw()["prompt"], "my draft")  # the draft kept its edits across the switch

    def test_use_on_new_makes_a_draft_from_the_tile(self):
        playground.save({"id": self.id, "prompt": "first draft"})
        clip = self.add_clip("rendered prompt")
        studio.open_tile({"id": self.id, "tile": clip})
        i = studio.new_draft({"id": self.id, "tile": clip})
        self.assertEqual((i["open"], i["readonly"]), ("draft:2", False))
        self.assertEqual(i["drafts"][0], {"ref": "draft:2", "label": "Draft · from " + clip.split("/")[-1], "from": clip})
        self.assertEqual(self.raw()["prompt"], "rendered prompt")
        playground.save({"id": self.id, "prompt": "second draft"})
        studio.new_draft({"id": self.id})  # "+ New": a third, empty one
        self.assertEqual(self.raw()["prompt"], "")
        studio.open_tile({"id": self.id, "tile": "draft:2"})
        self.assertEqual(self.raw()["prompt"], "second draft")
        studio.open_tile({"id": self.id, "tile": "draft:1"})
        self.assertEqual(self.raw()["prompt"], "first draft")
        self.assertEqual([d["ref"] for d in studio.info({"id": self.id})["drafts"]], ["draft:3", "draft:2", "draft:1"])

    def test_use_on_new_refuses_drafts_and_missing_tiles(self):
        for bad in ("draft:1", "clips/gone.mp4", "../x.mp4"):
            with self.assertRaises(ValueError):
                studio.new_draft({"id": self.id, "tile": bad})
        self.assertEqual(len(studio.info({"id": self.id})["drafts"]), 1)

    def test_delete_draft(self):
        studio.new_draft({"id": self.id})
        playground.save({"id": self.id, "prompt": "two"})
        studio.open_tile({"id": self.id, "tile": "draft:1"})
        i = studio.delete_draft({"id": self.id, "tile": "draft:2"})  # not the open one: only it goes
        self.assertEqual((i["open"], [d["ref"] for d in i["drafts"]]), ("draft:1", ["draft:1"]))
        i = studio.delete_draft({"id": self.id, "tile": "draft:1"})  # the last one, open: a fresh empty draft takes over
        self.assertEqual((i["open"], [d["ref"] for d in i["drafts"]], self.raw()["prompt"]), ("draft:3", ["draft:3"], ""))

    def test_delete_refuses_rendered_tiles_and_unknown_drafts(self):
        clip = self.add_clip("A")
        for bad in (clip, "draft:9"):
            with self.assertRaises(ValueError):
                studio.delete_draft({"id": self.id, "tile": bad})
        self.assertTrue((self.tmp / self.id / clip).is_file())

    def test_migrates_old_edited_copies_into_drafts(self):
        a, b = self.add_clip("A"), self.add_clip("B")
        st = self.raw()
        for k in ("drafts", "next_draft"):
            st.pop(k)
        st.update(open=a, dirty=True, prompt="A tweaked", names={"new": "loose one"},
                  tiles={"new": {**studio.settings_of(st), "prompt": "loose"}, b: {**studio.settings_of(st), "prompt": "B tweaked"}})
        (self.tmp / self.id / "playground.json").write_text(json.dumps(st), encoding="utf-8")
        i = studio.info({"id": self.id})
        got = {d["from"]: d for d in i["drafts"]}
        self.assertEqual(set(got), {None, a, b})
        self.assertEqual(got[None]["label"], "loose one")
        self.assertEqual(i["open"], got[a]["ref"])  # the open edited render became an open draft
        self.assertEqual(self.raw()["prompt"], "A tweaked")
        studio.open_tile({"id": self.id, "tile": got[b]["ref"]})
        self.assertEqual(self.raw()["prompt"], "B tweaked")
        studio.open_tile({"id": self.id, "tile": got[None]["ref"]})
        self.assertEqual(self.raw()["prompt"], "loose")
        self.assertNotIn("tiles", self.raw())

    def test_migrates_clean_render_open_read_only(self):
        a = self.add_clip("A")
        st = self.raw()
        for k in ("drafts", "next_draft"):
            st.pop(k)
        st.update(open=a, dirty=False, tiles={})
        (self.tmp / self.id / "playground.json").write_text(json.dumps(st), encoding="utf-8")
        i = studio.info({"id": self.id})
        self.assertEqual((i["open"], i["readonly"], len(i["drafts"])), (a, True, 1))

    def test_old_clip_with_stale_params_opens(self):
        clip = self.add_clip("old", params={**playground.load(self.id)["params"], "length": 3})
        studio.open_tile({"id": self.id, "tile": clip})
        self.assertEqual(self.raw()["prompt"], "old")
        self.assertNotIn("length", self.raw()["params"])

    def test_rename_tile(self):
        clip = self.add_clip("A")
        self.assertEqual(studio.rename({"id": self.id, "tile": clip, "name": "  Forest   take 1 "})["names"], {clip: "Forest take 1"})
        self.assertEqual(studio.rename({"id": self.id, "tile": clip, "name": ""})["names"], {})
        i = studio.rename({"id": self.id, "tile": "draft:1", "name": "my draft"})
        self.assertEqual((i["names"], i["drafts"][0]["label"]), ({"draft:1": "my draft"}, "my draft"))
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

    def test_render_only_from_a_draft_and_the_draft_stays(self):
        playground.save({"id": self.id, "prompt": "run me"})
        self.assertEqual(studio.run_source(self.id), "draft:1")
        self.add_clip("run me")  # the render lands as a new tile; nothing moves in the editor (Q1)
        i = studio.info({"id": self.id})
        self.assertEqual((i["open"], self.raw()["prompt"]), ("draft:1", "run me"))
        clip = self.add_clip("other")
        studio.open_tile({"id": self.id, "tile": clip})
        with self.assertRaises(ValueError):
            studio.run_source(self.id)  # read-only: no render from a rendered tile

    def test_banner_and_deleted_open_tile(self):
        a = self.add_clip("A")
        loopy = self.add_clip("A", loop={"source": "clips/" + a.split("/")[-1]})
        studio.open_tile({"id": self.id, "tile": loopy})
        self.assertTrue(studio.info({"id": self.id})["banner"].startswith("Loop of "))
        (self.tmp / self.id / loopy).unlink()
        self.assertEqual(studio.info({"id": self.id})["open"], "draft:1")

    def test_open_picture_tile_switches_to_image_side(self):
        ip.save({"id": self.id, "prompt": "a picture prompt"})
        src = self.tmp / "p.png"
        import base64
        src.write_bytes(base64.b64decode(PNG.split(",")[1]))
        ref = ip.add_images(self.id, [str(src)], [5], ip.snapshot(self.id))[0]["image"]
        ip.save({"id": self.id, "prompt": "changed since"})
        studio.open_tile({"id": self.id, "tile": ref})
        st = self.raw()
        self.assertEqual((st["side"], st["image"]["prompt"], st["open"]), ("image", "a picture prompt", ref))

    def test_run_source_none_for_other_kinds(self):
        self.assertIsNone(studio.run_source(ip.new({})["id"]))
        self.assertIsNone(studio.run_source(playground.new({})["id"]))

    def test_image_playground_unchanged(self):
        other = ip.new({})["id"]
        ip.save({"id": other, "prompt": "x"})
        st = json.loads((self.tmp / other / "playground.json").read_text(encoding="utf-8"))
        self.assertEqual((st["kind"], st["prompt"]), ("image", "x"))
        self.assertNotIn("image", st)


class ResharpenLast(unittest.TestCase):
    """Glia seed muse_resharpen stage 2: clip viewer "Resharpen last frame" fills the image half's Resharpen helper."""
    setUp, tearDown, raw = Studio.setUp, Studio.tearDown, Studio.raw
    def _clip(self, start=True):
        import base64
        d = self.tmp / self.id
        (d / "clips").mkdir(parents=True, exist_ok=True)
        meta = {"prompt": "<Subject 1> dances"}
        if start:
            (d / "start_1.png").write_bytes(base64.b64decode(PNG.split(",")[1]))
            meta["start_frame"] = "start_1.png"
        (d / "clips" / "c.mp4").write_bytes(b"x")
        (d / "clips" / "c.json").write_text(json.dumps(meta), encoding="utf-8")
        return f"{self.tmp.name}/{self.id}/clips/c.mp4"

    def _run(self, video):
        with mock.patch.object(playground, "last_frame", lambda req: {"data": PNG}):
            return studio.resharpen_last({"id": self.id, "video": video})

    def test_last_frame_and_persona_in_resharpen_on_the_image_side(self):
        st = self._run(self._clip())
        raw = self.raw()
        self.assertEqual(raw["side"], "image")
        img = ip.load(self.id)
        self.assertEqual(img["model"], "muse_resharpen")
        self.assertEqual(img["params"]["mode"], "max")
        self.assertEqual(img["prompt"], "")  # the video prompt is not taken over
        self.assertTrue(img["inputs"][0] and img["inputs"][1])
        snap = ip.snapshot(self.id)
        self.assertEqual(snap["persona"], img["inputs"][1])

    def test_clip_without_start_frame_has_no_persona(self):
        self._run(self._clip(start=False))
        img = ip.load(self.id)
        self.assertTrue(img["inputs"][0])
        self.assertIsNone(img["inputs"][1])

    def test_keeps_the_users_cheap_settings_but_picks_max(self):
        ip.save({"id": self.id, "model": "muse_resharpen", "params": {"mode": "cheap", "strength": 0.45}})
        self._run(self._clip())
        prm = ip.load(self.id)["params"]
        self.assertEqual((prm["mode"], prm["strength"]), ("max", 0.45))


if __name__ == "__main__":
    unittest.main()
