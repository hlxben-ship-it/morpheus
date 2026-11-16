"""lora_library + the Playground's library slots, on temp folders: no GPU, no WSL, no real LoRA."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lora_library as lib  # noqa: E402
import playground  # noqa: E402

LTX_KEYS = ["diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
            "diffusion_model.transformer_blocks.0.audio_attn1.to_k.lora_B.weight"]
H3_KEYS = ["diffusion_model.blocks.0.attn.q_proj.lora_A.weight", "diffusion_model.blocks.3.mlp.fc1.lora_B.weight"]


def fake_lora(path: Path, keys, meta=None, payload=b"x" * 64) -> Path:
    header = {k: {"dtype": "BF16", "shape": [16, 4096], "data_offsets": [0, 0]} for k in keys}
    if meta:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


class LibraryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patches = [mock.patch.object(lib, "DIR", self.tmp / "lib"), mock.patch.object(lib, "UPLOADS", self.tmp / "up"),
                        mock.patch.object(playground, "ROOT", self.tmp / "pg")]
        for p in self.patches:
            p.start()
        (self.tmp / "up").mkdir()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def upload(self, name, keys, payload=b"x" * 64):
        fake_lora(self.tmp / "up" / name, keys, payload=payload)
        return name

    def add(self, keys=LTX_KEYS, name="Dream", payload=b"x" * 64, **kw):
        return lib.add({"upload_id": self.upload(name.replace(" ", "_") + str(len(payload)), keys, payload), "name": name,
                        "keywords": "dr34m", "info": "long notes", "strength_min": 0.6, "strength_max": 0.9, **kw})

    def test_detect_family(self):
        self.assertEqual(lib.detect_family(LTX_KEYS), "ltx2")
        self.assertEqual(lib.detect_family(H3_KEYS), "minimax_h3")
        self.assertIsNone(lib.detect_family(["unet.down.0.weight"]))
        self.assertIsNone(lib.detect_family(LTX_KEYS + H3_KEYS))
        self.assertEqual(lib.detect_family(["lora_unet_blocks_2_attn_q_proj.lora_down.weight"]), "minimax_h3")

    def test_detect_image_families(self):
        sdxl = ["lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.lora_down.weight", "lora_te1_text_model_encoder_layers_0_mlp_fc1.alpha"]
        qwen = ["transformer_blocks.0.attn.add_k_proj.lora_down.weight", "transformer_blocks.9.txt_mlp.net.2.alpha"]
        self.assertEqual(lib.detect_family(sdxl), "sdxl")
        self.assertEqual(lib.detect_family(["unet.down_blocks.1.attentions.0.proj_in.lora_A.weight"]), "sdxl")
        self.assertEqual(lib.detect_family(qwen), "qwen_edit")
        self.assertEqual(lib.detect_family(["diffusion_model.transformer_blocks.3.img_mod.1.lora_B.weight"]), "qwen_edit")
        self.assertIsNone(lib.detect_family(sdxl + qwen))        # two families: the user picks
        self.assertNotEqual(lib.detect_family(LTX_KEYS), "qwen_edit")  # LTX's transformer_blocks are no Qwen blocks
        self.assertIn("h3_image", lib.FAMILIES)                   # H3-Image: picked by hand (its keys look like H3 video)

    def test_detect_krea2(self):
        krea = ["diffusion_model.blocks.0.attn.wq.lora_A.weight", "diffusion_model.blocks.0.attn.gate.alpha",
                "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.lora_B.weight"]
        self.assertEqual(lib.detect_family(krea), "krea2")
        self.assertEqual(lib.detect_family(["lora_unet_blocks_3_attn_wk.lora_down.weight"]), "krea2")
        self.assertNotEqual(lib.detect_family(H3_KEYS), "krea2")   # H3: attn.q_proj, not attn.wq
        self.assertNotEqual(lib.detect_family(LTX_KEYS), "krea2")  # LTX: to_gate_logits is no attn.gate
        self.assertIsNone(lib.detect_family(krea + H3_KEYS))

    def test_model_families(self):
        self.assertEqual(lib.family_of_model("ltx23"), "ltx2")
        self.assertEqual(lib.family_of_model("h3_gguf_q6"), "minimax_h3")

    def test_add_moves_file_and_indexes(self):
        e = self.add()
        self.assertEqual((e["id"], e["family"], e["family_source"]), ("dream", "ltx2", "header"))
        self.assertTrue(lib.file_path(e).is_file())
        self.assertEqual(lib.get("dream")["keywords"], "dr34m")
        self.assertFalse(any((self.tmp / "up").iterdir()))

    def test_duplicate_refused(self):
        self.add()
        with self.assertRaisesRegex(ValueError, "already in the library"):
            self.add(name="Again")

    def test_unknown_family_needs_pick(self):
        with self.assertRaisesRegex(ValueError, "pick the family"):
            self.add(keys=["x.weight"])
        e = self.add(keys=["x.weight"], name="Odd", payload=b"y" * 10, family="minimax_h3")
        self.assertEqual(e["family_source"], "user")

    def test_mismatch_needs_force(self):
        with self.assertRaisesRegex(ValueError, "use anyway"):
            self.add(family="minimax_h3")
        self.assertEqual(self.add(family="minimax_h3", force=True, payload=b"z" * 5)["family"], "minimax_h3")

    def test_strength_range_checked(self):
        with self.assertRaisesRegex(ValueError, "Strength range"):
            self.add(strength_min=1.2, strength_max=0.5)

    def test_update_and_picture(self):
        self.add()
        png = "data:image/png;base64,iVBORw0KGgo="
        e = lib.update({"id": "dream", "name": "Dream 2", "picture": png})
        self.assertEqual(e["name"], "Dream 2")
        self.assertIsNotNone(lib.picture("dream"))
        lib.update({"id": "dream", "picture": ""})
        self.assertIsNone(lib.picture("dream"))

    def test_slots_and_delete(self):
        self.add()
        pg = playground.new({})
        self.assertEqual(pg["model"], "ltx23")
        state = playground.save({"id": pg["id"], "loras": [{"id": "dream", "strength": 0.75}, {"id": "", "strength": 1}]})
        self.assertEqual(state["loras"][0]["id"], "dream")
        snap = playground.snapshot(pg["id"])
        self.assertEqual(snap["loras"][0]["name"], "Dream")
        self.assertTrue(snap["loras"][0]["path"].endswith("dream.safetensors"))
        self.assertEqual(lib.delete({"id": "dream"})["used_by"][0]["id"], pg["id"])
        self.assertTrue(lib.get("dream"))  # not without confirm
        lib.delete({"id": "dream", "confirm": True})
        self.assertIsNone(lib.get("dream"))
        self.assertEqual(playground.load(pg["id"])["loras"][0]["id"], "")

    def test_wrong_family_refused_and_cleared(self):
        self.add()
        pg = playground.new({})
        with self.assertRaisesRegex(ValueError, "this model takes"):
            playground.save({"id": pg["id"], "model": "h3_minimax", "loras": [{"id": "dream", "strength": 1}, {"id": "", "strength": 1}]})
        playground.save({"id": pg["id"], "loras": [{"id": "dream", "strength": 1}, {"id": "", "strength": 1}]})
        self.assertEqual(playground.save({"id": pg["id"], "model": "h3_minimax"})["loras"][0]["id"], "")

    def test_old_free_paths_dropped(self):
        pg = playground.new({})
        path = playground.folder(pg["id"]) / "playground.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["loras"] = [{"path": "C:/x/old.safetensors", "strength": 0.8}, {"path": "", "strength": 1.0}]
        path.write_text(json.dumps(state), encoding="utf-8")
        loras = playground.load(pg["id"])["loras"]
        self.assertEqual([l["id"] for l in loras], ["", ""])
        self.assertEqual(playground.snapshot(pg["id"])["loras"], [])


if __name__ == "__main__":
    unittest.main()
