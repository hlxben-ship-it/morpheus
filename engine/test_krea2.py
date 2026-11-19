"""Krea 2 pipeline pieces that need no GPU and no weights (Glia seed muse_krea2): checkpoint names, the text prefix cut,
the krea2edit reference fit and positions, LoRA mapping, and the sampling additions (FluxShift, beta, er_sde)."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "wsl"))
from image import krea2 as k2  # noqa: E402
from image import sampling  # noqa: E402

CFG = {"hidden": 256, "in_channels": 64, "heads": 2, "kv_heads": 1, "inter": 64, "layers": 2, "text_dim": 256, "text_heads": 2,
       "text_kv_heads": 2, "text_inter": 64, "text_layers": 12, "n_layerwise": 2, "n_refiner": 2}
COMFY = ["first.weight", "tmlp.0.bias", "tmlp.2.weight", "tproj.1.weight", "txtmlp.0.scale", "txtmlp.1.weight", "txtmlp.3.bias",
         "txtfusion.projector.weight", "txtfusion.layerwise_blocks.1.attn.wq.weight", "txtfusion.refiner_blocks.0.mlp.gate.weight",
         "txtfusion.refiner_blocks.0.prenorm.scale", "txtfusion.layerwise_blocks.0.attn.qknorm.knorm.scale",
         "blocks.1.attn.wo.weight", "blocks.0.attn.gate.weight", "blocks.0.attn.qknorm.qnorm.scale", "blocks.1.mlp.down.weight",
         "blocks.0.prenorm.scale", "blocks.0.postnorm.scale", "blocks.1.mod.lin", "last.linear.weight", "last.norm.scale",
         "last.modulation.lin"]


class Names(unittest.TestCase):
    def test_every_comfy_name_lands_on_a_parameter(self):
        names = {n for n, _ in k2.Krea2DiT(CFG).named_parameters()}
        for prefix in ("", "model.diffusion_model.", "diffusion_model."):
            for k in COMFY:
                self.assertIn(k2.rename(prefix + k), names, prefix + k)

    def test_foreign_names_are_dropped(self):
        self.assertIsNone(k2.rename("text_encoders.qwen3vl_4b.transformer.model.layers.0.mlp.weight"))
        self.assertIsNone(k2.rename("vae.decoder.conv_in.weight"))

    def test_assign_fills_every_parameter_and_refuses_gaps(self):
        sd = {k: torch.full_like(v, 0.5) for k, v in k2.Krea2DiT(CFG).state_dict().items()}
        with torch.device("meta"):
            m = k2.Krea2DiT(CFG)
        k2._assign(m, sd)
        self.assertTrue(all(p.device.type == "cpu" and float(p.flatten()[0]) == 0.5 for p in m.parameters()))
        sd.pop("img_in.weight")
        with torch.device("meta"):
            m = k2.Krea2DiT(CFG)
        with self.assertRaises(RuntimeError):
            k2._assign(m, sd)

    def test_config_from_shapes(self):
        sd = {k: p for k, p in k2.Krea2DiT(CFG).state_dict().items()}
        self.assertEqual(k2.dit_config(sd), CFG)


class Prefix(unittest.TestCase):
    def test_cuts_system_and_user_opening(self):
        ids = [151644, 8948, 198, 5, 6, 151645, 198, 151644, 872, 198, 42, 43, 151645, 198, 151644, 77091, 198]
        self.assertEqual(k2.prefix_len(ids), 10)   # the prompt (42) is the first kept token; the 3rd <|im_start|> is not a cut

    def test_without_user_turn_keeps_from_second_start(self):
        self.assertEqual(k2.prefix_len([151644, 1, 151644, 5, 6, 7]), 2)


class References(unittest.TestCase):
    def test_same_aspect_fills_the_target(self):
        box, size = k2.fit_pixels(2048, 2048, 1024, 1024)
        self.assertEqual(size, (1024, 1024)); self.assertEqual(box, (0, 0, 2048, 2048))

    def test_near_aspect_crops_to_fill(self):  # within 8 %: centre crop, exact target size (no edge duplication)
        box, size = k2.fit_pixels(1000, 1040, 1024, 1024)
        self.assertEqual(size, (1024, 1024)); self.assertEqual(box[2], box[3])

    def test_other_aspect_fits_inside_on_the_16_grid(self):
        box, (nw, nh) = k2.fit_pixels(800, 1600, 1024, 1024)  # portrait into square
        self.assertEqual(nh, 1024); self.assertEqual(nw % 16, 0); self.assertLessEqual(nw, 1024); self.assertLess(nw, 600)

    def test_ref_positions_centred_on_target(self):
        ids = k2.ref_ids(2, 4, 2, 4, 6, "cpu")
        self.assertTrue(torch.all(ids[:, 0] == 2))
        self.assertEqual(ids[:, 2].min().item(), 2.0); self.assertEqual(ids[:, 2].max().item(), 3.0)
        t = k2.target_ids(4, 6, "cpu")
        self.assertTrue(torch.all(t[:, 0] == 0)); self.assertEqual(t.shape, (24, 3))

    def test_shrunk_reference_spans_the_same_area(self):
        full = k2.ref_ids(1, 8, 8, 8, 8, "cpu")
        half = k2.ref_ids(1, 4, 4, 8, 8, "cpu", stride=2.0)
        self.assertEqual(half.shape, (16, 3))
        self.assertEqual(half[:, 1].min().item(), full[:, 1].min().item())
        self.assertEqual(half[:, 2].max().item(), 6.0)   # 3 * 2: spans 0..6 of the 0..7 grid
        self.assertEqual(k2.ref_ids(1, 4, 4, 8, 8, "cpu", stride=1.0)[:, 1].min().item(), 2.0)  # unscaled: centred, stride 1

    def test_pack_roundtrip(self):
        x = torch.randn(1, 16, 8, 6)
        self.assertTrue(torch.equal(k2.unpack(k2.pack(x), 8, 6), x))


class Rope(unittest.TestCase):
    def test_position_zero_is_identity(self):
        x = torch.randn(1, 3, 2, 128)
        cos, sin = k2.rope_freqs(torch.zeros(3, 3))
        self.assertTrue(torch.allclose(k2.rope_apply(x, (cos, sin)), x, atol=1e-6))

    def test_rotation_keeps_norm_and_moves(self):
        x = torch.randn(1, 2, 1, 128)
        ids = torch.tensor([[0.0, 5.0, 7.0], [1.0, 2.0, 3.0]])
        y = k2.rope_apply(x, k2.rope_freqs(ids))
        self.assertTrue(torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4))
        self.assertFalse(torch.allclose(y, x, atol=1e-3))


class Lora(unittest.TestCase):
    def test_peft_names_with_alpha_map_to_modules(self):
        sd = {"diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.zeros(4, 256), "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.zeros(256, 4),
              "diffusion_model.blocks.0.attn.wq.alpha": torch.tensor(2.0),
              "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.lora_A.weight": torch.zeros(4, 64),
              "diffusion_model.txtfusion.layerwise_blocks.1.mlp.down.lora_B.weight": torch.zeros(256, 4)}
        pairs = k2.lora_pairs(sd)
        mods = dict(k2.Krea2DiT(CFG).named_modules())
        self.assertEqual(set(pairs), {"transformer_blocks.0.attn.to_q", "text_fusion.layerwise_blocks.1.ff.down"})
        self.assertTrue(all(n in mods for n in pairs))
        self.assertEqual(pairs["transformer_blocks.0.attn.to_q"][2], 2.0)
        self.assertIsNone(pairs["text_fusion.layerwise_blocks.1.ff.down"][2])

    def test_unpaired_or_foreign_is_ignored(self):
        self.assertEqual(k2.lora_pairs({"diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.zeros(1)}), {})
        self.assertEqual(k2.lora_pairs({"vae.x.lora_A.weight": torch.zeros(1), "vae.x.lora_B.weight": torch.zeros(1)}), {})


class Int8(unittest.TestCase):
    def test_int8_linear_is_close_and_small(self):
        torch.manual_seed(0)
        seq = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16, bias=False))
        x = torch.randn(4, 64)
        ref = seq(x)
        k2.quantize_linears(seq)
        self.assertIsInstance(seq[0], k2.Int8Linear); self.assertIsInstance(seq[2], k2.Int8Linear)
        self.assertEqual(seq[0].q.dtype, torch.int8)
        self.assertEqual(list(seq.parameters()), [])        # all buffers: the Streamer moves them
        self.assertLess((seq(x) - ref).abs().max().item() / ref.abs().max().item(), 0.02)


class Sampling(unittest.TestCase):
    def test_flux_shift_table(self):
        ms = sampling.FluxShift(1.15)
        import math
        e = math.exp(1.15)
        self.assertAlmostEqual(float(ms.sigma(0.5)), e / (e + 1), places=5)
        self.assertAlmostEqual(float(ms.sigmas[-1]), 1.0, places=6)
        self.assertEqual(len(ms.sigmas), 10000)
        self.assertLess(ms.percent_to_sigma(1e-4), 1.0); self.assertEqual(ms.percent_to_sigma(0), 1.0)

    def test_beta_schedule(self):
        s = sampling.sigmas_for(sampling.FluxShift(1.15), "beta", 12)
        self.assertEqual(len(s), 13); self.assertEqual(float(s[-1]), 0.0); self.assertAlmostEqual(float(s[0]), 1.0, places=4)
        self.assertTrue(torch.all(s[:-1] > s[1:]))

    def test_unknown_names_refused(self):
        with self.assertRaises(ValueError):
            sampling.sigmas_for(sampling.FluxShift(), "exponential", 8)
        with self.assertRaises(ValueError):
            sampling.sample("lms", lambda x, s: x, torch.zeros(1), torch.tensor([1.0, 0.0]), "flow", 1)

    def test_er_sde_lands_on_a_fixed_target(self):
        target = torch.full((1, 4, 4, 4), 0.3)
        s = sampling.sigmas_for(sampling.FluxShift(1.15), "beta", 10)
        x = sampling.noise_scaling("flow", s[0], torch.randn(1, 4, 4, 4), torch.zeros(1, 4, 4, 4), True)
        out = sampling.sample("er_sde", lambda x, sig: target, x, s, "flow", 7, ms=sampling.FluxShift(1.15))
        self.assertTrue(torch.allclose(out, target, atol=1e-5))

    def test_cfg(self):
        c, u = torch.tensor([2.0]), torch.tensor([1.0])
        self.assertEqual(float(k2._cfg(c, u, 1.0)), 2.0); self.assertEqual(float(k2._cfg(c, u, 2.0)), 3.0)


if __name__ == "__main__":
    unittest.main()


class Resharpen(unittest.TestCase):
    """Glia seed muse_resharpen: output size and the schedule cut by strength."""

    def test_size_follows_the_picture_times_scale_in_multiples_of_16(self):
        self.assertEqual(k2.resharpen_size(1024, 1536, 1), (1024, 1536))
        self.assertEqual(k2.resharpen_size(1000, 750, 2), (2000, 1488))
        self.assertEqual(k2.resharpen_size(1001, 999, 1), (992, 992))

    def test_a_too_big_2x_is_capped(self):
        w, h = k2.resharpen_size(1536, 1536, 2)
        self.assertLessEqual(w * h, k2.RESHARPEN_MAX_PX)
        self.assertEqual((w, h), (2048, 2048))
        w, h = k2.resharpen_size(1536, 1024, 2)
        self.assertLessEqual(w * h, k2.RESHARPEN_MAX_PX)
        self.assertAlmostEqual(w / h, 1.5, delta=0.02)

    def test_strength_cuts_the_schedule_low_end_stays_near_the_input(self):
        ms = sampling.FluxShift(k2.SHIFT)
        full = sampling.sigmas_for(ms, "beta", 12, 1.0)
        low = sampling.sigmas_for(ms, "beta", 12, 0.05)
        mid = sampling.sigmas_for(ms, "beta", 12, 0.3)
        self.assertEqual(len(low), 13); self.assertEqual(len(mid), 13)
        self.assertTrue(torch.equal(full, sampling.sigmas_for(ms, "beta", 12, 1.0)))
        self.assertGreater(float(full[0]), 0.99)
        self.assertLess(float(low[0]), 0.15)  # 0.05: start mostly = the picture
        self.assertLess(float(low[0]), float(mid[0])); self.assertLess(float(mid[0]), float(full[0]))
        x = sampling.noise_scaling("flow", low[0], torch.zeros(1), torch.ones(1), True)
        self.assertGreater(float(x), 0.85)
