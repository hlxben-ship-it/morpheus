"""Tests of the Sol merge (no WSL, no GPU): python -m unittest engine.test_wsl_runner (Morpheus root)."""
import importlib.util
import unittest
from pathlib import Path

from engine import jobs, wsl_runner as wr

# WanGP's own parser, loaded as a lone file: its package __init__ pulls in diffusers
_spec = importlib.util.spec_from_file_location("generation_overrides", Path(__file__).parent / "wangp" / "shared" / "utils" / "generation_overrides.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
parse_overrides = _mod.parse_overrides


class WithSol(unittest.TestCase):
    def test_keeps_the_bf16_encoder_override(self):
        s = wr.with_sol({"model_type": "x", "overrides": "text_encoder_quantization='bf16'"})
        self.assertEqual(parse_overrides(s["overrides"]), {"text_encoder_quantization": "bf16", "attention_mode": "sol"})
        self.assertEqual(s["attention_sparsity"], 1.3)

    def test_without_own_override_only_sol(self):
        self.assertEqual(wr.with_sol({"model_type": "x"})["overrides"], "attention_mode='sol'")
        self.assertEqual(wr.with_sol({"overrides": ""})["overrides"], "attention_mode='sol'")

    def test_does_not_touch_the_input(self):
        s = {"overrides": "text_encoder_quantization='bf16'"}
        wr.with_sol(s)
        self.assertEqual(s, {"overrides": "text_encoder_quantization='bf16'"})

    def test_studio_bf16_job_reaches_wangp_with_both(self):
        """The real path: the job jobs.py builds for the bf16 encoder still says bf16 after the Sol merge."""
        bf16 = {"overrides": f"text_encoder_quantization={jobs.H3_ENCODERS['4b_bf16'][1]!r}"}
        self.assertIn("text_encoder_quantization", parse_overrides(wr.with_sol(bf16)["overrides"]))


if __name__ == "__main__":
    unittest.main()
