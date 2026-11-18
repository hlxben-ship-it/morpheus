"""int8 backend choice of the warm worker (engine/wsl/int8_choice.py). Run: python -m unittest engine.test_int8_choice"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "wsl"))
from int8_choice import int8_kernels  # noqa: E402


class Int8Kernels(unittest.TestCase):
    def test_default_is_triton(self):
        self.assertEqual(int8_kernels({}), "triton")
        self.assertEqual(int8_kernels({"MORPHEUS_INT8_KERNELS": ""}), "triton")

    def test_override_passes(self):
        for v in ("auto", "kitchen", "disabled", "wgpkernels", "triton"):
            self.assertEqual(int8_kernels({"MORPHEUS_INT8_KERNELS": v}), v)
        self.assertEqual(int8_kernels({"MORPHEUS_INT8_KERNELS": " Kitchen "}), "kitchen")

    def test_unknown_value_is_blocked(self):
        self.assertEqual(int8_kernels({"MORPHEUS_INT8_KERNELS": "cuda"}), "triton")


if __name__ == "__main__":
    unittest.main()
