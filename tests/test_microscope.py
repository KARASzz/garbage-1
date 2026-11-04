import unittest
from pathlib import Path
from trash_talk.model_runtime import ModelRuntime


class TestMicroscope(unittest.TestCase):
    def test_inspect_sequence_structure_on_base_model(self):
        root = Path(".")
        model_dir = "models/garbage-10m" if (root / "models/garbage-10m/current.json").exists() else "models/garbage-1"
        runtime = ModelRuntime(root, model_dir=model_dir, backend="torch-cpu")
        self.assertTrue(runtime.ready)

        res = runtime.generate(prefix="我觉得", max_new_tokens=5, temperature=0.85)
        self.assertIn("microscope", res)
        microscope = res["microscope"]
        self.assertIsNotNone(microscope)
        self.assertIn("chars", microscope)
        self.assertIn("attentions", microscope)
        self.assertIn("top_candidates", microscope)

        chars = microscope["chars"]
        attentions = microscope["attentions"]
        top_candidates = microscope["top_candidates"]

        self.assertEqual(len(chars), len(attentions))
        self.assertEqual(len(chars), len(top_candidates))

        # Check causal attention structure
        for i, row in enumerate(attentions):
            self.assertEqual(len(row), i + 1, f"Row {i} should have {i+1} weights")
            self.assertAlmostEqual(sum(row), 1.0, delta=0.05, msg=f"Row {i} should sum to ~1.0")

        # Check top candidates
        for cands in top_candidates:
            self.assertIsInstance(cands, list)
            if cands:
                self.assertLessEqual(len(cands), 3)
                for cand in cands:
                    self.assertIn("char", cand)
                    self.assertIn("prob", cand)
                    self.assertGreaterEqual(cand["prob"], 0.0)
                    self.assertLessEqual(cand["prob"], 1.0)

    def test_inspect_sequence_structure_on_mlx(self):
        try:
            import mlx.core as mx
            if not mx.metal.is_available():
                return
        except ImportError:
            return

        root = Path(".")
        model_dir = "models/garbage-10m" if (root / "models/garbage-10m/current.json").exists() else "models/garbage-1"
        runtime = ModelRuntime(root, model_dir=model_dir, backend="mlx")
        self.assertTrue(runtime.ready)

        res = runtime.generate(prefix="我觉得", max_new_tokens=5, temperature=0.85)
        self.assertIn("microscope", res)
        microscope = res["microscope"]
        self.assertIsNotNone(microscope)
        self.assertEqual(len(microscope["chars"]), len(microscope["attentions"]))
        for i, row in enumerate(microscope["attentions"]):
            self.assertEqual(len(row), i + 1)
            self.assertAlmostEqual(sum(row), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
