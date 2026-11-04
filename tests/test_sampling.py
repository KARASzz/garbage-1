import unittest
import numpy as np
from trash_talk.sampling import sample_next_token


class TestSampling(unittest.TestCase):
    def test_temperature_zero_is_argmax(self):
        logits = np.array([-1.0, 5.0, 2.0, 0.0], dtype=np.float32)
        idx = sample_next_token(logits, temperature=0.0)
        self.assertEqual(idx, 1)

    def test_temperature_sampling_valid_range(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        rng = np.random.default_rng(42)
        for _ in range(50):
            idx = sample_next_token(logits, temperature=0.8, top_p=0.95, rng=rng)
            self.assertIn(idx, [0, 1, 2, 3])

    def test_repetition_penalty_depresses_seen_token(self):
        # Token 0 has slightly higher score, but if heavily penalized, token 1 should win
        logits = np.array([2.0, 1.9, 0.0], dtype=np.float32)
        idx = sample_next_token(logits, temperature=0.01, repetition_penalty=2.0, history=[0])
        self.assertEqual(idx, 1)

    def test_top_p_filters_low_probability_tokens(self):
        # Token 0 has massive probability, top_p=0.5 should only select token 0
        logits = np.array([100.0, -100.0, -100.0], dtype=np.float32)
        idx = sample_next_token(logits, temperature=1.0, top_p=0.5)
        self.assertEqual(idx, 0)


if __name__ == "__main__":
    unittest.main()
