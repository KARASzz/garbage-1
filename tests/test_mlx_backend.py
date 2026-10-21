"""Numerical and behavioral checks for the native MLX model backend."""

from __future__ import annotations

import unittest

import mlx.core as mx
import numpy as np
import torch

from trash_talk.garbage_model import build_model, solve_config
from trash_talk.mlx_backend import MLXBackend


class MLXBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(211)
        cls.config = solve_config(816)
        cls.reference = build_model(cls.config).eval()
        cls.weights = {
            name: parameter.detach().cpu().numpy().copy()
            for name, parameter in cls.reference.named_parameters()
        }
        cls.backend = MLXBackend(cls.config, cls.weights)

    def test_unique_parameter_budget_dtype_and_tied_embedding(self) -> None:
        weights = self.backend.export_weights()
        count = sum(array.size for array in weights.values())
        self.assertEqual(count, 100_000)
        self.assertTrue(all(array.dtype == np.float32 for array in weights.values()))
        self.assertTrue(all(value.dtype == mx.float32 for value in self.backend.parameters.values()))
        self.assertNotIn("lm_head.weight", self.backend.parameters)
        self.assertNotIn("lm_head.weight", weights)
        self.assertEqual(
            weights["token_embedding.weight"].shape,
            self.weights["token_embedding.weight"].shape,
        )

    def test_logits_match_torch_and_causality_holds(self) -> None:
        ids = [[1, 4, 5, 6, 7]]
        with torch.inference_mode():
            expected = self.reference(torch.tensor(ids, dtype=torch.long)).cpu().numpy()
        actual = np.asarray(self.backend.logits(ids))
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-4)
        np.testing.assert_array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_array_equal(
            np.asarray(self.backend.logits(mx.array(ids))), actual
        )

        prefix = [[1, 4, 5, 6]]
        left = np.asarray(self.backend.logits([prefix[0] + [3, 4]]))[:, :4]
        right = np.asarray(self.backend.logits([prefix[0] + [7, 3]]))[:, :4]
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)

    def test_greedy_generation_is_deterministic_and_preserves_prompt(self) -> None:
        prompt = [1, 4, 5]
        first = self.backend.generate(prompt, max_new_tokens=5, eos_id=2)
        second = self.backend.generate(prompt, max_new_tokens=5, eos_id=2)
        self.assertEqual(first, second)
        self.assertEqual(first[: len(prompt)], prompt)
        self.assertGreaterEqual(len(first), len(prompt))

        forced_eos_weights = {name: value.copy() for name, value in self.weights.items()}
        forced_eos_weights["lm_bias"] = np.zeros(self.config.vocab_size, dtype=np.float32)
        forced_eos_weights["lm_bias"][2] = 100.0
        forced_eos = MLXBackend(self.config, forced_eos_weights)
        self.assertEqual(forced_eos.generate(prompt, max_new_tokens=8, eos_id=2), prompt + [2])

    def test_train_step_updates_real_fp32_parameters(self) -> None:
        before = self.backend.weight_digest()
        loss = self.backend.train_step(
            x=[[1, 4, 5, 6], [1, 6, 5, 4]],
            y=[[4, 5, 6, 2], [6, 5, 4, 2]],
            learning_rate=0.005,
            pad_id=0,
        )
        after = self.backend.weight_digest()
        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(before, after)
        self.assertTrue(all(value.dtype == mx.float32 for value in self.backend.parameters.values()))

    def test_padding_only_batch_has_zero_loss_and_does_not_update(self) -> None:
        before = self.backend.weight_digest()
        loss = self.backend.train_step(
            x=[[1, 4, 5], [1, 6, 5]],
            y=[[0, 0, 0], [0, 0, 0]],
            pad_id=0,
        )
        self.assertEqual(loss, 0.0)
        self.assertEqual(self.backend.weight_digest(), before)

    def test_tied_alias_must_match_and_unknown_architecture_is_rejected(self) -> None:
        aliased = dict(self.weights)
        aliased["lm_head.weight"] = aliased["token_embedding.weight"].copy()
        MLXBackend(self.config, aliased)

        aliased["lm_head.weight"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "完全相同"):
            MLXBackend(self.config, aliased)

        class UnsupportedConfig:
            architecture = "moe_transformer_v1"
            ffn_type = "dense"

        with self.assertRaisesRegex(ValueError, "MoE 架构尚未实现"):
            MLXBackend(UnsupportedConfig(), self.weights)


if __name__ == "__main__":
    unittest.main()
