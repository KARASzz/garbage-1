"""Behavioral checks for the exact-budget character Transformer."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from trash_talk.garbage_model import (
    CharacterTokenizer,
    ModelConfig,
    SPECIAL_TOKENS,
    build_model,
    greedy_tokens,
    solve_config,
)


class CharacterTokenizerTests(unittest.TestCase):
    def test_training_only_fit_unknown_theme_tokens_and_round_trip(self) -> None:
        tokenizer = CharacterTokenizer.fit(["é好", "日常"])

        self.assertEqual(tokenizer.vocab_size, len(SPECIAL_TOKENS) + 4)
        self.assertEqual(tokenizer.pad_id, 0)
        self.assertEqual(tokenizer.bos_id, 1)
        self.assertEqual(tokenizer.eos_id, 2)
        self.assertEqual(tokenizer.unk_id, 3)

        encoded = tokenizer.encode("e\u0301好你", theme="logic", bos=True, eos=True)
        self.assertEqual(encoded[0], tokenizer.bos_id)
        self.assertNotEqual(encoded[1], tokenizer.unk_id)
        self.assertIn(tokenizer.unk_id, encoded)
        self.assertEqual(encoded[-1], tokenizer.eos_id)
        self.assertEqual(tokenizer.decode(encoded), "é好")

        restored = CharacterTokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.to_dict(), tokenizer.to_dict())
        self.assertEqual(restored.encode("好", theme="game"), tokenizer.encode("好", theme="game"))

    def test_theme_and_serialization_validation(self) -> None:
        tokenizer = CharacterTokenizer.fit(["甲"])
        with self.assertRaisesRegex(ValueError, "未知主题"):
            tokenizer.encode("甲", theme="unknown")

        invalid = deepcopy(tokenizer.to_dict())
        invalid["special_tokens"][0] = "<OTHER>"
        with self.assertRaisesRegex(ValueError, "特殊 token"):
            CharacterTokenizer.from_dict(invalid)


class ModelConfigurationTests(unittest.TestCase):
    def test_solver_builds_exact_fp32_models_for_multiple_vocabularies(self) -> None:
        for vocab_size in (8, 64, 816, 1024, 2048):
            with self.subTest(vocab_size=vocab_size):
                config = solve_config(vocab_size)
                self.assertEqual(config.architecture, "dense_transformer_v1")
                self.assertEqual(config.ffn_type, "dense")
                self.assertEqual(config.vocab_size, vocab_size)
                self.assertGreaterEqual(config.context, 96)
                self.assertLessEqual(config.context, 512)

                model = build_model(config)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 100_000)
                self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
                self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_config_serialization_and_unknown_architectures_are_explicit(self) -> None:
        config = solve_config(816)
        restored = ModelConfig.from_dict(config.to_dict())
        self.assertEqual(restored.to_dict(), config.to_dict())

        for field, value, error in (
            ("architecture", "moe_transformer_v1", "MoE"),
            ("architecture", "future_unknown", "未知模型架构"),
            ("ffn_type", "moe", "MoE FFN"),
            ("ffn_type", "future_unknown", "未知 FFN"),
        ):
            data = config.to_dict()
            data[field] = value
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, error):
                    ModelConfig.from_dict(data)

    def test_solver_reports_vocabularies_that_do_not_fit_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "无法在"):
            solve_config(4096)


class TransformerBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.config = solve_config(8)
        self.model = build_model(self.config)
        self.model.eval()

    def test_forward_is_causal_and_has_stable_state_dict_names(self) -> None:
        prefix = torch.tensor([[1, 4, 5, 6]], dtype=torch.long)
        first = torch.cat((prefix, torch.tensor([[7, 4]], dtype=torch.long)), dim=1)
        second = torch.cat((prefix, torch.tensor([[6, 5]], dtype=torch.long)), dim=1)
        with torch.inference_mode():
            logits_first = self.model(first)
            logits_second = self.model(second)

        self.assertEqual(logits_first.shape, (1, 6, self.config.vocab_size))
        torch.testing.assert_close(logits_first[:, :4], logits_second[:, :4])
        keys = set(self.model.state_dict())
        self.assertIn("token_embedding.weight", keys)
        self.assertIn("position_embedding.weight", keys)
        self.assertIn("blocks.0.qkv.weight", keys)
        self.assertIn("blocks.1.ffn.up.weight", keys)
        self.assertIn("final_ln.weight", keys)
        self.assertIn("lm_head.weight", keys)
        self.assertIn("lm_bias", keys)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            fp32_logits = self.model(prefix)
        self.assertEqual(fp32_logits.dtype, torch.float32)

    def test_full_context_uses_every_unique_parameter(self) -> None:
        ids = torch.arange(self.config.context, dtype=torch.long).remainder(
            self.config.vocab_size
        )[None, :]
        self.model.zero_grad(set_to_none=True)
        self.model(ids).square().mean().backward()

        unused = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is None
        ]
        self.assertEqual(unused, [])
        self.assertTrue(torch.all(self.model.position_embedding.weight.grad.abs().sum(dim=1) > 0))

    def test_greedy_argmax_is_deterministic_and_stops_on_eos(self) -> None:
        prompt = [1, 4, 5]
        self.model.lm_bias.data[2] = 100.0
        first = greedy_tokens(self.model, prompt, max_new_tokens=8, eos_id=2)
        second = greedy_tokens(self.model, prompt, max_new_tokens=8, eos_id=2)
        self.assertEqual(first, second)
        self.assertEqual(first, prompt + [2])

    def test_greedy_context_is_truncated_and_new_token_limit_is_respected(self) -> None:
        class RecordingModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))
                self.config = SimpleNamespace(context=4, vocab_size=8)
                self.lengths: list[int] = []

            def forward(self, ids: torch.Tensor) -> torch.Tensor:
                self.lengths.append(ids.shape[1])
                logits = torch.zeros((1, ids.shape[1], 8), device=ids.device)
                logits[:, :, 7] = 1.0
                return logits

        model = RecordingModel()
        output = greedy_tokens(model, [1, 3, 4, 5, 6], max_new_tokens=3, eos_id=2)
        self.assertEqual(output, [1, 3, 4, 5, 6, 7, 7, 7])
        self.assertEqual(model.lengths, [4, 4, 4])


if __name__ == "__main__":
    unittest.main()
