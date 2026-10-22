"""Numerical and persistence checks for online preference training.

All models, checkpoints, feedback rows, and adapter weights in this module are
created under temporary directories.  The repository's model assets are never
loaded or changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from trash_talk.feedback import FeedbackStore
from trash_talk.garbage_model import (
    CharacterTokenizer,
    ModelConfig,
    _parameter_count_for,
    build_model,
)
from trash_talk.lora import initial_adapter, make_backend
from trash_talk.model_runtime import ModelRuntime
from trash_talk.preference_training import (
    log_probabilities,
    make_batch,
    objective,
    preference_step,
)
from trash_talk.torch_backend import TorchBackend


try:
    MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
    if MLX_AVAILABLE:
        import mlx.core as _mlx_core

        MLX_AVAILABLE = _mlx_core.metal.is_available()
except Exception:
    MLX_AVAILABLE = False


PROMPT = "你说："
ORIGINAL = "逻辑停在门口。"
CORRECTION = "我把答案带回来了。"
REPLAY = "另一条正常训练锚点。"
LONG_CORRECTION = "这段更长的改写用来确认最后一个字和句尾标记都保留在训练窗口里。"


def _small_model(tokenizer: CharacterTokenizer, *, context: int = 8, seed: int = 3041):
    """Build a small, ordinary-bias model that both backends can share."""
    width, heads, ffn_dims = 48, 6, [64, 64]
    target = _parameter_count_for(tokenizer.vocab_size, context, width, ffn_dims)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context=context,
        d_model=width,
        heads=heads,
        ffn_dims=ffn_dims,
        target_parameters=target,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_model(config).eval()
    weights = {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in model.named_parameters()
    }
    return config, weights


def _weights_digest(weights: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for _, value in sorted(weights.items()):
        digest.update(np.asarray(value).astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_small_runtime_root(root: Path) -> Path:
    """Create the minimal valid app tree around a small random checkpoint."""
    data_dir = root / "data" / "garbage-1"
    prefixes = [{"theme": "daily", "prefix": PROMPT}]
    replay = [{"theme": "daily", "text": REPLAY}]
    tokenizer = CharacterTokenizer.fit([PROMPT, ORIGINAL, CORRECTION, REPLAY, LONG_CORRECTION])
    config, weights = _small_model(tokenizer, context=8, seed=9841)

    _write_json(
        data_dir / "prefixes.json",
        {"format_version": 1, "count": len(prefixes), "examples": prefixes},
    )
    _write_json(
        data_dir / "seed_replay.json",
        {"format_version": 1, "count": len(replay), "examples": replay},
    )

    model_dir = root / "models" / "garbage-1"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "bootstrap.npz"
    with weights_path.open("wb") as stream:
        np.savez(stream, **weights)
    _write_json(
        checkpoint_dir / "bootstrap.json",
        {
            "format": "garbage-1-checkpoint-v1",
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "weights": weights_path.name,
            "weight_sha256": _weights_digest(weights),
            "parameter_count": config.target_parameters,
            "step": 0,
            "last_job_id": 0,
            "last_preference_id": 0,
            "run_id": "preference-test-run",
        },
    )
    _write_json(model_dir / "current.json", {"checkpoint": "checkpoints/bootstrap.json"})
    return root


class PreferenceMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = CharacterTokenizer.fit(
            [PROMPT, ORIGINAL, CORRECTION, REPLAY, LONG_CORRECTION, "abcdefg", "xy"]
        )
        cls.config, cls.weights = _small_model(cls.tokenizer)

    def _batch(self, chosen: str = "abcdefg", rejected: str = "xy", *, context: int = 8):
        prompt_ids = self.tokenizer.encode(PROMPT, theme="daily")
        return make_batch(
            self.tokenizer,
            context,
            prompt_ids,
            chosen,
            rejected,
        )

    def _backend(self):
        return TorchBackend(self.config, self.weights, device="cpu")

    def test_equal_policy_and_reference_with_no_sft_has_log_two_loss(self) -> None:
        logps = [-3.25, -7.5, -1.0]
        counts = [4.0, 9.0, 2.0]

        loss = objective(
            logps,
            counts,
            reference=logps[:2],
            beta=0.2,
            chosen_weight=0.0,
            anchor_weight=0.0,
        )

        self.assertAlmostEqual(loss, float(np.log(2.0)), places=12)

    def test_prompt_and_padding_targets_do_not_contribute_to_log_probability(self) -> None:
        backend = self._backend()
        batch = self._batch(chosen="abcdefg", rejected="xy")
        expected, expected_counts = log_probabilities(backend, batch)
        changed = {
            "x": [list(row) for row in batch["x"]],
            "y": [list(row) for row in batch["y"]],
            "mask": [list(row) for row in batch["mask"]],
            "groups": list(batch["groups"]),
        }

        # Prompt targets and right-padding targets are masked.  Give them
        # different valid token IDs and confirm the measured logps are stable.
        for row_index, mask_row in enumerate(changed["mask"]):
            for position, active in enumerate(mask_row):
                if not active:
                    target = changed["y"][row_index][position]
                    changed["y"][row_index][position] = (target + 1) % self.tokenizer.vocab_size
            # Every PAD input is trailing padding; causal attention means it
            # cannot affect an earlier, unmasked completion target.
            changed["x"][row_index] = [
                self.tokenizer.encode("a", bos=False)[0]
                if token == self.tokenizer.pad_id
                else token
                for token in changed["x"][row_index]
            ]

        actual, actual_counts = log_probabilities(backend, changed)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(actual_counts, expected_counts)
        self.assertEqual(expected_counts[0], len(self.tokenizer.encode("abcdefg", bos=False, eos=True)))
        self.assertEqual(expected_counts[1], len(self.tokenizer.encode("xy", bos=False, eos=True)))

    def test_long_completion_keeps_every_tail_token_and_eos_target(self) -> None:
        batch = self._batch(chosen=LONG_CORRECTION, rejected="xy", context=5)
        completion_ids = self.tokenizer.encode(LONG_CORRECTION, bos=False, eos=True)
        selected = [
            target
            for targets, masks, group in zip(batch["y"], batch["mask"], batch["groups"], strict=True)
            if group == 0
            for target, active in zip(targets, masks, strict=True)
            if active
        ]

        self.assertEqual(selected, completion_ids)
        self.assertEqual(selected[-1], self.tokenizer.eos_id)

    def test_pure_dpo_step_increases_chosen_rejected_margin(self) -> None:
        backend = self._backend()
        batch = self._batch()
        reference, _ = log_probabilities(backend, batch)
        before_digest = backend.weight_digest()

        result = preference_step(
            backend,
            batch,
            reference,
            learning_rate=0.1,
            beta=0.3,
            chosen_weight=0.0,
            anchor_weight=0.0,
        )

        self.assertNotEqual(backend.weight_digest(), before_digest)
        self.assertGreater(result["preference_margin_after"], result["preference_margin_before"])

    def test_lora_dpo_updates_only_adapter_and_leaves_reference_and_base_frozen(self) -> None:
        adapter_arrays = initial_adapter(self.config, rank=2, seed=421)
        policy = make_backend(
            self.config,
            self.weights,
            adapter_arrays,
            backend="torch-cpu",
            rank=2,
            alpha=4,
        )
        reference = TorchBackend(self.config, self.weights, device="cpu")
        batch = self._batch()
        reference_logps, _ = log_probabilities(reference, batch)
        base_digest = policy.base_digest()
        reference_digest = reference.weight_digest()
        adapter_digest = policy.weight_digest()
        before = policy.export_weights()

        preference_step(
            policy,
            batch,
            reference_logps,
            learning_rate=0.1,
            beta=0.3,
            chosen_weight=0.0,
            anchor_weight=0.0,
        )

        after = policy.export_weights()
        self.assertNotEqual(policy.weight_digest(), adapter_digest)
        self.assertEqual(policy.base_digest(), base_digest)
        self.assertEqual(reference.weight_digest(), reference_digest)
        self.assertEqual(set(after), set(before))
        self.assertTrue(all(name.endswith((".A", ".B")) for name in after))
        self.assertTrue(
            all(np.array_equal(before[name], after[name]) for name in before if name.endswith(".A"))
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in policy.base.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in policy.base.model.parameters()))

    @unittest.skipUnless(MLX_AVAILABLE, "MLX Metal is unavailable on this host")
    def test_mlx_and_torch_cpu_preference_logps_are_close(self) -> None:
        from trash_talk.mlx_backend import MLXBackend

        batch = self._batch(chosen=LONG_CORRECTION, rejected="xy", context=5)
        torch_backend = TorchBackend(self.config, self.weights, device="cpu")
        mlx_backend = MLXBackend(self.config, self.weights)

        torch_logps, torch_counts = log_probabilities(torch_backend, batch)
        mlx_logps, mlx_counts = log_probabilities(mlx_backend, batch)

        np.testing.assert_array_equal(mlx_counts, torch_counts)
        np.testing.assert_allclose(mlx_logps, torch_logps, rtol=1e-4, atol=1e-4)


class PreferenceRuntimePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = _make_small_runtime_root(Path(self.temp.name))
        self.runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertTrue(self.runtime.ready, self.runtime.error)
        self.store = FeedbackStore(self.root / "data" / "feedback.sqlite3")

    def test_learn_preference_persists_immutable_checkpoint_and_duplicate_id_is_idempotent(self) -> None:
        context = self.runtime.preference_context({"theme": "daily", "prompt": PROMPT})
        self.assertIsNotNone(context)
        sample_id = self.store.add_sample(
            PROMPT + ORIGINAL,
            "garbage-1",
            "daily",
            self.runtime.metadata["step"],
            self.runtime.metadata["run_id"],
            context=context,
        )
        correction = self.store.correct(sample_id, PROMPT + CORRECTION)
        job = self.store.claim_preference()
        self.assertEqual(job["id"], correction["preference_id"])
        self.assertEqual(job["chosen_completion"], CORRECTION)
        self.assertEqual(job["rejected_completion"], ORIGINAL)

        source_checkpoint = self.runtime.checkpoint_path
        initial_digest = self.runtime.backend.weight_digest()
        trained = self.runtime.learn_preference(job, self.store)

        self.assertTrue(trained["trained"])
        self.assertEqual(trained["step_after"], trained["step_before"] + 1)
        self.assertNotEqual(self.runtime.backend.weight_digest(), initial_digest)
        self.assertEqual(self.runtime.metadata["last_preference_id"], job["id"])
        self.assertNotEqual(self.runtime.checkpoint_path, source_checkpoint)
        self.assertTrue(source_checkpoint.is_file(), "reference checkpoint must remain immutable")
        self.assertTrue(self.runtime.checkpoint_path.is_file())

        reloaded = ModelRuntime(self.root, backend="torch-cpu")
        self.assertTrue(reloaded.ready, reloaded.error)
        persisted_digest = reloaded.backend.weight_digest()
        persisted_checkpoint = reloaded.checkpoint_path
        duplicate = reloaded.learn_preference(job, self.store)

        self.assertEqual(duplicate, trained)
        self.assertEqual(reloaded.backend.weight_digest(), persisted_digest)
        self.assertEqual(reloaded.checkpoint_path, persisted_checkpoint)
        self.assertEqual(reloaded.metadata["step"], trained["step_after"])


if __name__ == "__main__":
    unittest.main()
