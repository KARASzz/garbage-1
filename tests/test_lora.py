"""LoRA adapter checks with all model and annotation state kept temporary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tests.test_model_runtime import _make_app_root
from trash_talk.feedback import FeedbackStore
from trash_talk.lora import (
    STYLE_NAMES,
    LoRARuntime,
    StyleHub,
    initial_adapter,
    make_backend,
)
from trash_talk.model_runtime import ModelRuntime


try:
    MLX_AVAILABLE = importlib.util.find_spec("mlx") is not None
    if MLX_AVAILABLE:
        import mlx.core as mx

        MLX_AVAILABLE = mx.metal.is_available()
except Exception:
    MLX_AVAILABLE = False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_lora_app_root(parent: Path) -> Path:
    """Build a disposable app root and copy its fixture checkpoint as pinned base."""
    source = parent / "fixture-source"
    app = parent / "app"
    _make_app_root(source)

    shutil.copytree(source / "data" / "garbage-1", app / "data" / "garbage-1")
    base_parent = app / "models" / "garbage-1" / "lora"
    base_parent.mkdir(parents=True, exist_ok=True)
    # Copy from the sibling fixture root so the destination is never inside
    # the source models/garbage-1 tree.
    shutil.copytree(source / "models" / "garbage-1", base_parent / "base")

    styles = []
    for style in STYLE_NAMES:
        styles.append(
            {
                "id": style,
                "train": [f"训练语料{style}甲"],
                "validation": [f"验证语料{style}乙"],
            }
        )
    _write_json(
        app / "data" / "lora" / "styles.json",
        {"format": "garbage-1-lora-styles-v1", "styles": styles},
    )
    return app


class AdapterShapeTests(unittest.TestCase):
    def test_rank_eight_attention_and_ffn_adapter_budget(self) -> None:
        config = SimpleNamespace(d_model=160, ffn_dims=[858, 859])
        arrays = initial_adapter(config, rank=8, include_ffn=True)
        self.assertEqual(sum(a.size for a in arrays.values()), 47_952)

    def test_rank_four_attention_adapter_has_7680_parameters(self) -> None:
        config = SimpleNamespace(d_model=160)
        arrays = initial_adapter(config, rank=4)

        self.assertEqual(len(arrays), 8)
        self.assertTrue(all(name.endswith((".A", ".B")) for name in arrays))
        self.assertEqual(sum(array.size for array in arrays.values()), 7680)


class TemporaryLoRACase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = _make_lora_app_root(Path(self.temp.name))
        self.base_dir = self.root / "models" / "garbage-1" / "lora" / "base"

    def _base(self) -> ModelRuntime:
        runtime = ModelRuntime(self.root, backend="torch-cpu", model_dir=self.base_dir)
        self.assertTrue(runtime.ready, runtime.error)
        return runtime


class LoRABackendTests(TemporaryLoRACase):
    @unittest.skipUnless(MLX_AVAILABLE, "MLX Metal is unavailable on this host")
    def test_attention_and_ffn_lora_freezes_base_and_matches_backends(self) -> None:
        base = self._base()
        arrays = initial_adapter(base.config, rank=8, include_ffn=True)
        rng = np.random.default_rng(71)
        for k, array in arrays.items():
            if k.endswith(".B"):
                arrays[k] = rng.normal(0, .002, size=array.shape).astype(np.float32)
        mlx = make_backend(base.config, base.backend.export_weights(), arrays, "mlx", 8, 16)
        torch_backend = make_backend(base.config, base.backend.export_weights(), arrays, "torch-cpu", 8, 16)
        ids = [[1, 4, 5, 6]]
        np.testing.assert_allclose(np.asarray(mlx.logits(ids)), torch_backend.logits(ids).cpu().numpy(), atol=1e-4, rtol=1e-4)
        for backend in (mlx, torch_backend):
            before = backend.base_digest()
            adapter_before = backend.weight_digest()
            loss = backend.train_step(ids, [[4, 5, 6, 2]])
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(backend.base_digest(), before)
            self.assertNotEqual(backend.weight_digest(), adapter_before)

    def test_torch_cpu_zero_adapter_matches_base_and_training_freezes_base(self) -> None:
        base = self._base()
        weights = base.backend.export_weights()
        arrays = initial_adapter(base.config, rank=4)
        adapter = make_backend(base.config, weights, arrays, backend="torch-cpu")
        ids = [[1, 4, 5, 6]]
        ids_tensor = torch.tensor(ids, dtype=torch.long)

        with torch.no_grad():
            expected = base.backend.model(ids_tensor).cpu().numpy()
        np.testing.assert_array_equal(adapter.logits(ids).cpu().numpy(), expected)

        before_base = adapter.base.weight_digest()
        before_adapter = adapter.export_weights()
        loss = adapter.train_step(
            x=[[1, 4, 5, 6]],
            y=[[4, 5, 6, 2]],
            learning_rate=0.01,
            pad_id=0,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(adapter.weight_digest(), before_adapter_digest := _adapter_digest(before_adapter))
        self.assertEqual(adapter.base.weight_digest(), before_base)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.base.model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in adapter.base.model.parameters()))
        self.assertEqual(set(adapter.export_weights()), set(arrays))

    @unittest.skipUnless(MLX_AVAILABLE, "MLX Metal is unavailable on this host")
    def test_mlx_zero_adapter_matches_base_and_nonzero_logits_match_torch(self) -> None:
        from trash_talk.lora import make_backend

        base = self._base()
        weights = base.backend.export_weights()
        arrays = initial_adapter(base.config, rank=4, seed=17)
        mlx_adapter = make_backend(base.config, weights, arrays, backend="mlx")
        ids = [[1, 4, 5, 6]]

        from trash_talk.mlx_backend import MLXBackend

        mlx_base = MLXBackend(base.config, weights)
        np.testing.assert_array_equal(np.asarray(mlx_adapter.logits(ids)), np.asarray(mlx_base.logits(ids)))

        nonzero = {name: value.copy() for name, value in arrays.items()}
        rng = np.random.default_rng(903)
        for name in nonzero:
            if name.endswith(".B"):
                nonzero[name] = rng.normal(0.0, 0.002, size=nonzero[name].shape).astype(np.float32)
        mlx_adapter = make_backend(base.config, weights, nonzero, backend="mlx")
        torch_adapter = make_backend(base.config, weights, nonzero, backend="torch-cpu")
        mlx_logits = np.asarray(mlx_adapter.logits(ids))
        torch_logits = torch_adapter.logits(ids).cpu().numpy()

        self.assertLess(float(np.max(np.abs(mlx_logits - torch_logits))), 1e-4)
        np.testing.assert_allclose(mlx_logits, torch_logits, rtol=1e-4, atol=1e-4)

    @unittest.skipUnless(MLX_AVAILABLE, "MLX Metal is unavailable on this host")
    def test_mlx_training_changes_adapter_and_leaves_base_unchanged(self) -> None:
        base = self._base()
        weights = base.backend.export_weights()
        arrays = initial_adapter(base.config, rank=4, seed=29)
        adapter = make_backend(base.config, weights, arrays, backend="mlx")
        before_base = adapter.base_digest()
        before_adapter = adapter.weight_digest()

        loss = adapter.train_step(
            x=[[1, 4, 5, 6]],
            y=[[4, 5, 6, 2]],
            learning_rate=0.01,
            pad_id=0,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertNotEqual(adapter.weight_digest(), before_adapter)
        self.assertEqual(adapter.base_digest(), before_base)
        self.assertTrue(all(name.endswith((".A", ".B")) for name in adapter.export_weights()))


def _adapter_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


class LoRARuntimePersistenceTests(TemporaryLoRACase):
    def test_learn_saves_only_adapter_reload_checks_base_sha(self) -> None:
        runtime = LoRARuntime(self.root, "mouthy", backend="torch-cpu", allow_untrained=True)
        self.assertTrue(runtime.ready, runtime.error)
        original_base_sha = runtime.base_sha
        base_snapshot = {
            path.relative_to(self.base_dir): path.read_bytes()
            for path in self.base_dir.rglob("*")
            if path.is_file()
        }

        store = FeedbackStore(self.root / "data" / "lora" / "mouthy.sqlite3")
        sample_id = store.add_sample("我一看就知道你把回声当论据", "garbage-1", "daily", 0, "lora-test")
        job_id = store.rate(sample_id, 1)["job_id"]
        job = store.claim_job()
        self.assertEqual(job["id"], job_id)
        result = runtime.learn(job, store)
        self.assertTrue(result["trained"])
        self.assertEqual(result["step_after"], 1)
        self.assertEqual(runtime.backend.base_digest(), original_base_sha)

        pointer_path = self.root / "models" / "garbage-1" / "lora" / "mouthy" / "current.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        metadata_path = pointer_path.parent / pointer["checkpoint"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["base_sha256"], original_base_sha)
        with np.load(metadata_path.parent / metadata["weights"], allow_pickle=False) as archive:
            saved = {name: archive[name] for name in archive.files}
        self.assertEqual(set(saved), set(runtime.backend.export_weights()))
        self.assertTrue(saved)
        self.assertTrue(all(name.endswith((".A", ".B")) for name in saved))
        self.assertEqual(
            base_snapshot,
            {
                path.relative_to(self.base_dir): path.read_bytes()
                for path in self.base_dir.rglob("*")
                if path.is_file()
            },
        )

        reloaded = LoRARuntime(self.root, "mouthy", backend="torch-cpu")
        self.assertTrue(reloaded.ready, reloaded.error)
        self.assertEqual(reloaded.status()["step"], 1)
        self.assertEqual(reloaded.base_sha, original_base_sha)
        self.assertEqual(reloaded.backend.weight_digest(), runtime.backend.weight_digest())

        base_checkpoint = self.base_dir / "checkpoints" / "bootstrap.npz"
        with np.load(base_checkpoint, allow_pickle=False) as archive:
            changed_base = {name: archive[name].copy() for name in archive.files}
        changed_base["lm_bias"][0] += np.float32(0.125)
        with base_checkpoint.open("wb") as stream:
            np.savez(stream, **changed_base)
        base_metadata_path = self.base_dir / "checkpoints" / "bootstrap.json"
        base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        for name, value in sorted(changed_base.items()):
            digest.update(value.astype("<f4", copy=False).tobytes())
        base_metadata["weight_sha256"] = digest.hexdigest()
        _write_json(base_metadata_path, base_metadata)

        with self.assertRaisesRegex(ValueError, "LoRA 与固定底座不匹配"):
            LoRARuntime(self.root, "mouthy", backend="torch-cpu")


class StyleHubRoutingTests(TemporaryLoRACase):
    def test_same_text_ratings_stay_in_style_database_and_out_of_base(self) -> None:
        text = "同一句只在自己的风格里留下评价"
        mouthy_store = FeedbackStore(self.root / "data" / "lora" / "mouthy.sqlite3")
        mad_store = FeedbackStore(self.root / "data" / "lora" / "mad.sqlite3")
        base_store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        mouthy_id = mouthy_store.add_sample(text, "garbage-1", "daily", 0, "mouthy-run")
        mad_store.add_sample(text, "garbage-1", "daily", 0, "mad-run")
        base_store.add_sample(text, "garbage-1", "daily", 0, "base-run")

        fake_learner = Mock()
        hub = StyleHub(self.root, "torch-cpu")
        with patch("trash_talk.lora.LoRARuntime", return_value=Mock()), patch(
            "trash_talk.lora.BackgroundLearner", return_value=fake_learner
        ):
            result = hub.rate(mouthy_id, 1)

        self.assertEqual(result["style"], "mouthy")
        self.assertEqual(mouthy_store.counts()["up"], 1)
        self.assertEqual(mad_store.counts()["up"], 0)
        self.assertEqual(base_store.counts()["up"], 0)
        fake_learner.notify.assert_called_once_with()
        hub.close()


if __name__ == "__main__":
    unittest.main()
