"""Temporary CPU integration checks for immutable LoRA base synchronization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_lora import _make_lora_app_root
from trash_talk.feedback import FeedbackStore, text_digest
from trash_talk.lora import LoRARuntime, STYLE_NAMES, StyleHub
from trash_talk.lora_sync import LoRASyncManager
from trash_talk.model_runtime import ModelRuntime


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_root(
    parent: Path,
    *,
    styles=(),
    incompatible: str | None = None,
    change_weights: bool = True,
) -> Path:
    """Complete the LoRA fixture with a separate, loadable main model."""
    root = _make_lora_app_root(parent)
    source = parent / "fixture-source"
    shutil.copy2(source / "data" / "corpus.json", root / "data" / "corpus.json")
    shutil.copytree(source / "data" / "garbage-1", root / "data" / "garbage-1", dirs_exist_ok=True)
    shutil.copytree(source / "models" / "garbage-1", root / "models" / "garbage-1", dirs_exist_ok=True)

    model_dir = root / "models" / "garbage-1"
    pointer = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    metadata_path = model_dir / pointer["checkpoint"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(metadata_path.parent / metadata["weights"], allow_pickle=False) as archive:
        weights = {name: archive[name].copy() for name in archive.files}
    # A small real FP32 change selects a new immutable target SHA while leaving
    # the architecture, vocabulary, and fixture loss well within tolerance.
    if change_weights:
        weights["lm_bias"][1] += np.float32(1e-3)
    with (metadata_path.parent / metadata["weights"]).open("wb") as stream:
        np.savez(stream, **weights)
    digest = hashlib.sha256()
    for name, array in sorted(weights.items()):
        digest.update(array.astype("<f4", copy=False).tobytes())
    metadata["weight_sha256"] = digest.hexdigest()

    if incompatible == "config":
        config = metadata["config"]
        config["heads"] = config["d_model"] if config["heads"] != config["d_model"] else 1
    elif incompatible == "tokenizer":
        characters = metadata["tokenizer"]["characters"]
        characters[-1] = "\U0010ffff"
    elif incompatible is not None:
        raise AssertionError(f"unknown incompatibility fixture: {incompatible}")
    _write_json(metadata_path, metadata)

    for style in styles:
        runtime = LoRARuntime(root, style, backend="torch-cpu", allow_untrained=True)
        runtime._save(dict(runtime.metadata))
    return root


def _start_hub(root: Path, *, steps=2) -> tuple[StyleHub, LoRASyncManager]:
    base_runtime = ModelRuntime(root, backend="torch-cpu")
    if not base_runtime.ready:
        raise AssertionError(base_runtime.error)
    hub = StyleHub(root, "torch-cpu", base_runtime=base_runtime)
    manager = LoRASyncManager(hub, steps=steps, batch_size=2)
    # Use the deliberately small test settings instead of the lazy production default.
    hub._sync = manager
    return hub, manager


def _wait_for_terminal(manager: LoRASyncManager, timeout: float = 120) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.status()["job"]
        if job is not None and job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"LoRA sync did not finish: {manager.status()}")


def _current_pointer(root: Path, style: str) -> Path:
    return root / "models" / "garbage-1" / "lora" / style / "current.json"


def _read_style_metadata(root: Path, style: str) -> tuple[Path, dict]:
    pointer_path = _current_pointer(root, style)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    metadata_path = pointer_path.parent / pointer["checkpoint"]
    return metadata_path, json.loads(metadata_path.read_text(encoding="utf-8"))


class LoRASyncTests(unittest.TestCase):
    def test_sync_rebases_all_checkpointed_styles_reload_and_skip_same_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            styles = tuple(STYLE_NAMES)
            root = _sync_root(parent, styles=styles)
            old_base = root / "models" / "garbage-1" / "lora" / "base"
            old_base_snapshot = {p.relative_to(old_base): p.read_bytes() for p in old_base.rglob("*") if p.is_file()}
            old_style_files = {}
            for style in styles:
                metadata_path, metadata = _read_style_metadata(root, style)
                self.assertNotIn("base_ref", metadata)  # Exercise the legacy `lora/base` pointer.
                weight_path = metadata_path.parent / metadata["weights"]
                old_style_files[style] = (metadata_path, metadata_path.read_bytes(), weight_path, weight_path.read_bytes())

            # A rated example without a queued learner job verifies migration replay
            # includes human positives while keeping this test deterministic.
            store = FeedbackStore(root / "data" / "lora" / "mouthy.sqlite3")
            positive_text = "这条人工正样本会加入嘴臭风格的迁移训练"
            sample_id = store.add_sample(positive_text, "garbage-1", "daily", 0, "sync-test")
            connection = store.connect()
            try:
                connection.execute(
                    "INSERT INTO ratings(content_sha256,sample_id,rating) VALUES (?,?,1)",
                    (text_digest(positive_text), sample_id),
                )
            finally:
                connection.close()

            hub, manager = _start_hub(root)
            self.addCleanup(hub.close)
            entered_pin, release_pin = threading.Event(), threading.Event()
            original_pin = manager._pin

            def pause_pin(job):
                entered_pin.set()
                if not release_pin.wait(10):
                    raise TimeoutError("test did not release pinned-base barrier")
                return original_pin(job)

            manager._pin = pause_pin
            try:
                initial = manager.start()
                self.assertTrue(entered_pin.wait(10), "sync worker did not reach target pinning")
                duplicate = manager.start()
                self.assertEqual(duplicate["id"], initial["id"])
            finally:
                release_pin.set()

            job = _wait_for_terminal(manager)
            self.assertEqual(job["status"], "completed", job["message"])
            self.assertEqual({item["status"] for item in job["results"]}, {"synced"})
            target_sha = job["target_sha256"]
            self.assertNotEqual(target_sha, json.loads(old_style_files["mouthy"][1])["base_sha256"])
            self.assertEqual(manager.status()["latest_base_step"], job["target_step"])

            mouthy_metadata = _read_style_metadata(root, "mouthy")[1]
            self.assertTrue(mouthy_metadata["base_ref"].startswith(f"bases/{target_sha}-"))
            target_base = root / "models" / "garbage-1" / "lora" / mouthy_metadata["base_ref"]
            self.assertEqual(json.loads((target_base / "current.json").read_text(encoding="utf-8")), {"checkpoint": "base.json"})
            self.assertTrue((target_base / "base.npz").is_file())
            self.assertEqual(json.loads((target_base / "base.json").read_text(encoding="utf-8"))["weight_sha256"], target_sha)
            for style in styles:
                metadata_path, metadata = _read_style_metadata(root, style)
                self.assertEqual(metadata["base_ref"], mouthy_metadata["base_ref"])
                self.assertEqual(metadata["base_sha256"], target_sha)
                self.assertEqual(metadata["migration"]["to_base_sha256"], target_sha)
                prior_meta_path, prior_meta_bytes, prior_weight_path, prior_weight_bytes = old_style_files[style]
                self.assertEqual(prior_meta_path.read_bytes(), prior_meta_bytes)
                self.assertEqual(prior_weight_path.read_bytes(), prior_weight_bytes)
                reloaded = LoRARuntime(root, style, backend="torch-cpu")
                self.assertTrue(reloaded.ready, reloaded.error)
                self.assertEqual(reloaded.base_sha, target_sha)
                self.assertEqual(reloaded.metadata["base_ref"], metadata["base_ref"])

            migration = _read_style_metadata(root, "mouthy")[1]["migration"]
            self.assertEqual(migration["positive_sentences"], 1)
            self.assertEqual(
                old_base_snapshot,
                {p.relative_to(old_base): p.read_bytes() for p in old_base.rglob("*") if p.is_file()},
            )
            self.assertTrue(all(entry["ready"] for entry in hub.status()["styles"][1:]))

            pointers_after_migration = {style: _current_pointer(root, style).read_bytes() for style in styles}
            manager.thread.join(timeout=5)
            self.assertFalse(manager.thread.is_alive(), "completed sync still holds its single-writer lease")
            second = manager.start()
            skipped_job = _wait_for_terminal(manager)
            self.assertNotEqual(second["id"], job["id"])
            self.assertEqual(skipped_job["status"], "completed")
            self.assertEqual({item["status"] for item in skipped_job["results"]}, {"skipped"})
            self.assertEqual(
                pointers_after_migration,
                {style: _current_pointer(root, style).read_bytes() for style in styles},
            )

    def test_config_or_tokenizer_mismatch_is_saved_and_keeps_legacy_pointer(self) -> None:
        for incompatibility in ("config", "tokenizer"):
            with self.subTest(incompatibility=incompatibility), tempfile.TemporaryDirectory() as temporary:
                # Keep the checkpoint weights identical to the old base so this
                # also guards against an unsafe same-SHA early-skip shortcut.
                root = _sync_root(
                    Path(temporary),
                    styles=("mouthy",),
                    incompatible=incompatibility,
                    change_weights=False,
                )
                pointer_path = _current_pointer(root, "mouthy")
                old_pointer = pointer_path.read_bytes()
                hub, manager = _start_hub(root)
                try:
                    with patch("trash_talk.lora_sync.LOGGER.exception"):
                        manager.start()
                        job = _wait_for_terminal(manager)
                    self.assertEqual(job["status"], "partial", job["message"])
                    mouthy = next(item for item in job["results"] if item["style"] == "mouthy")
                    self.assertEqual(mouthy["status"], "failed")
                    self.assertIn("架构或词表映射改变", mouthy["message"])
                    self.assertEqual(job["target_sha256"], _read_style_metadata(root, "mouthy")[1]["base_sha256"])
                    self.assertEqual(pointer_path.read_bytes(), old_pointer)
                    self.assertTrue((pointer_path.parent / json.loads(old_pointer)["checkpoint"]).is_file())
                    saved = json.loads((root / "data" / "lora" / "sync-state.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved["status"], "partial")
                finally:
                    hub.close()

    def test_validation_regression_is_recorded_without_replacing_style_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _sync_root(Path(temporary), styles=("mouthy",))
            pointer_path = _current_pointer(root, "mouthy")
            old_pointer = pointer_path.read_bytes()
            _, old_metadata = _read_style_metadata(root, "mouthy")
            old_sha = old_metadata["base_sha256"]
            hub, manager = _start_hub(root)
            self.addCleanup(hub.close)

            def regressed_validation(backend, tokenizer, context, texts):
                return 1.0 if backend.base_digest() == old_sha else 2.0

            with (
                patch("trash_talk.lora_sync.validation_loss", side_effect=regressed_validation),
                patch("trash_talk.lora_sync.LOGGER.exception"),
            ):
                manager.start()
                job = _wait_for_terminal(manager)

            self.assertEqual(job["status"], "partial")
            mouthy = next(item for item in job["results"] if item["style"] == "mouthy")
            self.assertEqual(mouthy["status"], "failed")
            self.assertIn("迁移验证退化", mouthy["message"])
            self.assertEqual(pointer_path.read_bytes(), old_pointer)
            self.assertEqual(_read_style_metadata(root, "mouthy")[1]["base_sha256"], old_sha)
            saved = json.loads((root / "data" / "lora" / "sync-state.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["results"][0]["status"], "failed")

    def test_restart_recovers_published_pointer_before_sync_state_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _sync_root(Path(temporary), styles=tuple(STYLE_NAMES))
            hub, manager = _start_hub(root)
            try:
                first = manager.start()
                completed = _wait_for_terminal(manager)
                self.assertEqual(completed["status"], "completed", completed["message"])
                manager.thread.join(timeout=5)
                self.assertFalse(manager.thread.is_alive())
                pointers = {style: _current_pointer(root, style).read_bytes() for style in STYLE_NAMES}
                metadata = _read_style_metadata(root, "mouthy")[1]
                self.assertEqual(metadata["migration"]["job_id"], first["id"])
            finally:
                hub.close()

            state_path = root / "data" / "lora" / "sync-state.json"
            interrupted = json.loads(state_path.read_text(encoding="utf-8"))
            interrupted["status"] = "queued"
            mouthy_result = next(item for item in interrupted["results"] if item["style"] == "mouthy")
            mouthy_result["status"] = "running"
            _write_json(state_path, interrupted)

            resumed_hub, resumed_manager = _start_hub(root)
            try:
                resumed = _wait_for_terminal(resumed_manager)
                self.assertEqual(resumed["id"], first["id"])
                self.assertEqual(resumed["status"], "completed", resumed["message"])
                resumed_mouthy = next(item for item in resumed["results"] if item["style"] == "mouthy")
                self.assertEqual(resumed_mouthy["status"], "synced")
                self.assertEqual(
                    {style: _current_pointer(root, style).read_bytes() for style in STYLE_NAMES},
                    pointers,
                )
                self.assertEqual(_read_style_metadata(root, "mouthy")[1]["migration"]["job_id"], first["id"])
                saved = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["status"], "completed")
            finally:
                resumed_hub.close()

    def test_call_exclusive_holds_learner_lease_without_blocking_inference_or_losing_feed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _sync_root(Path(temporary), styles=("mouthy",), change_weights=False)
            hub, _ = _start_hub(root)
            runtime, store, learner = hub.get("mouthy")
            entered, release = threading.Event(), threading.Event()
            callback_thread = []
            callback_result, callback_error = [], []

            def exclusive_callback():
                callback_thread.append(threading.get_ident())
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release exclusive learner callback")
                return "exclusive-result"

            def invoke_exclusive():
                try:
                    callback_result.append(learner.call_exclusive(exclusive_callback, timeout=20))
                except Exception as error:
                    callback_error.append(error)

            caller = threading.Thread(target=invoke_exclusive, name="test-call-exclusive")
            caller.start()
            try:
                self.assertTrue(entered.wait(10), "exclusive callback did not run on the learner worker")
                self.assertEqual(callback_thread, [learner.thread.ident])

                text = "独占同步期间送来的人工正样本"
                sample_id = store.add_sample(text, "garbage-1", "daily", runtime.metadata["step"], runtime.metadata["run_id"])
                queued = store.rate(sample_id, 1)
                job_id = queued["job_id"]
                self.assertIsNotNone(job_id)
                learner.notify()
                self.assertEqual(store.queue_counts()["pending"], 1)

                generated = runtime.generate(prefix="我觉得", theme="daily", max_new_tokens=2)
                self.assertTrue(generated["text"].strip())
                self.assertEqual(store.queue_counts()["pending"], 1)
                release.set()
                caller.join(timeout=20)
                self.assertFalse(caller.is_alive(), "call_exclusive did not return after its callback was released")
                self.assertEqual(callback_error, [])
                self.assertEqual(callback_result, ["exclusive-result"])
                deadline = time.monotonic() + 30
                status = None
                while time.monotonic() < deadline:
                    connection = store.connect()
                    try:
                        status = connection.execute("SELECT status FROM training_jobs WHERE id=?", (job_id,)).get
                    finally:
                        connection.close()
                    if status in ("done", "error"):
                        break
                    time.sleep(0.05)
                self.assertEqual(status, "done", f"queued training job was lost or failed: {store.queue_counts()}")
                self.assertEqual(store.queue_counts()["pending"], 0)
                self.assertIn(text, {example["text"] for example in store.positive_examples()})
            finally:
                release.set()
                caller.join(timeout=20)
                hub.close()


if __name__ == "__main__":
    unittest.main()
