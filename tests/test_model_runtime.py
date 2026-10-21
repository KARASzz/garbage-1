"""Integration checks for the optional model and its annotation loop.

Every checkpoint, corpus database, and feedback database in this module lives
under unittest's temporary directory.  The application's data/models tree is
never opened or changed.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_backend import make_corpus
from trash_talk.feedback import FeedbackStore, text_digest
from trash_talk.garbage_model import CharacterTokenizer, build_model, solve_config
from trash_talk.model_runtime import BackgroundLearner, ModelRuntime
import trash_talk.model_runtime as model_runtime_module
from trash_talk.server import create_server


try:
    MLX_METAL_AVAILABLE = importlib.util.find_spec("mlx") is not None
    if MLX_METAL_AVAILABLE:
        import mlx.core as _mlx_core

        MLX_METAL_AVAILABLE = _mlx_core.metal.is_available()
except Exception:
    MLX_METAL_AVAILABLE = False


PREFIXES = [
    {"theme": "daily", "prefix": "我觉得"},
    {"theme": "logic", "prefix": "你把"},
]
REPLAY = [
    {"theme": "daily", "text": "我觉得你像一个装满回声的盒子"},
    {"theme": "logic", "text": "你把逻辑丢在门外了"},
    {"theme": "boast", "text": "我还没认真你就开始找借口"},
    {"theme": "game", "text": "你连开局都能走成残局"},
]
POSITIVE_TEXT = "我一看就知道你把回声当论据"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_app_root(root: Path, *, checkpoint: bool = True) -> Path:
    """Create a disposable app root and, optionally, an exact-budget model."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_json(data_dir / "corpus.json", make_corpus())
    _write_json(
        data_dir / "garbage-1" / "prefixes.json",
        {"format_version": 1, "count": len(PREFIXES), "examples": PREFIXES},
    )
    _write_json(
        data_dir / "garbage-1" / "seed_replay.json",
        {"format_version": 1, "count": len(REPLAY), "examples": REPLAY},
    )

    if not checkpoint:
        return root

    tokenizer = CharacterTokenizer.fit(
        [entry["prefix"] for entry in PREFIXES]
        + [entry["text"] for entry in REPLAY]
        + [POSITIVE_TEXT]
    )
    config = solve_config(tokenizer.vocab_size)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7129)
        model = build_model(config)
    # Make greedy generation emit visible text even from an untrained model.
    # Training still sees the full canonical parameter set and can change it.
    with torch.no_grad():
        model.lm_bias.fill_(-500.0)
        visible_id = tokenizer.encode("我", bos=False)[0]
        model.lm_bias[visible_id] = 500.0

    weights = {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in model.named_parameters()
    }
    self_count = sum(array.size for array in weights.values())
    if self_count != 100_000 or len(weights) != len(set(weights)):
        raise AssertionError("fixture must contain 100000 unique canonical parameters")
    if any(array.dtype != np.float32 for array in weights.values()):
        raise AssertionError("fixture weights must be FP32")
    digest = hashlib.sha256()
    for name, array in sorted(weights.items()):
        digest.update(array.astype("<f4", copy=False).tobytes())

    model_dir = root / "models" / "garbage-1"
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / "bootstrap.npz"
    with weights_path.open("wb") as stream:
        np.savez(stream, **weights)
    metadata_path = checkpoint_dir / "bootstrap.json"
    _write_json(
        metadata_path,
        {
            "format": "garbage-1-checkpoint-v1",
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "weights": weights_path.name,
            "weight_sha256": digest.hexdigest(),
            "step": 0,
            "last_job_id": 0,
            "run_id": "fixture-run-1",
        },
    )
    _write_json(model_dir / "current.json", {"checkpoint": "checkpoints/bootstrap.json"})
    return root


class TemporaryRuntimeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = _make_app_root(Path(self.temp.name))


def _wait_until(predicate, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _checkpoint_state(root: Path) -> tuple[str, dict]:
    model_dir = root / "models" / "garbage-1"
    pointer = json.loads((model_dir / "current.json").read_text(encoding="utf-8"))
    metadata_path = model_dir / pointer["checkpoint"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(metadata_path.parent / metadata["weights"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(array.astype("<f4", copy=False).tobytes())
    return digest.hexdigest(), metadata


class RuntimeIntegrationTests(TemporaryRuntimeCase):
    def test_wrapped_training_assets_reject_bad_count_or_format(self) -> None:
        prefix_path = self.root / "data" / "garbage-1" / "prefixes.json"
        valid_prefixes = json.loads(prefix_path.read_text(encoding="utf-8"))
        self.assertEqual(valid_prefixes["format_version"], 1)
        self.assertEqual(valid_prefixes["count"], len(valid_prefixes["examples"]))
        self.assertTrue(all("prefix" in item and "text" not in item for item in valid_prefixes["examples"]))

        bad_count = {**valid_prefixes, "count": valid_prefixes["count"] + 1}
        _write_json(prefix_path, bad_count)
        invalid_count_runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertFalse(invalid_count_runtime.ready)
        self.assertIn("计数", invalid_count_runtime.error)

        _write_json(prefix_path, valid_prefixes)
        replay_path = self.root / "data" / "garbage-1" / "seed_replay.json"
        valid_replay = json.loads(replay_path.read_text(encoding="utf-8"))
        _write_json(replay_path, {**valid_replay, "format_version": 2})
        invalid_format_runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertFalse(invalid_format_runtime.ready)
        self.assertIn("版本", invalid_format_runtime.error)

    def test_pending_and_downvoted_sentences_never_become_training_targets(self) -> None:
        runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertTrue(runtime.ready, runtime.error)
        store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        before = runtime.backend.weight_digest()

        pending_text = "尚未评价的句子"
        pending_id = store.add_sample(pending_text, "rule", "测试主题", 0)
        self.assertEqual(store.counts()["pending"], 1)
        self.assertEqual(store.positive_examples(), [])
        self.assertEqual(store.queue_counts()["pending"], 0)
        pending_result = runtime.learn(
            {"id": 41, "sample_id": pending_id, "action": "positive", "text": pending_text, "theme": "测试主题"},
            store,
        )
        self.assertFalse(pending_result["trained"])
        self.assertEqual(runtime.backend.weight_digest(), before)
        self.assertEqual(runtime.status()["step"], 0)

        downvoted_text = "这句已经被点踩"
        downvoted_id = store.add_sample(downvoted_text, "rule", "测试主题", 0)
        store.rate(downvoted_id, -1)
        self.assertEqual(store.positive_examples(), [])
        self.assertEqual(store.negative_hashes(), {text_digest(downvoted_text)})
        negative_job = store.claim_job()
        self.assertEqual(negative_job["action"], "exclude")
        negative_result = runtime.learn(negative_job, store)
        self.assertFalse(negative_result["trained"])
        self.assertEqual(negative_result["samples"], 0)
        store.finish_job(negative_job["id"], negative_result)

        # A queued like that is superseded by a dislike must also be filtered
        # using the latest rating when the worker eventually examines it.
        revised_text = "先赞后踩不会变成正例"
        revised_id = store.add_sample(revised_text, "rule", "测试主题", 0)
        store.rate(revised_id, 1)
        store.rate(revised_id, -1)
        stale_like = store.claim_job()
        self.assertEqual(stale_like["action"], "positive")
        self.assertEqual(stale_like["current_rating"], -1)
        stale_result = runtime.learn(stale_like, store)
        self.assertFalse(stale_result["trained"])
        self.assertEqual(stale_result["samples"], 0)
        store.finish_job(stale_like["id"], stale_result)
        exclude_job = store.claim_job()
        self.assertEqual(exclude_job["action"], "exclude")
        exclude_result = runtime.learn(exclude_job, store)
        self.assertFalse(exclude_result["trained"])
        store.finish_job(exclude_job["id"], exclude_result)

        self.assertEqual(store.positive_texts(), [])
        self.assertEqual(store.negative_hashes(), {text_digest(downvoted_text), text_digest(revised_text)})
        self.assertEqual(runtime.backend.weight_digest(), before)
        self.assertEqual(runtime.status()["step"], 0)
        self.assertEqual(store.queue_counts(), {"pending": 0, "running": 0, "done": 3, "error": 0})

    def test_worker_checkpoint_persists_and_recovered_job_does_not_apply_gradient_twice(self) -> None:
        runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertTrue(runtime.ready, runtime.error)
        self.assertEqual(runtime.status()["backend"], "torch-cpu")
        self.assertEqual(runtime.status()["parameters"], 100_000)
        self.assertEqual(runtime.status()["dtype"], "float32")
        store = FeedbackStore(self.root / "data" / "feedback.sqlite3")

        first_id = store.add_sample(POSITIVE_TEXT, "rule", "daily", 0, "fixture-run-1")
        first_vote = store.rate(first_id, 1)
        self.assertTrue(first_vote["changed"])
        self.assertEqual(store.queue_counts(), {"pending": 1, "running": 0, "done": 0, "error": 0})
        initial_digest = runtime.backend.weight_digest()
        learner = BackgroundLearner(runtime, store)
        self.addCleanup(learner.close)
        learner.notify()
        self.assertTrue(_wait_until(lambda: store.queue_counts()["done"] == 1), store.queue_counts())
        learner.close()

        trained_digest, saved = _checkpoint_state(self.root)
        self.assertTrue(saved["weight_sha256"])
        self.assertEqual(trained_digest, saved["weight_sha256"])
        self.assertNotEqual(initial_digest, trained_digest)
        self.assertEqual(runtime.status()["step"], 1)
        self.assertEqual(saved["step"], 1)
        self.assertGreater(float(saved["last_online_loss"]), 0.0)
        self.assertEqual(store.counts()["up"], 1)
        self.assertEqual(store.queue_counts()["done"], 1)

        # Simulate a process dying after the pointer swap but before SQLite
        # acknowledges its running job. The restarted worker reclaims it.
        second_text = "你看我还是能从句子里找到逻辑"
        second_id = store.add_sample(second_text, "garbage-1", "logic", runtime.status()["step"], runtime.status()["run_id"])
        second_job_id = store.rate(second_id, 1)["job_id"]
        interrupted_job = store.claim_job()
        self.assertEqual(interrupted_job["id"], second_job_id)
        pre_crash_digest = runtime.backend.weight_digest()
        first_attempt = runtime.learn(interrupted_job, store)
        self.assertTrue(first_attempt["trained"])
        self.assertEqual((first_attempt["step_before"], first_attempt["step_after"]), (1, 2))
        self.assertNotEqual(pre_crash_digest, runtime.backend.weight_digest())
        committed_digest, committed_metadata = _checkpoint_state(self.root)
        self.assertEqual(committed_metadata["step"], 2)
        self.assertEqual(store.queue_counts()["running"], 1)

        reopened_store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        restarted = ModelRuntime(self.root, backend="torch-cpu")
        self.assertTrue(restarted.ready, restarted.error)
        self.assertEqual(restarted.backend.weight_digest(), committed_digest)
        self.assertEqual(restarted.status()["step"], 2)
        self.assertEqual(reopened_store.counts()["up"], 2)
        self.assertCountEqual(reopened_store.positive_texts(), [POSITIVE_TEXT, second_text])

        recovery = BackgroundLearner(restarted, reopened_store)
        self.addCleanup(recovery.close)
        recovery.notify()
        self.assertTrue(_wait_until(lambda: reopened_store.queue_counts()["done"] == 2), reopened_store.queue_counts())
        recovery.close()
        after_recovery_digest, after_recovery_metadata = _checkpoint_state(self.root)
        self.assertEqual(after_recovery_digest, committed_digest)
        self.assertEqual(after_recovery_metadata["step"], 2)
        self.assertEqual(restarted.backend.weight_digest(), committed_digest)

        connection = reopened_store.connect()
        try:
            attempts, state = connection.execute(
                "SELECT attempts,status FROM training_jobs WHERE id=?", (second_job_id,)
            ).get
            event = connection.execute(
                "SELECT step_before,step_after,trained,examples,loss,weight_sha256 "
                "FROM background_training_events WHERE job_id=?",
                (second_job_id,),
            ).get
        finally:
            connection.close()
        self.assertEqual((attempts, state), (2, "done"))
        self.assertEqual(
            event,
            (
                first_attempt["step_before"],
                first_attempt["step_after"],
                1,
                first_attempt["samples"],
                first_attempt["loss"],
                first_attempt["weight_sha256"],
            ),
        )
        self.assertEqual(event[5], committed_digest)

    def test_worker_retries_a_transient_claim_error_and_continues(self) -> None:
        runtime = ModelRuntime(self.root, backend="torch-cpu")
        store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        sample_id = store.add_sample(POSITIVE_TEXT, "rule", "daily", 0)
        store.rate(sample_id, 1)
        initial_digest = runtime.backend.weight_digest()
        real_claim_job = store.claim_job
        claim_calls = 0

        def fail_first_claim():
            nonlocal claim_calls
            claim_calls += 1
            if claim_calls == 1:
                raise RuntimeError("injected one-time claim failure")
            return real_claim_job()

        store.claim_job = fail_first_claim
        learner = BackgroundLearner(runtime, store)
        self.addCleanup(learner.close)
        learner.notify()
        self.assertTrue(_wait_until(lambda: store.queue_counts()["done"] == 1), store.queue_counts())
        learner.close()

        self.assertGreaterEqual(claim_calls, 2)
        self.assertNotEqual(runtime.backend.weight_digest(), initial_digest)
        self.assertEqual(runtime.status()["step"], 1)
        self.assertEqual(store.queue_counts(), {"pending": 0, "running": 0, "done": 1, "error": 0})

    def test_worker_retries_transient_ack_without_relearning(self) -> None:
        runtime = ModelRuntime(self.root, backend="torch-cpu")
        store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        sample_id = store.add_sample(POSITIVE_TEXT, "rule", "daily", 0)
        job_id = store.rate(sample_id, 1)["job_id"]
        initial_digest = runtime.backend.weight_digest()
        real_finish_job = store.finish_job
        finish_calls = 0

        def fail_first_finish(finished_job_id, result, error=None):
            nonlocal finish_calls
            finish_calls += 1
            if finish_calls == 1:
                raise RuntimeError("injected one-time SQLite acknowledgement failure")
            return real_finish_job(finished_job_id, result, error=error)

        store.finish_job = fail_first_finish
        learner = BackgroundLearner(runtime, store)
        self.addCleanup(learner.close)
        learner.notify()
        self.assertTrue(_wait_until(lambda: store.queue_counts()["done"] == 1), store.queue_counts())
        learner.close()

        self.assertEqual(finish_calls, 2)
        self.assertNotEqual(runtime.backend.weight_digest(), initial_digest)
        self.assertEqual(runtime.status()["step"], 1)
        connection = store.connect()
        try:
            event = connection.execute(
                "SELECT step_before,step_after,trained,examples,loss,weight_sha256 "
                "FROM background_training_events WHERE job_id=?",
                (job_id,),
            ).get
        finally:
            connection.close()
        self.assertEqual(event[0:3], (0, 1, 1))
        self.assertEqual(event[3], 9)
        self.assertGreater(event[4], 0.0)
        self.assertEqual(event[5], runtime.backend.weight_digest())

    def test_second_worker_waits_for_lease_then_uses_latest_checkpoint(self) -> None:
        feedback_path = self.root / "data" / "feedback.sqlite3"
        first_store = FeedbackStore(feedback_path)
        second_store = FeedbackStore(feedback_path)
        first_runtime = ModelRuntime(self.root, backend="torch-cpu")
        second_runtime = ModelRuntime(self.root, backend="torch-cpu")
        self.assertEqual(first_runtime.backend.weight_digest(), second_runtime.backend.weight_digest())

        first_worker = BackgroundLearner(first_runtime, first_store)
        self.addCleanup(first_worker.close)
        first_id = first_store.add_sample(POSITIVE_TEXT, "rule", "daily", 0)
        first_job = first_store.rate(first_id, 1)["job_id"]
        first_worker.notify()
        self.assertTrue(_wait_until(lambda: first_store.queue_counts()["done"] == 1), first_store.queue_counts())
        first_digest = first_runtime.backend.weight_digest()
        self.assertEqual(first_runtime.status()["step"], 1)

        contention_seen = threading.Event()
        real_flock = model_runtime_module.fcntl.flock

        def observe_contended_lease(file_descriptor, operation):
            try:
                return real_flock(file_descriptor, operation)
            except BlockingIOError:
                contention_seen.set()
                raise

        # Ensure the second worker actually attempted the same on-disk lease
        # while the first was still the sole writer.
        with patch.object(model_runtime_module.fcntl, "flock", observe_contended_lease):
            second_worker = BackgroundLearner(second_runtime, second_store)
            self.addCleanup(second_worker.close)
            self.assertTrue(contention_seen.wait(3), "second worker did not wait on the shared lease")

        first_worker.close()
        second_id = second_store.add_sample(
            "你看我还是能从句子里找到逻辑", "rule", "logic", 1, first_runtime.status()["run_id"]
        )
        second_job = second_store.rate(second_id, 1)["job_id"]
        second_worker.notify()
        self.assertTrue(_wait_until(lambda: second_store.queue_counts()["done"] == 2), second_store.queue_counts())

        self.assertEqual(second_runtime.status()["step"], 2)
        self.assertNotEqual(second_runtime.backend.weight_digest(), first_digest)
        connection = second_store.connect()
        try:
            first_event = connection.execute(
                "SELECT step_before,step_after,trained FROM background_training_events WHERE job_id=?",
                (first_job,),
            ).get
            second_event = connection.execute(
                "SELECT step_before,step_after,trained FROM background_training_events WHERE job_id=?",
                (second_job,),
            ).get
        finally:
            connection.close()
        self.assertEqual(first_event, (0, 1, 1))
        self.assertEqual(second_event, (1, 2, 1))

    @unittest.skipUnless(MLX_METAL_AVAILABLE, "MLX Metal is unavailable on this host")
    def test_mlx_runtime_performs_real_inference_online_learning_and_reload(self) -> None:
        import mlx.core as mx

        self.assertTrue(mx.metal.is_available(), "this integration check requires MLX Metal")
        runtime = ModelRuntime(self.root, backend="mlx")
        self.assertTrue(runtime.ready, runtime.error)
        self.assertEqual(runtime.status()["backend"], "mlx")
        initial = runtime.backend.weight_digest()
        generated = runtime.generate(prefix="我觉得", theme="daily", max_new_tokens=3)
        self.assertEqual(generated["text"], "我觉得我我我")
        self.assertEqual(generated["generated_tokens"], 3)

        store = FeedbackStore(self.root / "data" / "feedback.sqlite3")
        sample_id = store.add_sample(POSITIVE_TEXT, "garbage-1", "daily", 0, runtime.status()["run_id"])
        job_id = store.rate(sample_id, 1)["job_id"]
        job = store.claim_job()
        self.assertEqual(job["id"], job_id)
        result = runtime.learn(job, store)
        self.assertTrue(result["trained"])
        self.assertEqual(result["step_after"], 1)
        self.assertNotEqual(initial, runtime.backend.weight_digest())
        store.finish_job(job_id, result)

        persisted_digest, persisted = _checkpoint_state(self.root)
        self.assertEqual(persisted_digest, runtime.backend.weight_digest())
        self.assertEqual(persisted["step"], 1)
        restarted = ModelRuntime(self.root, backend="mlx")
        self.assertTrue(restarted.ready, restarted.error)
        self.assertEqual(restarted.backend.weight_digest(), persisted_digest)
        self.assertEqual(restarted.generate(prefix="我觉得", theme="daily", max_new_tokens=2)["text"], "我觉得我我")
        duplicate = restarted.learn(job, store)
        self.assertTrue(duplicate["trained"])
        self.assertEqual(duplicate, result)
        self.assertEqual(restarted.status()["step"], 1)
        self.assertEqual(restarted.backend.weight_digest(), persisted_digest)


class HttpRuntimeIntegrationTests(TemporaryRuntimeCase):
    @staticmethod
    def _start(root: Path, *, backend: str = "torch-cpu"):
        server = create_server(root, port=0, seed=37, backend=backend)
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        return server, thread

    @staticmethod
    def _request(server, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            request_headers = dict(headers or {})
            payload = None
            if body is not None:
                payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
                request_headers.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            return response.status, response.read(), dict(response.headers.items())
        finally:
            connection.close()

    def _cleanup_server(self, server, thread) -> None:
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.shutdown)

    def test_http_routes_preserve_get_shape_return_complete_results_and_reject_bad_requests(self) -> None:
        server, thread = self._start(self.root)
        self._cleanup_server(server, thread)
        port = server.server_port
        origin = f"http://127.0.0.1:{port}"

        status, body, _ = self._request(server, "GET", "/api/draw")
        self.assertEqual(status, 200)
        self.assertEqual(set(json.loads(body)), {"text", "id", "theme", "block_count"})

        status, body, _ = self._request(
            server, "POST", "/api/draw", {}, {"Origin": origin}
        )
        rule = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(rule),
            {"text", "id", "theme", "block_count", "sample_id", "source", "rating", "model"},
        )
        self.assertEqual(rule["source"], "rule")
        self.assertEqual(rule["rating"], 0)
        self.assertTrue(rule["sample_id"])
        self.assertEqual(rule["model"]["backend"], "torch-cpu")

        status, body, _ = self._request(
            server, "POST", "/api/ai", {}, {"Origin": origin}
        )
        generated = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(
            set(generated),
            {
                "text", "theme", "id", "block_count", "model", "prompt",
                "generated_tokens", "sample_id", "source", "rating",
            },
        )
        self.assertEqual(generated["source"], "garbage-1")
        self.assertTrue(generated["text"])
        self.assertGreater(generated["generated_tokens"], 0)
        self.assertTrue(generated["sample_id"])
        self.assertEqual(generated["model"]["ready"], True)

        for route in ("/api/ai", "/api/draw"):
            with self.subTest(route=route):
                code, _, _ = self._request(
                    server, "POST", route, {}, {"Origin": "http://evil.example"}
                )
                self.assertEqual(code, 403)
        code, _, _ = self._request(
            server,
            "POST",
            "/api/feedback",
            {"sample_id": generated["sample_id"], "rating": 1},
            {"Host": "evil.example", "Origin": origin},
        )
        self.assertEqual(code, 403)
        code, _, _ = self._request(
            server,
            "POST",
            "/api/feedback",
            {"sample_id": generated["sample_id"], "rating": 1},
            {"Content-Type": "text/plain", "Origin": origin},
        )
        self.assertEqual(code, 415)
        code, body, _ = self._request(
            server,
            "POST",
            "/api/feedback",
            {"sample_id": generated["sample_id"], "rating": True},
            {"Origin": origin},
        )
        self.assertEqual(code, 400)
        self.assertIn("评价", json.loads(body)["error"])

        code, body, _ = self._request(
            server,
            "POST",
            "/api/feedback",
            {"sample_id": generated["sample_id"], "rating": 1},
            {"Origin": origin},
        )
        feedback = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(set(feedback), {"sample_id", "rating", "changed", "counts", "message", "job_id"})
        self.assertEqual(feedback["sample_id"], generated["sample_id"])
        self.assertEqual(feedback["rating"], 1)
        self.assertTrue(feedback["changed"])
        self.assertIsNotNone(feedback["job_id"])

        code, body, _ = self._request(server, "GET", "/api/model")
        model = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(
            set(model),
            {
                "name", "parameters", "dtype", "temperature", "backend", "ready",
                "step", "run_id", "architecture", "error", "counts", "queue",
            },
        )
        self.assertEqual(model["name"], "garbage-1")
        self.assertEqual(model["backend"], "torch-cpu")
        self.assertEqual(model["ready"], True)
        self.assertIn("counts", model)
        self.assertIn("queue", model)
        self.assertEqual(model["counts"]["up"], 1)
        self.assertEqual(set(model["queue"]), {"pending", "running", "done", "error"})

    def test_missing_model_keeps_rule_post_available_and_returns_ai_503(self) -> None:
        with tempfile.TemporaryDirectory() as missing_temp:
            missing_root = _make_app_root(Path(missing_temp), checkpoint=False)
            for name in ("index.html", "style.css", "app.js", "favicon.svg"):
                source = Path(__file__).resolve().parents[1] / "web" / name
                target = missing_root / "web" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
            server, thread = self._start(missing_root)
            try:
                origin = f"http://127.0.0.1:{server.server_port}"

                status, body, _ = self._request(
                    server, "POST", "/api/ai", {}, {"Origin": origin}
                )
                self.assertEqual(status, 503)
                self.assertIn("规则抽卡仍可用", json.loads(body)["error"])

                status, body, _ = self._request(
                    server, "POST", "/api/draw", {}, {"Origin": origin}
                )
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["source"], "rule")

                status, body, _ = self._request(server, "GET", "/api/model")
                self.assertEqual(status, 200)
                self.assertFalse(json.loads(body)["ready"])
            finally:
                server.shutdown()
                thread.join(3)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
