"""Versioned FP32 checkpoints, native inference, and a durable quiet learner.

Human annotations remain in SQLite, independent of architecture/weights. A
checkpoint pointer is replaced only after both new immutable files exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import random
import queue
import threading
import uuid

from .feedback import text_digest

LOGGER = logging.getLogger(__name__)
BACKEND_REGISTRY = {}


def register_backend(name, factory):
    if name in BACKEND_REGISTRY or not callable(factory):
        raise ValueError("后端名称重复或 factory 无效")
    BACKEND_REGISTRY[name] = factory


def _mlx(config, weights):
    from .mlx_backend import MLXBackend
    return MLXBackend(config, weights)


def _cuda(config, weights):
    from .torch_backend import TorchBackend
    return TorchBackend(config, weights, device="cuda")


def _cpu(config, weights):
    from .torch_backend import TorchBackend
    return TorchBackend(config, weights, device="cpu")


register_backend("mlx", _mlx)
register_backend("cuda", _cuda)
register_backend("torch-cpu", _cpu)


def _relative_file(parent, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("checkpoint 路径必须是相对路径")
    path = (parent / relative).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("checkpoint 路径越界")
    return path


def _atomic_json(path, value):
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _sync_directory(path.parent)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_examples(path, *, prefixes=False):
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        examples = value.get("examples")
        if value.get("format_version") != 1 or not isinstance(examples, list) or value.get("count") != len(examples):
            raise ValueError(f"{path.name} 的版本、计数或 examples 无效")
    else:
        examples = value  # Read older array-only development checkpoints too.
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"{path.name} 缺少可信样例")
    result = []
    for item in examples:
        if not isinstance(item, dict):
            raise ValueError(f"{path.name} 样例必须是对象")
        text = item.get("prefix", item.get("text")) if prefixes else item.get("text")
        if not isinstance(text, str) or not text.strip() or item.get("theme") not in ("game", "logic", "boast", "daily"):
            raise ValueError(f"{path.name} 样例文本或主题无效")
        result.append({"text": text, "theme": item["theme"]})
    return result


class ModelRuntime:
    def __init__(self, root: Path, *, backend="mlx", model_dir=None):
        self.root = Path(root)
        self.model_dir = Path(model_dir or self.root / "models" / "garbage-1")
        self.backend_name = backend
        self.lock = threading.RLock()
        self.checkpoint_path = None
        self.backend = None
        self.metadata = {}
        self.error = None
        self.prefixes = []
        self.replay = []
        pointer = self.model_dir / "current.json"
        if not pointer.exists():
            self.error = "初始权重不存在，请先准备 garbage-1 checkpoint"
            return
        try:
            import numpy as np
            from .garbage_model import CharacterTokenizer, ModelConfig
            entry = json.loads(pointer.read_text())
            metadata_path = _relative_file(self.model_dir, entry["checkpoint"])
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("format") != "garbage-1-checkpoint-v1":
                raise ValueError("未知 checkpoint 格式；保留原权重")
            self.config = ModelConfig.from_dict(metadata["config"])
            self.tokenizer = CharacterTokenizer.from_dict(metadata["tokenizer"])
            if self.config.vocab_size != self.tokenizer.vocab_size:
                raise ValueError("词表和模型配置不匹配")
            weights_path = _relative_file(metadata_path.parent, metadata["weights"])
            with np.load(weights_path, allow_pickle=False) as archive:
                weights = {name: archive[name] for name in archive.files}
            parameter_count = sum(array.size for array in weights.values())
            if parameter_count != self.config.target_parameters or any(
                array.dtype != np.float32 for array in weights.values()
            ):
                raise ValueError(
                    f"checkpoint 参数数或精度错误：实际 {parameter_count} 个，"
                    f"配置 {self.config.target_parameters} 个 FP32 参数"
                )
            if metadata.get("parameter_count", parameter_count) != parameter_count:
                raise ValueError("checkpoint 元数据中的参数数与权重不一致")
            self.backend = BACKEND_REGISTRY[backend](self.config, weights)
            if metadata.get("weight_sha256") and metadata["weight_sha256"] != self.backend.weight_digest():
                raise ValueError("权重摘要不匹配")
            self.metadata = metadata
            self.checkpoint_path = metadata_path
            data_dir = self.root / "data" / "garbage-1"
            self.prefixes = _load_examples(data_dir / "prefixes.json", prefixes=True)
            self.replay = _load_examples(data_dir / "seed_replay.json")
        except Exception as error:
            self.backend = None
            self.error = str(error)
            LOGGER.exception("garbage-1 load failed; rule generator remains available")

    @property
    def ready(self):
        return self.backend is not None

    def reload(self):
        return ModelRuntime(self.root, backend=self.backend_name, model_dir=self.model_dir)

    def status(self):
        metadata = self.metadata
        parameters = metadata.get("parameter_count", metadata.get("config", {}).get("target_parameters", 100000))
        return {"name": "garbage-1", "parameters": parameters, "dtype": "float32", "temperature": 0,
                "backend": self.backend_name, "ready": self.ready, "step": metadata.get("step", 0),
                "run_id": metadata.get("run_id", "uninitialized"), "architecture": metadata.get("config", {}).get("architecture", "dense_transformer_v1"),
                "error": self.error}

    def generate(self, *, prefix=None, theme=None, max_new_tokens=64):
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")
        with self.lock:
            chosen = random.SystemRandom().choice(self.prefixes) if prefix is None else {"text": prefix, "theme": theme}
            prefix_text = chosen.get("text", chosen.get("prefix"))
            prompt = self.tokenizer.encode(prefix_text, theme=chosen["theme"], bos=True)
            ids = self.backend.generate(prompt, max_new_tokens, self.tokenizer.eos_id)
            text = self.tokenizer.decode(ids)
            if not text.strip():
                raise RuntimeError("模型生成了空输出")
            status = self.status()
            return {"text": text, "theme": chosen["theme"], "id": uuid.uuid4().hex, "block_count": 0,
                    "model": status, "prompt": prefix_text, "generated_tokens": len(ids) - len(prompt)}

    def preference_context(self, result):
        """Call under lock with generation so prompt and reference describe the same weights."""
        if not self.ready or self.checkpoint_path is None:
            return None
        theme = result.get("theme")
        theme = theme if theme in ("game", "logic", "boast", "daily") else "daily"
        prefix = result.get("prompt", "")
        return {"prompt_ids": self.tokenizer.encode(prefix, theme=theme),
                "neutral_prompt_ids": self.tokenizer.encode("", theme=theme), "prefix": prefix,
                "theme": theme, "reference": {"checkpoint": str(self.checkpoint_path.relative_to(self.root / "models"))},
                "run_id": self.metadata["run_id"], "step": self.metadata["step"]}

    def learn_preference(self, job, store):
        from .preference_training import learn_runtime
        return learn_runtime(self, job, store)

    def _save(self, metadata):
        import numpy as np
        directory = self.model_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        generation = f"online-{metadata['step']:08d}-{uuid.uuid4().hex[:10]}"
        weights = directory / f"{generation}.npz"
        # Passing an opened stream prevents numpy from appending an extension.
        with weights.open("xb") as stream:
            np.savez(stream, **self.backend.export_weights())
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
        metadata = {**metadata, "weights": weights.name, "weight_sha256": self.backend.weight_digest(),
                    "updated_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(directory / f"{generation}.json", metadata)
        _atomic_json(self.model_dir / "current.json", {"checkpoint": f"checkpoints/{generation}.json"})
        self.metadata = metadata
        self.checkpoint_path = directory / f"{generation}.json"

    def learn(self, job, store):
        if not self.ready:
            raise RuntimeError(self.error or "模型未就绪")
        with self.lock:
            before = self.metadata["step"]
            result = {"run_id": self.metadata["run_id"], "step_before": before, "step_after": before,
                      "trained": False, "samples": 0, "loss": None, "weight_sha256": self.backend.weight_digest()}
            # Covers a crash after atomic checkpoint save but before SQLite ack.
            if job["id"] <= self.metadata.get("last_job_id", 0):
                if job["id"] == self.metadata.get("last_job_id") and self.metadata.get("last_job_result"):
                    return dict(self.metadata["last_job_result"])
                return {**result, "recovered_from_checkpoint": True}
            negatives = store.negative_hashes()
            positives = store.positive_examples(128)
            positives = [item for item in positives if text_digest(item["text"]) not in negatives]
            latest = next((item for item in positives if item["text"] == job["text"]), None)
            if job["action"] != "positive" or latest is None:
                return result
            # Prefer the annotation that triggered this job; replay prevents a
            # single recent sentence from replacing the whole tiny language.
            rng = random.Random(job["id"])
            seeds = [item for item in self.replay if text_digest(item["text"]) not in negatives]
            examples = [latest] * 4 + rng.sample(positives, min(4, len(positives))) + rng.sample(seeds, min(8, len(seeds)))
            encoded = []
            for item in examples:
                theme = item["theme"] if item["theme"] in ("game", "logic", "boast", "daily") else None
                encoded.extend(self.tokenizer.encode(item["text"], theme=theme, eos=True))
            # Use complete context windows, with EOS/BOS separating sentences.
            context = self.config.context
            windows = max(1, math.ceil((len(encoded) - 1) / context))
            needed = windows * context + 1
            encoded = (encoded * math.ceil(needed / len(encoded)))[:needed]
            chunks = [encoded[offset:offset + context + 1] for offset in range(0, windows * context, context)]
            x, y = [row[:-1] for row in chunks], [row[1:] for row in chunks]
            previous_weights = self.backend.export_weights()
            try:
                loss = self.backend.train_step(x, y, learning_rate=0.003, pad_id=self.tokenizer.pad_id)
                if not math.isfinite(loss):
                    raise RuntimeError("训练 loss 不是有限值，保留上次权重")
                if self.backend.weight_digest() == result["weight_sha256"]:
                    raise RuntimeError("梯度步骤没有改变真实权重")
                completed = {**result, "trained": True, "samples": len(examples), "loss": loss,
                             "step_after": before + 1, "weight_sha256": self.backend.weight_digest()}
                metadata = {**self.metadata, "step": before + 1, "last_job_id": job["id"],
                            "last_online_loss": loss, "last_job_result": completed}
                self._save(metadata)
            except Exception:
                self.backend = BACKEND_REGISTRY[self.backend_name](self.config, previous_weights)
                raise
            return completed


class BackgroundLearner:
    def __init__(self, runtime, store):
        self.runtime, self.store = runtime, store
        self.wake = threading.Event()
        self.stop = threading.Event()
        self.thread = None
        self.controls = queue.Queue()
        self._prefer_corrections = True
        # One writer across processes/backends, including differently bound ports.
        self._lease = store.path.with_suffix(".learner.lock").open("a")
        self.thread = threading.Thread(target=self._run, name="garbage-1-learner", daemon=True)
        self.thread.start()

    def notify(self):
        self.wake.set()

    def call_exclusive(self, callback, timeout=30):
        """Execute on the actual writer thread, under its cross-process lease."""
        import time
        request = {"fn": callback, "started": threading.Event(), "done": threading.Event(),
                   "cancel": threading.Event(), "guard": threading.Lock(), "result": None, "error": None}
        self.controls.put(request)
        self.notify()
        deadline = time.monotonic() + timeout
        while not request["done"].wait(.1):
            if self.stop.is_set():
                request["cancel"].set()
                raise RuntimeError("训练 worker 正在关闭，同步等待重试")
            if time.monotonic() > deadline:
                with request["guard"]:
                    if not request["started"].is_set():
                        request["cancel"].set()
                        raise RuntimeError("训练写入锁被其他进程占用，请稍后同步")
        if request["error"] is not None:
            raise request["error"]
        return request["result"]

    def _run(self):
        try:
            while not self.stop.is_set():
                try:
                    fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.store.recover_jobs()
                    self.store.recover_preferences()
                    # A previous process may have updated the pointer while
                    # this process waited for the single-writer lease.
                    replacement = self.runtime.reload()
                    if not replacement.ready:
                        with self.runtime.lock:
                            self.runtime.backend = None
                            self.runtime.error = replacement.error
                        self.stop.wait(1)
                        continue
                    if replacement.ready:
                        with self.runtime.lock:
                            self.runtime.backend = replacement.backend
                            self.runtime.metadata = replacement.metadata
                            self.runtime.config = replacement.config
                            self.runtime.tokenizer = replacement.tokenizer
                            self.runtime.prefixes = replacement.prefixes
                            self.runtime.replay = replacement.replay
                            self.runtime.error = None
                            self.runtime.checkpoint_path = replacement.checkpoint_path
                            if hasattr(replacement, "base_sha"):
                                self.runtime.base_sha = replacement.base_sha
                    self._consume()
                    return
                except BlockingIOError:
                    self.stop.wait(1)
                except Exception:
                    LOGGER.exception("Worker startup failed; queue and annotations preserved")
                    self.stop.wait(1)
        finally:
            # Keep the OS lease until the actual writer has stopped, even if
            # a caller's bounded close() wait has already elapsed.
            self._lease.close()

    def _consume(self):
        while not self.stop.is_set():
            try:
                request = self.controls.get_nowait()
            except queue.Empty:
                request = None
            if request is not None:
                with request["guard"]:
                    canceled = request["cancel"].is_set()
                    if not canceled:
                        request["started"].set()
                try:
                    if not canceled:
                        request["result"] = request["fn"]()
                except Exception as error:
                    request["error"] = error
                finally:
                    request["done"].set()
                continue
            if not self.runtime.ready:
                self.wake.wait(1)
                self.wake.clear()
                continue
            try:
                preference = False
                if self._prefer_corrections:
                    job = self.store.claim_preference()
                    preference = job is not None
                    if job is None:
                        job = self.store.claim_job()
                else:
                    job = self.store.claim_job()
                    if job is None:
                        job = self.store.claim_preference()
                        preference = job is not None
            except Exception:
                LOGGER.exception("Queue read failed; retrying without losing annotations")
                self.stop.wait(1)
                continue
            if job is None:
                self.wake.wait(1)
                self.wake.clear()
                continue
            self._prefer_corrections = not preference
            try:
                result = self.runtime.learn_preference(job, self.store) if preference else self.runtime.learn(job, self.store)
                error = None
            except Exception as failure:
                LOGGER.exception("Background learning failed; annotations are preserved")
                status = self.runtime.status()
                result = {"run_id": status["run_id"], "step_before": status["step"],
                          "step_after": status["step"], "trained": False, "samples": 0}
                error = str(failure)
            # SQLite acknowledgement failure is not a failed gradient update.
            # Retain the original result and retry; restart uses last_job_result.
            while not self.stop.is_set():
                try:
                    if preference:
                        self.store.finish_preference(job["id"], result, error=error)
                    else:
                        self.store.finish_job(job["id"], result, error=error)
                    break
                except Exception:
                    LOGGER.exception("Queue acknowledgement failed; keeping original training result")
                    self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
