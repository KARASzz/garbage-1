"""Independent FP32 LoRA adapters, pinned bases and isolated annotation queues."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import threading
import uuid

import numpy as np

from .feedback import FeedbackStore, text_digest
from .model_runtime import ModelRuntime, BackgroundLearner, _atomic_json, _relative_file, _sync_directory

STYLE_NAMES = dict(zip(
    ("mouthy", "mad", "sassy", "anime", "femboy", "office", "soldier", "alien", "hostess"),
    ("嘴臭", "疯子", "娘炮", "二次元", "伪娘", "社畜", "兵王", "外星人", "陪酒女"), strict=True))


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def initial_adapter(config, rank=4, seed=42, include_ffn=False):
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank 必须为正整数")
    rng = np.random.default_rng(seed)
    result = {}
    for i in range(2):
        for name, out in (("qkv", 3 * config.d_model), ("attn_out", config.d_model)):
            prefix = f"blocks.{i}.{name}"
            result[prefix + ".A"] = (rng.normal(size=(rank, config.d_model)) * .02).astype(np.float32)
            result[prefix + ".B"] = np.zeros((out, rank), dtype=np.float32)
        if include_ffn:
            hidden = config.ffn_dims[i]
            for name, width_in, width_out in (("up", config.d_model, hidden), ("down", hidden, config.d_model)):
                prefix = f"blocks.{i}.ffn.{name}"
                result[prefix + ".A"] = (rng.normal(size=(rank, width_in)) * .02).astype(np.float32)
                result[prefix + ".B"] = np.zeros((width_out, rank), dtype=np.float32)
    return result


def validate_adapter(config, arrays, rank):
    expected = initial_adapter(config, rank, include_ffn=any(".ffn." in k for k in arrays))
    if set(arrays) != set(expected):
        raise ValueError("LoRA 参数名称不匹配")
    for k, value in arrays.items():
        if value.shape != expected[k].shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"LoRA 参数格式无效: {k}")


def make_backend(config, weights, arrays, backend="mlx", rank=4, alpha=8):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha 必须为正有限数")
    validate_adapter(config, arrays, rank)
    if backend == "mlx":
        return MLXLoRA(config, weights, arrays, alpha / rank)
    if backend in ("cuda", "torch-cpu"):
        return TorchLoRA(config, weights, arrays, alpha / rank, backend)
    raise ValueError("LoRA 后端未知")


class MLXLoRA:
    def __init__(self, config, weights, arrays, scale):
        import mlx.core as mx
        from .mlx_backend import MLXBackend
        self.base = MLXBackend(config, weights)
        self.config, self.scale = config, scale
        self.adapter = {k: mx.array(v, dtype=mx.float32) for k, v in arrays.items()}
        self.m, self.v, self.adam_step = {}, {}, 0
        mx.eval(self.adapter)

    def _effective(self, adapter):
        result = dict(self.base.parameters)
        for name in adapter:
            if name.endswith(".A"):
                prefix = name[:-2]
                result[prefix + ".weight"] = result[prefix + ".weight"] + self.scale * (adapter[prefix + ".B"] @ adapter[name])
        return result

    def logits(self, ids):
        import mlx.core as mx
        result = self.base._forward(self._effective(self.adapter), mx.array(ids, dtype=mx.int32))
        mx.eval(result)
        return result

    def generate(self, prompt_ids, max_new_tokens, eos_id):
        result = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = int(self.logits([result[-self.config.context:]])[0, -1].argmax().item())
            result.append(next_id)
            if next_id == eos_id:
                break
        return result

    def train_step(self, x, y, learning_rate=.003, pad_id=0, *, optimizer="sgd"):
        import mlx.core as mx
        inputs, targets = mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)
        def loss_fn(adapter):
            scores = self.base._forward(self._effective(adapter), inputs)
            selected = mx.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
            mask = (targets != pad_id).astype(mx.float32)
            return mx.sum((mx.logsumexp(scores, axis=-1) - selected) * mask) / mx.maximum(mx.sum(mask), 1)
        loss, grads = mx.value_and_grad(loss_fn)(self.adapter)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
        clip = mx.minimum(1., 1. / (norm + 1e-12))
        if optimizer == "adam":
            self.adam_step += 1
            for k, g in grads.items():
                g = g * clip
                self.m[k] = .9 * self.m.get(k, mx.zeros_like(g)) + .1 * g
                self.v[k] = .999 * self.v.get(k, mx.zeros_like(g)) + .001 * g * g
                corrected_m = self.m[k] / (1 - .9 ** self.adam_step)
                corrected_v = self.v[k] / (1 - .999 ** self.adam_step)
                self.adapter[k] = self.adapter[k] - learning_rate * corrected_m / (mx.sqrt(corrected_v) + 1e-8)
        else:
            self.adapter = {k: value - learning_rate * clip * grads[k] for k, value in self.adapter.items()}
        mx.eval(loss, self.adapter, self.m, self.v)
        return float(loss.item())

    def export_weights(self):
        return {k: np.array(v, dtype=np.float32, copy=True) for k, v in self.adapter.items()}

    def weight_digest(self):
        return array_digest(self.export_weights())

    def base_digest(self):
        return self.base.weight_digest()


class TorchLoRA:
    """Same A/B file format; CUDA is explicit, with CPU as test reference."""
    def __init__(self, config, weights, arrays, scale, backend):
        import torch
        from .torch_backend import TorchBackend
        self.base = TorchBackend(config, weights, device="cuda" if backend == "cuda" else "cpu")
        self.config, self.scale = config, scale
        self.base.model.requires_grad_(False)
        self.adapter = {k: torch.tensor(v.copy(), device=self.base.device, requires_grad=True) for k, v in arrays.items()}
        self.optimizer = None

    def _logits(self, ids):
        from torch.func import functional_call
        params = dict(self.base.model.named_parameters())
        for name in self.adapter:
            if name.endswith(".A"):
                prefix = name[:-2]
                params[prefix + ".weight"] = params[prefix + ".weight"] + self.scale * (self.adapter[prefix + ".B"] @ self.adapter[name])
        return functional_call(self.base.model, params, (ids,))

    def logits(self, ids):
        import torch
        with torch.no_grad():
            return self._logits(torch.tensor(ids, dtype=torch.long, device=self.base.device))

    def generate(self, prompt_ids, max_new_tokens, eos_id):
        result = list(prompt_ids)
        for _ in range(max_new_tokens):
            next_id = int(self.logits([result[-self.config.context:]])[0, -1].argmax().item())
            result.append(next_id)
            if next_id == eos_id:
                break
        return result

    def train_step(self, x, y, learning_rate=.003, pad_id=0, *, optimizer="sgd"):
        import torch
        for value in self.adapter.values():
            value.grad = None
        ids = torch.tensor(x, dtype=torch.long, device=self.base.device)
        targets = torch.tensor(y, dtype=torch.long, device=self.base.device)
        scores = self._logits(ids)
        loss = torch.nn.functional.cross_entropy(scores.flatten(0, 1), targets.flatten(), ignore_index=pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.adapter.values()), 1.)
        if optimizer == "adam":
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(list(self.adapter.values()), lr=learning_rate)
            self.optimizer.step()
        else:
            with torch.no_grad():
                for value in self.adapter.values():
                    value.add_(value.grad, alpha=-learning_rate)
        return float(loss.detach())

    def export_weights(self):
        return {k: v.detach().cpu().numpy().copy() for k, v in self.adapter.items()}

    def weight_digest(self):
        return array_digest(self.export_weights())

    def base_digest(self):
        return self.base.weight_digest()


def read_catalog(root):
    path = Path(root) / "data/lora/styles.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    styles = data["styles"]
    if data["format"] != "garbage-1-lora-styles-v1" or {s["id"] for s in styles} != set(STYLE_NAMES):
        raise ValueError("风格语料清单无效")
    for style in styles:
        train, validation = style["train"], style["validation"]
        if not train or not validation or any(not isinstance(t, str) or not t.strip() for t in train + validation):
            raise ValueError("风格语料有空值")
        if len(set(train + validation)) != len(train + validation):
            raise ValueError("风格训练验证语料重复")
    return styles


class LoRARuntime(ModelRuntime):
    def __init__(self, root, style, *, backend="mlx", allow_untrained=False):
        if not isinstance(style, str) or style not in STYLE_NAMES:
            raise ValueError("未知 LoRA 风格")
        self.style = style
        self.style_data = next(item for item in read_catalog(root) if item["id"] == style)
        model_root = Path(root) / "models/garbage-1/lora"
        pointer = model_root / style / "current.json"
        saved = None
        if pointer.exists():
            saved_path = _relative_file(pointer.parent, json.loads(pointer.read_text())["checkpoint"])
            saved = json.loads(saved_path.read_text())
        base_ref = saved.get("base_ref", "base") if saved is not None else "base"
        super().__init__(root, backend=backend, model_dir=_relative_file(model_root, base_ref))
        if not self.ready:
            raise RuntimeError(self.error or "固定底座未准备")
        base_weights = self.backend.export_weights()
        self.base_sha = self.backend.weight_digest()
        self.model_dir = model_root / style
        pointer = self.model_dir / "current.json"
        if pointer.exists():
            metadata_path, metadata = saved_path, saved
            if metadata.get("format") != "garbage-1-lora-v1" or metadata.get("style") != style or metadata.get("base_sha256") != self.base_sha:
                raise ValueError("LoRA 与固定底座不匹配，保留文件")
            with np.load(_relative_file(metadata_path.parent, metadata["weights"]), allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            if array_digest(arrays) != metadata["weight_sha256"]:
                raise ValueError("LoRA 权重摘要不匹配")
            self.metadata = metadata
        else:
            if not allow_untrained:
                raise RuntimeError("该风格 LoRA 尚未训练")
            arrays = initial_adapter(self.config)
            self.metadata = {**self.metadata, "format": "garbage-1-lora-v1", "style": style,
                             "base_sha256": self.base_sha, "base_step": self.metadata["step"],
                             "step": 0, "last_job_id": 0, "last_job_result": None,
                             "run_id": f"lora-{style}-{uuid.uuid4().hex[:12]}", "rank": 4, "alpha": 8}
        self.backend = make_backend(self.config, base_weights, arrays, backend, self.metadata["rank"], self.metadata["alpha"])
        # Only train split is used for replay and inference prefixes.
        self.replay = [{"text": t, "theme": "daily"} for t in self.style_data["train"]]
        self.prefixes = [{"text": t[:4], "theme": "daily"} for t in self.style_data["train"]]

    def reload(self):
        return LoRARuntime(self.root, self.style, backend=self.backend_name)

    def status(self):
        result = super().status()
        result["adapter"] = {"id": self.style, "name": STYLE_NAMES[self.style], "step": self.metadata["step"],
                             "parameters": sum(v.size for v in self.backend.export_weights().values()),
                             "rank": self.metadata["rank"], "base_step": self.metadata["base_step"],
                             "base_sha256": self.base_sha}
        return result

    def _save(self, metadata):
        directory = self.model_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        name = f"adapter-{metadata['step']:08d}-{uuid.uuid4().hex[:10]}"
        weights = directory / (name + ".npz")
        with weights.open("xb") as stream:
            np.savez(stream, **self.backend.export_weights())
            stream.flush()
            import os
            os.fsync(stream.fileno())
        _sync_directory(directory)
        metadata = {**metadata, "weights": weights.name, "weight_sha256": self.backend.weight_digest()}
        _atomic_json(directory / (name + ".json"), metadata)
        _atomic_json(self.model_dir / "current.json", {"checkpoint": f"checkpoints/{name}.json"})
        self.metadata = metadata

    def learn(self, job, store):
        with self.lock:
            before = self.metadata["step"]
            result = {"run_id": self.metadata["run_id"], "step_before": before, "step_after": before,
                      "trained": False, "samples": 0, "loss": None, "weight_sha256": self.backend.weight_digest()}
            if job["id"] <= self.metadata.get("last_job_id", 0):
                return dict(self.metadata.get("last_job_result") or result) if job["id"] == self.metadata.get("last_job_id") else result
            negatives = store.negative_hashes()
            positives = store.positive_examples(128)
            latest = next((p for p in positives if p["text"] == job["text"]), None)
            if job["action"] != "positive" or latest is None:
                return result
            rng = random.Random(job["id"])
            replay = [p for p in self.replay if text_digest(p["text"]) not in negatives]
            examples = [latest] * 4 + rng.sample(positives, min(4, len(positives))) + rng.sample(replay, min(8, len(replay)))
            x, y = training_batch(self.tokenizer, self.config.context, [p["text"] for p in examples])
            previous = self.backend.export_weights()
            try:
                loss = self.backend.train_step(x, y, learning_rate=.01, pad_id=self.tokenizer.pad_id)
                if not math.isfinite(loss) or self.backend.weight_digest() == result["weight_sha256"] or self.backend.base_digest() != self.base_sha:
                    raise RuntimeError("LoRA 更新检查失败，保留上次权重")
                completed = {**result, "trained": True, "samples": len(examples), "loss": loss,
                             "step_after": before + 1, "weight_sha256": self.backend.weight_digest()}
                self._save({**self.metadata, "step": before + 1, "last_job_id": job["id"], "last_job_result": completed})
                return completed
            except Exception:
                self.backend = make_backend(self.config, self.backend.base.export_weights(), previous,
                                            self.backend_name, self.metadata["rank"], self.metadata["alpha"])
                raise


def training_batch(tokenizer, context, texts):
    """Keep every byte and sentence ending; windows never silently drop long feeds."""
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text, theme="daily", eos=True))
    chunks = [ids[i:i + context + 1] for i in range(0, len(ids) - 1, context)]
    width = max(len(c) for c in chunks)
    chunks = [c + [tokenizer.pad_id] * (width - len(c)) for c in chunks]
    return [c[:-1] for c in chunks], [c[1:] for c in chunks]


class StyleHub:
    def __init__(self, root, backend, base_runtime=None):
        self.root, self.backend = Path(root), backend
        self.base_runtime = base_runtime
        self.entries = {}
        self.lock = threading.Lock()
        self._sync = None

    def status(self):
        base = self.base_runtime.status() if self.base_runtime is not None else {}
        styles = [{"id": "base", "name": "原版", "ready": base.get("ready", False), "step": base.get("step"), "parameters": base.get("parameters", 0)}]
        for style, name in STYLE_NAMES.items():
            pointer = self.root / "models/garbage-1/lora" / style / "current.json"
            entry = {"id": style, "name": name, "ready": False, "step": 0, "parameters": 0}
            try:
                metadata = json.loads(_relative_file(pointer.parent, json.loads(pointer.read_text())["checkpoint"]).read_text())
                checkpoint = _relative_file(pointer.parent, json.loads(pointer.read_text())["checkpoint"])
                with np.load(_relative_file(checkpoint.parent, metadata["weights"]), allow_pickle=False) as archive:
                    arrays = {k: archive[k] for k in archive.files}
                pinned = _relative_file(self.root / "models/garbage-1/lora", metadata.get("base_ref", "base"))
                base_meta = json.loads(_relative_file(pinned, json.loads((pinned / "current.json").read_text())["checkpoint"]).read_text())
                from .garbage_model import ModelConfig
                validate_adapter(ModelConfig.from_dict(base_meta["config"]), arrays, metadata["rank"])
                valid = (metadata["format"] == "garbage-1-lora-v1" and metadata["style"] == style
                         and metadata["base_sha256"] == base_meta["weight_sha256"]
                         and metadata["weight_sha256"] == array_digest(arrays))
                entry.update(ready=valid, step=metadata["step"], parameters=sum(v.size for v in arrays.values()),
                             base_step=metadata.get("base_step"), base_sha256=metadata.get("base_sha256"))
            except (OSError, KeyError, ValueError):
                pass
            styles.append(entry)
        return {"styles": styles, "default": "base"}

    def get(self, style):
        if not isinstance(style, str) or style not in STYLE_NAMES:
            raise ValueError("未知 LoRA 风格")
        with self.lock:
            if style not in self.entries:
                runtime = LoRARuntime(self.root, style, backend=self.backend)
                store = FeedbackStore(self.root / "data/lora" / (style + ".sqlite3"))
                learner = BackgroundLearner(runtime, store)
                self.entries[style] = (runtime, store, learner)
            return self.entries[style]

    def rate(self, sample_id, rating):
        if not isinstance(sample_id, str):
            raise ValueError("样本 ID 无效")
        import sqlite3
        for style in STYLE_NAMES:
            path = self.root / "data/lora" / (style + ".sqlite3")
            if not path.exists():
                continue
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                found = connection.execute("SELECT 1 FROM samples WHERE id=?", (sample_id,)).fetchone()
            if found:
                _, store, learner = self.get(style)
                result = store.rate(sample_id, rating)
                learner.notify()
                return {**result, "style": style}
        raise LookupError("样本不存在")

    def close(self):
        if self._sync is not None:
            self._sync.close()
        for _, _, learner in self.entries.values():
            learner.close()

    @property
    def sync(self):
        with self.lock:
            if self._sync is None:
                from .lora_sync import LoRASyncManager
                self._sync = LoRASyncManager(self)
            return self._sync

    def start_sync(self):
        return self.sync.start()

    def sync_status(self):
        return self.sync.status()
