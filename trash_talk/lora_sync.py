"""Durable one-click re-adaptation of style LoRAs to an immutable latest base."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import shutil
import threading
import uuid

import numpy as np

from .feedback import text_digest
from .lora import STYLE_NAMES, make_backend, training_batch
from .model_runtime import ModelRuntime, _atomic_json, _relative_file, _sync_directory

LOGGER = logging.getLogger(__name__)
ACTIVE = ("queued", "running")


def metadata_digest(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validation_loss(backend, tokenizer, context, texts):
    x, y = training_batch(tokenizer, context, texts)
    raw = backend.logits(x)
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=np.float64)
    targets = np.asarray(y)
    peak = scores.max(axis=-1)
    normalizer = peak + np.log(np.exp(scores - peak[..., None]).sum(axis=-1))
    selected = np.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
    mask = targets != tokenizer.pad_id
    result = float((normalizer - selected)[mask].mean())
    if not math.isfinite(result):
        raise RuntimeError("验证 loss 非有限值，保留旧版本")
    return result


class LoRASyncManager:
    def __init__(self, hub, steps=200, batch_size=8):
        if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("同步步数与批量必须为正整数")
        self.hub, self.root = hub, Path(hub.root)
        self.steps, self.batch_size = steps, batch_size
        self.path = self.root / "data/lora/sync-state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.closed = False
        old = self._read()
        if old is not None and old["status"] in ACTIVE:
            # Resume the exact target snapshot selected before the restart.
            with self.lock:
                lease = self._lease()
                if lease is not None:
                    self._launch(old, lease)

    def _read(self):
        if not self.path.exists():
            return None
        result = json.loads(self.path.read_text())
        if result.get("format") != "garbage-1-lora-sync-v1":
            raise ValueError("同步状态格式无效，保留原文件")
        return result

    def _latest(self):
        parent = self.root / "models/garbage-1"
        path = _relative_file(parent, json.loads((parent / "current.json").read_text())["checkpoint"])
        metadata = json.loads(path.read_text())
        if metadata.get("format") != "garbage-1-checkpoint-v1":
            raise ValueError("最新底座格式不兼容")
        sha = metadata.get("weight_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("最新底座缺少有效权重摘要")
        return path, metadata

    def _lease(self):
        lease = self.path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            return None
        return lease

    def _launch(self, job, lease):
        self.thread = threading.Thread(target=self._run, args=(job, lease), name="lora-base-sync", daemon=True)
        self.thread.start()

    def start(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("同步服务正在关闭")
            old = self._read()
            if self.thread is not None and self.thread.is_alive():
                return old
            lease = self._lease()
            if lease is None:
                if old is None:
                    raise RuntimeError("其他进程正在同步，请稍后重试")
                return old
            try:
                # An abandoned active job is resumed rather than overwritten.
                if old is not None and old["status"] in ACTIVE:
                    self._launch(old, lease)
                    return old
                path, metadata = self._latest()
                job = {"format": "garbage-1-lora-sync-v1", "id": uuid.uuid4().hex,
                       "status": "queued", "target_checkpoint": str(path.relative_to(self.root / "models/garbage-1")),
                       "target_step": metadata["step"], "target_sha256": metadata["weight_sha256"],
                       "target_metadata_sha256": metadata_digest(metadata),
                       "created_at": datetime.now(timezone.utc).isoformat(), "completed": 0, "total": len(STYLE_NAMES),
                       "results": [{"style": s, "name": name, "status": "pending"} for s, name in STYLE_NAMES.items()],
                       "message": "已排队，正在准备点击时的最新底座"}
                _atomic_json(self.path, job)
                self._launch(job, lease)
                return json.loads(json.dumps(job))
            except Exception:
                lease.close()
                raise

    def status(self):
        with self.lock:
            job = self._read()
            try:
                _, metadata = self._latest()
                latest = metadata["step"]
            except (OSError, KeyError, ValueError):
                latest = None
            return {"job": job, "latest_base_step": latest}

    def _write(self, job):
        with self.lock:
            _atomic_json(self.path, job)

    def _pin(self, job):
        parent = self.root / "models/garbage-1"
        source = _relative_file(parent, job["target_checkpoint"])
        metadata = json.loads(source.read_text())
        if metadata["weight_sha256"] != job["target_sha256"]:
            raise ValueError("目标底座摘要已改变，拒绝同步")
        if job.get("target_metadata_sha256") and metadata_digest(metadata) != job["target_metadata_sha256"]:
            raise ValueError("目标底座元数据已改变，拒绝同步")
        ref = "bases/" + job["target_sha256"] + "-" + metadata_digest({"config": metadata["config"], "tokenizer": metadata["tokenizer"]})[:16]
        destination = parent / "lora" / ref
        if not (destination / "current.json").exists():
            destination.mkdir(parents=True, exist_ok=True)
            weights = _relative_file(source.parent, metadata["weights"])
            shutil.copyfile(weights, destination / "base.npz")
            with (destination / "base.npz").open("rb") as stream:
                import os
                os.fsync(stream.fileno())
            _sync_directory(destination)
            _atomic_json(destination / "base.json", {**metadata, "weights": "base.npz", "pinned_from": job["target_checkpoint"]})
            _atomic_json(destination / "current.json", {"checkpoint": "base.json"})
        target = ModelRuntime(self.root, backend=self.hub.backend, model_dir=destination)
        if not target.ready or target.backend.weight_digest() != job["target_sha256"]:
            raise RuntimeError(target.error or "目标底座校验失败")
        return target, ref

    def _adapt(self, runtime, store, target, ref, job, item):
        """Called on that style's lease-holding writer; inference keeps old weights."""
        with runtime.lock:
            metadata = dict(runtime.metadata)
            old_sha = runtime.base_sha
            if runtime.config.to_dict() != target.config.to_dict() or runtime.tokenizer.to_dict() != target.tokenizer.to_dict():
                raise ValueError("底座架构或词表映射改变，需要另行转换；旧角色保持可用")
            if old_sha == job["target_sha256"]:
                return {"status": "synced" if metadata.get("migration", {}).get("job_id") == job["id"] else "skipped",
                        "message": "已使用目标底座", "base_step": metadata["base_step"]}
            arrays = runtime.backend.export_weights()
            old_backend = runtime.backend
        negatives = store.negative_hashes()
        validation = [t for t in runtime.style_data["validation"] if text_digest(t) not in negatives]
        if not validation:
            raise ValueError("没有可用验证语料，保留旧角色")
        # Do not train on the held-out gate, even if a user fed that exact sentence.
        heldout = set(runtime.style_data["validation"])
        seeds = [t for t in runtime.style_data["train"] if text_digest(t) not in negatives]
        positives = [p["text"] for p in store.positive_examples(128) if p["text"] not in heldout and text_digest(p["text"]) not in negatives]
        train = seeds + positives * 4
        if not train:
            raise ValueError("当前角色没有可用训练语料")
        item.update(message="正在使用角色语料重新适配", current_step=0, training_steps=self.steps)
        self._write(job)
        old_loss = validation_loss(old_backend, runtime.tokenizer, runtime.config.context, validation)
        base_weights = target.backend.export_weights()
        candidate = make_backend(target.config, base_weights, arrays, self.hub.backend, metadata["rank"], metadata["alpha"])
        before = validation_loss(candidate, target.tokenizer, target.config.context, validation)
        best_loss, best, best_step = before, arrays, 0
        rng = random.Random(f"{job['id']}:{runtime.style}")
        interval = min(20, self.steps)
        for step in range(1, self.steps + 1):
            if self.stop.is_set():
                raise InterruptedError("服务关闭，同步将在重启后继续")
            texts = rng.sample(train, min(self.batch_size, len(train)))
            x, y = training_batch(target.tokenizer, target.config.context, texts)
            loss = candidate.train_step(x, y, learning_rate=.0005, pad_id=target.tokenizer.pad_id, optimizer="adam")
            if not math.isfinite(loss):
                raise RuntimeError("迁移训练 loss 非有限值，保留旧版本")
            if step % interval == 0 or step == self.steps:
                score = validation_loss(candidate, target.tokenizer, target.config.context, validation)
                if score < best_loss:
                    best_loss, best, best_step = score, candidate.export_weights(), step
                item.update(current_step=step, training_steps=self.steps)
                self._write(job)
        if best_loss > old_loss + max(.05, old_loss * .05):
            raise RuntimeError(f"迁移验证退化：旧 {old_loss:.4f}，新 {best_loss:.4f}；保留旧角色")
        if candidate.base_digest() != job["target_sha256"]:
            raise RuntimeError("冻结底座检查失败，拒绝发布")
        candidate = make_backend(target.config, base_weights, best, self.hub.backend, metadata["rank"], metadata["alpha"])
        migration = {"job_id": job["id"], "from_base_sha256": old_sha, "from_base_step": metadata["base_step"],
                     "to_base_sha256": job["target_sha256"], "to_base_step": job["target_step"],
                     "old_validation_loss": old_loss, "rebased_validation_loss": before,
                     "new_validation_loss": best_loss, "attempted_steps": self.steps, "retained_step": best_step,
                     "train_sentences": len(train), "positive_sentences": len(positives), "validation_sentences": len(validation),
                     "validation_tolerance": max(.05, old_loss * .05)}
        updated = {**metadata, "base_ref": ref, "base_sha256": job["target_sha256"], "base_step": job["target_step"],
                   "config": target.config.to_dict(), "tokenizer": target.tokenizer.to_dict(),
                   "run_id": f"lora-{runtime.style}-{uuid.uuid4().hex[:12]}", "step": metadata["step"] + best_step,
                   "migration": migration}
        # The only interval that blocks inference is the small publication window.
        with runtime.lock:
            previous = (runtime.backend, runtime.config, runtime.tokenizer, runtime.base_sha, runtime.metadata)
            try:
                runtime.backend, runtime.config, runtime.tokenizer, runtime.base_sha = candidate, target.config, target.tokenizer, job["target_sha256"]
                runtime._save(updated)
            except Exception:
                runtime.backend, runtime.config, runtime.tokenizer, runtime.base_sha, runtime.metadata = previous
                raise
        return {"status": "synced", "message": "重新适配并通过验证，已切换", "base_step": job["target_step"], "validation": migration}

    def _run(self, job, lease):
        try:
            job.update(status="running", message="正在冻结目标底座并逐个适配角色")
            self._write(job)
            target, ref = self._pin(job)
            for item in job["results"]:
                if self.stop.is_set():
                    raise InterruptedError("等待重启后继续")
                if item["status"] in ("synced", "skipped"):
                    continue
                style = item["style"]
                if not (self.root / "models/garbage-1/lora" / style / "current.json").exists():
                    item.update(status="skipped", message="尚无适配器，跳过")
                else:
                    item.update(status="running", message="等待角色训练写入锁")
                    self._write(job)
                    try:
                        runtime, store, worker = self.hub.get(style)
                        result = worker.call_exclusive(lambda: self._adapt(runtime, store, target, ref, job, item))
                        item.update(result)
                    except InterruptedError:
                        raise
                    except Exception as error:
                        LOGGER.exception("LoRA sync failed for %s", style)
                        item.update(status="failed", message=str(error))
                job["completed"] = sum(i["status"] in ("synced", "skipped", "failed") for i in job["results"])
                job["message"] = f"已处理 {job['completed']}/{job['total']} 个角色"
                self._write(job)
            failures = sum(i["status"] == "failed" for i in job["results"])
            job["status"] = "failed" if failures == job["total"] else "partial" if failures else "completed"
            job["message"] = f"同步结束：{sum(i['status'] == 'synced' for i in job['results'])} 个已同步，{sum(i['status'] == 'skipped' for i in job['results'])} 个跳过，{failures} 个失败保留旧版"
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            self._write(job)
        except InterruptedError:
            # Keep active state for recovery; the role pointer is old or a fully committed new version.
            job.update(status="queued", message="服务停止，重启后继续同步")
            self._write(job)
        except Exception as error:
            LOGGER.exception("LoRA base sync failed")
            job.update(status="failed", message=str(error))
            self._write(job)
        finally:
            lease.close()

    def close(self):
        with self.lock:
            self.closed = True
            self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
