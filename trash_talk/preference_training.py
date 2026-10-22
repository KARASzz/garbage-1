"""Online DPO + chosen/replay SFT, using immutable generation checkpoints as references.

DPO follows https://github.com/eric-mitchell/direct-preference-optimization
Prompt and padding targets are masked. Long completions use explicit context
windows rather than silently discarding the tail of a human correction.
"""
from __future__ import annotations

import json
import math
import random

import numpy as np

from .model_runtime import BACKEND_REGISTRY, ModelRuntime, _relative_file


def make_batch(tokenizer, context, prompt_ids, chosen, rejected, anchors=()):
    if not prompt_ids or any(type(i) is not int or not 0 <= i < tokenizer.vocab_size for i in prompt_ids):
        raise ValueError("偏好输入 token 无效")
    completions = [tokenizer.encode(t, bos=False, eos=True) for t in (chosen, rejected)]
    if completions[0] == completions[1]:
        raise ValueError("两个答案编码后相同，无法构成偏好对")
    sequences = [(list(prompt_ids) + ids, len(prompt_ids), group) for group, ids in enumerate(completions)]
    for item in anchors:
        theme = item.get("theme") if item.get("theme") in ("daily", "logic", "game", "boast") else None
        prefix = tokenizer.encode("", theme=theme)
        sequences.append((prefix + tokenizer.encode(item["text"], bos=False, eos=True), len(prefix), 2))
    rows = []
    for ids, prefix_length, group in sequences:
        for offset in range(0, len(ids) - 1, context):
            x, y = ids[offset:offset + context], ids[offset + 1:offset + context + 1]
            x = x[:len(y)]
            mask = [float(offset + i + 1 >= prefix_length) for i in range(len(y))]
            rows.append((x, y, mask, group))
    width = max(len(row[0]) for row in rows)
    x, y, masks, groups = [], [], [], []
    for inputs, targets, mask, group in rows:
        pad = width - len(inputs)
        x.append(inputs + [tokenizer.pad_id] * pad)
        y.append(targets + [tokenizer.pad_id] * pad)
        masks.append(mask + [0.] * pad)
        groups.append(group)
    return {"x": x, "y": y, "mask": masks, "groups": groups}


def log_probabilities(backend, batch):
    raw = backend.logits(batch["x"])
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=np.float64)
    targets = np.asarray(batch["y"])
    peak = scores.max(axis=-1)
    selected = np.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
    token_logps = selected - peak - np.log(np.exp(scores - peak[..., None]).sum(axis=-1))
    mask, groups = np.asarray(batch["mask"]), np.asarray(batch["groups"])
    sums, counts = [], []
    for group in range(3):
        weights = mask * (groups == group)[:, None]
        sums.append(float((token_logps * weights).sum()))
        counts.append(float(weights.sum()))
    if not all(math.isfinite(v) for v in sums):
        raise RuntimeError("偏好计算出现非有限概率")
    return sums, counts


def objective(logps, counts, reference, beta=.1, chosen_weight=.5, anchor_weight=.1):
    ratio = (logps[0] - logps[1]) - (reference[0] - reference[1])
    dpo = float(np.logaddexp(0., -beta * ratio))
    chosen_sft = -logps[0] / max(counts[0], 1.)
    anchor_sft = -logps[2] / max(counts[2], 1.)
    return dpo + chosen_weight * chosen_sft + anchor_weight * anchor_sft


def preference_step(backend, batch, reference, *, learning_rate=.001, beta=.1, chosen_weight=.5, anchor_weight=.1):
    """One clipped step; a line search rejects nonfinite or increasing loss."""
    if not all(math.isfinite(v) and v > 0 for v in (learning_rate, beta)):
        raise ValueError("偏好学习率和 beta 无效")
    before, counts = log_probabilities(backend, batch)
    before_loss = objective(before, counts, reference, beta, chosen_weight, anchor_weight)
    mlx = type(backend).__name__ in ("MLXBackend", "MLXLoRA")
    if mlx:
        import mlx.core as mx
        lora = hasattr(backend, "adapter")
        params = backend.adapter if lora else backend.parameters
        inputs, targets = mx.array(batch["x"], dtype=mx.int32), mx.array(batch["y"], dtype=mx.int32)
        mask, groups = mx.array(batch["mask"], dtype=mx.float32), mx.array(batch["groups"])
        def loss_fn(values):
            scores = backend.base._forward(backend._effective(values), inputs) if lora else backend._forward(values, inputs)
            selected = mx.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
            logps = selected - mx.logsumexp(scores, axis=-1)
            sums = [mx.sum(logps * mask * (groups == i)[:, None]) for i in range(3)]
            ratio = (sums[0] - sums[1]) - float(reference[0] - reference[1])
            dpo = mx.logsumexp(mx.stack([mx.array(0., dtype=mx.float32), -beta * ratio]))
            return dpo - chosen_weight * sums[0] / max(counts[0], 1.) - anchor_weight * sums[2] / max(counts[2], 1.)
        loss, grads = mx.value_and_grad(loss_fn)(params)
        norm = mx.sqrt(sum(mx.sum(g * g) for g in grads.values()))
        scale = mx.minimum(1., 1. / (norm + 1e-12))
        mx.eval(loss, grads, norm)
        if not math.isfinite(float(loss.item())) or not math.isfinite(float(norm.item())):
            raise RuntimeError("偏好梯度非有限值")
        original = dict(params)
        def apply(rate):
            updated = {k: value - rate * scale * grads[k] for k, value in original.items()}
            mx.eval(updated)
            if lora:
                backend.adapter = updated
            else:
                backend._parameters = updated
        def restore():
            if lora:
                backend.adapter = original
            else:
                backend._parameters = original
    else:
        import torch
        lora = hasattr(backend, "adapter")
        params = backend.adapter if lora else dict(backend.model.named_parameters())
        device = backend.base.device if lora else backend.device
        for value in params.values():
            value.grad = None
        inputs, targets = torch.tensor(batch["x"], device=device), torch.tensor(batch["y"], device=device)
        mask, groups = torch.tensor(batch["mask"], device=device), torch.tensor(batch["groups"], device=device)
        scores = backend._logits(inputs) if lora else backend.model(inputs)
        logps = scores.gather(-1, targets[..., None])[..., 0] - scores.logsumexp(-1)
        sums = [(logps * mask * (groups == i)[:, None]).sum() for i in range(3)]
        ratio = sums[0] - sums[1] - float(reference[0] - reference[1])
        loss = torch.nn.functional.softplus(-beta * ratio) - chosen_weight * sums[0] / max(counts[0], 1.) - anchor_weight * sums[2] / max(counts[2], 1.)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(norm)):
            raise RuntimeError("偏好梯度非有限值")
        original = {k: v.detach().clone() for k, v in params.items()}
        def apply(rate):
            with torch.no_grad():
                for k, value in params.items():
                    value.copy_(original[k] - rate * value.grad)
        def restore():
            with torch.no_grad():
                for k, value in params.items():
                    value.copy_(original[k])
    try:
        for attempt in range(6):
            rate = learning_rate / (2 ** attempt)
            apply(rate)
            after, _ = log_probabilities(backend, batch)
            after_loss = objective(after, counts, reference, beta, chosen_weight, anchor_weight)
            if math.isfinite(after_loss) and after_loss <= before_loss + 1e-6:
                return {"loss": after_loss, "loss_before": before_loss, "beta": beta,
                        "chosen_sft_weight": chosen_weight, "anchor_sft_weight": anchor_weight,
                        "reference_chosen_logp": reference[0], "reference_rejected_logp": reference[1],
                        "chosen_logp_before": before[0], "rejected_logp_before": before[1],
                        "chosen_logp_after": after[0], "rejected_logp_after": after[1],
                        "preference_margin_before": before[0] - before[1], "preference_margin_after": after[0] - after[1],
                        "chosen_token_count": counts[0], "rejected_token_count": counts[1], "learning_rate": rate}
        raise RuntimeError("偏好更新未改善目标，保留上次权重")
    except Exception:
        restore()
        raise


def load_reference(runtime, context):
    from .garbage_model import ModelConfig, CharacterTokenizer
    from .lora import make_backend
    path = _relative_file(runtime.root / "models", context["reference"]["checkpoint"])
    metadata = json.loads(path.read_text())
    config = ModelConfig.from_dict(metadata["config"])
    tokenizer = CharacterTokenizer.from_dict(metadata["tokenizer"])
    if config.to_dict() != runtime.config.to_dict() or tokenizer.to_dict() != runtime.tokenizer.to_dict():
        raise ValueError("这条改写的网络或词表版本不兼容，已保留偏好数据")
    with np.load(_relative_file(path.parent, metadata["weights"]), allow_pickle=False) as data:
        weights = {k: data[k] for k in data.files}
    if metadata["format"] == "garbage-1-lora-v1":
        pinned = _relative_file(runtime.root / "models/garbage-1/lora", metadata.get("base_ref", "base"))
        base = ModelRuntime(runtime.root, backend=runtime.backend_name, model_dir=pinned)
        if not base.ready or base.backend.weight_digest() != metadata["base_sha256"]:
            raise ValueError("偏好参考底座校验失败")
        backend = make_backend(config, base.backend.export_weights(), weights, runtime.backend_name, metadata["rank"], metadata["alpha"])
    elif metadata["format"] == "garbage-1-checkpoint-v1":
        backend = BACKEND_REGISTRY[runtime.backend_name](config, weights)
    else:
        raise ValueError("偏好参考模型格式不支持")
    if backend.weight_digest() != metadata["weight_sha256"]:
        raise ValueError("偏好参考权重摘要不匹配")
    return backend


def learn_runtime(runtime, job, store):
    from .feedback import text_digest
    from .lora import make_backend
    with runtime.lock:
        if not runtime.ready:
            raise RuntimeError("偏好训练模型未就绪")
        if job["id"] <= runtime.metadata.get("last_preference_id", 0):
            return dict(runtime.metadata["last_preference_result"]) if job["id"] == runtime.metadata.get("last_preference_id") else {"trained": False, "recovered_from_checkpoint": True}
        context = job["context"]
        reference = load_reference(runtime, context)
        negatives = store.negative_hashes()
        seeds = [p for p in runtime.replay if text_digest(p["text"]) not in negatives]
        anchors = random.Random(job["id"]).sample(seeds, min(4, len(seeds)))
        batch = make_batch(runtime.tokenizer, runtime.config.context, job["prompt_ids"], job["chosen_completion"], job["rejected_completion"], anchors)
        reference_logps, _ = log_probabilities(reference, batch)
        before, previous = runtime.metadata["step"], runtime.backend.export_weights()
        before_digest = runtime.backend.weight_digest()
        lora = hasattr(runtime, "base_sha")
        try:
            metrics = preference_step(runtime.backend, batch, reference_logps, learning_rate=.001 if lora else .0005)
            if runtime.backend.weight_digest() == before_digest:
                raise RuntimeError("偏好步骤未改变真实权重")
            if lora and runtime.backend.base_digest() != runtime.base_sha:
                raise RuntimeError("偏好步骤改变了冻结底座")
            result = {**metrics, "algorithm": "dpo+chosen_sft+replay_sft", "trained": True,
                      "run_id": runtime.metadata["run_id"], "step_before": before, "step_after": before + 1,
                      "samples": 1, "preference_id": job["id"], "weight_sha256": runtime.backend.weight_digest(),
                      "reference": context["reference"], "prompt_mode": job["prompt_mode"]}
            runtime._save({**runtime.metadata, "step": before + 1, "last_preference_id": job["id"], "last_preference_result": result})
            return result
        except Exception:
            if lora:
                runtime.backend = make_backend(runtime.config, runtime.backend.base.export_weights(), previous,
                                               runtime.backend_name, runtime.metadata["rank"], runtime.metadata["alpha"])
            else:
                runtime.backend = BACKEND_REGISTRY[runtime.backend_name](runtime.config, previous)
            raise
