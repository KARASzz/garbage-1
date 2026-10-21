#!/usr/bin/env python3
"""Train style adapters on a pinned snapshot; never updates the live base."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import random
import shutil
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trash_talk.lora import LoRARuntime, STYLE_NAMES, read_catalog, training_batch, make_backend, initial_adapter
from trash_talk.model_runtime import _atomic_json, _relative_file, ModelRuntime


def pin_base(root):
    parent = root / "models/garbage-1"
    target = parent / "lora/base"
    if (target / "current.json").exists():
        return
    pointer = json.loads((parent / "current.json").read_text())
    path = _relative_file(parent, pointer["checkpoint"])
    metadata = json.loads(path.read_text())
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_relative_file(path.parent, metadata["weights"]), target / "base.npz")
    _atomic_json(target / "base.json", {**metadata, "weights": "base.npz", "pinned_from": str(path.relative_to(parent))})
    _atomic_json(target / "current.json", {"checkpoint": "base.json"})


def evaluate(runtime, texts):
    x, y = training_batch(runtime.tokenizer, runtime.config.context, texts)
    raw = runtime.backend.logits(x)
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=np.float64)
    targets = np.array(y)
    peak = scores.max(axis=-1)
    logsum = peak + np.log(np.exp(scores - peak[..., None]).sum(axis=-1))
    selected = np.take_along_axis(scores, targets[..., None], axis=-1)[..., 0]
    mask = targets != runtime.tokenizer.pad_id
    return float((logsum - selected)[mask].mean())


def main():
    parser = argparse.ArgumentParser(description="为九种风格训练独立 LoRA，不修改现有底座")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--styles", nargs="+", choices=list(STYLE_NAMES), default=list(STYLE_NAMES))
    parser.add_argument("--backend", choices=("mlx", "cuda", "torch-cpu"), default="mlx")
    parser.add_argument("--retrain", action="store_true", help="保留旧快照，重新训练 rank8 注意力与前馈 LoRA")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps 和 batch-size 必须为正数")
    parent = ROOT / "models/garbage-1/lora"
    parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with (parent / ".prepare.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pin_base(ROOT)
        catalog = {s["id"]: s for s in read_catalog(ROOT)}
        for style in args.styles:
            if (parent / style / "current.json").exists() and not args.retrain:
                print(json.dumps({"style": style, "skipped": "已有适配器，保留现有训练"}, ensure_ascii=False), flush=True)
                continue
            locks = ROOT / "data/lora"
            locks.mkdir(parents=True, exist_ok=True)
            style_lease = (locks / (style + ".learner.lock")).open("a")
            fcntl.flock(style_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runtime = LoRARuntime(ROOT, style, backend=args.backend, allow_untrained=True)
            if args.retrain or runtime.metadata["step"] == 0:
                arrays = initial_adapter(runtime.config, rank=8, include_ffn=True)
                runtime.backend = make_backend(runtime.config, runtime.backend.base.export_weights(), arrays, args.backend, 8, 16)
                runtime.metadata.update(rank=8, alpha=16, step=0, last_job_id=0, last_job_result=None,
                                        run_id=f"lora-{style}-{uuid.uuid4().hex[:12]}")
            initial = evaluate(runtime, catalog[style]["validation"])
            best_loss, best_step, best = initial, 0, runtime.backend.export_weights()
            base_digest = runtime.backend.base_digest()
            rng = random.Random(20261021)
            base_weights = runtime.backend.base.export_weights()
            for step in range(1, args.steps + 1):
                texts = rng.sample(catalog[style]["train"], min(args.batch_size, len(catalog[style]["train"])))
                x, y = training_batch(runtime.tokenizer, runtime.config.context, texts)
                loss = runtime.backend.train_step(x, y, learning_rate=.002, pad_id=runtime.tokenizer.pad_id, optimizer="adam")
                if not np.isfinite(loss):
                    raise RuntimeError("训练出现非有限 loss")
                if step % 40 == 0 or step == args.steps:
                    validation = evaluate(runtime, catalog[style]["validation"])
                    if validation < best_loss:
                        best_loss, best_step, best = validation, step, runtime.backend.export_weights()
                    print(json.dumps({"style": style, "step": step, "train_loss": loss, "validation_loss": validation}, ensure_ascii=False), flush=True)
            if best_step == 0:
                raise RuntimeError(f"{style} 未改善验证 loss；不发布未训练适配器")
            if runtime.backend.base_digest() != base_digest:
                raise RuntimeError("底座权重被改变")
            runtime.backend = make_backend(runtime.config, base_weights, best, args.backend, runtime.metadata["rank"], runtime.metadata["alpha"])
            parameter_count = sum(v.size for v in best.values())
            metadata = {**runtime.metadata, "step": best_step,
                        "adapter_parameters": parameter_count,
                        "pretraining": {"optimizer": "adam", "learning_rate": .002, "attempted_steps": args.steps,
                                        "train_sentences": len(catalog[style]["train"]),
                                        "validation_sentences": len(catalog[style]["validation"]),
                                        "validation_before": initial, "validation_after": best_loss,
                                        "optimizer_state_saved": False, "online_optimizer": "sgd"}}
            runtime._save(metadata)
            # Same prompts with adapter enabled/disabled isolate adapter influence.
            samples = []
            for text in catalog[style]["validation"][:2]:
                prompt = runtime.tokenizer.encode(text[:4], theme="daily")
                base_ids = runtime.backend.base.generate(prompt, 40, runtime.tokenizer.eos_id)
                adapter_ids = runtime.backend.generate(prompt, 40, runtime.tokenizer.eos_id)
                samples.append({"prompt": text[:4], "base": runtime.tokenizer.decode(base_ids),
                                "adapter": runtime.tokenizer.decode(adapter_ids), "changed": base_ids != adapter_ids})
            report = {"style": style, "name": STYLE_NAMES[style], "parameters": parameter_count, "rank": runtime.metadata["rank"],
                      "base_frozen": runtime.backend.base_digest() == base_digest, **metadata["pretraining"], "samples": samples}
            reports.append(report)
            output = ROOT / "output/lora-report.json"
            if output.exists():
                old = json.loads(output.read_text()).get("styles", [])
            else:
                old = []
            combined = [r for r in old if r["style"] != style] + [report]
            _atomic_json(output, {"format": "garbage-1-lora-training-report-v1", "styles": combined})
            print(json.dumps({"style": style, "published_step": best_step, "validation_before": initial, "validation_after": best_loss}, ensure_ascii=False), flush=True)
            style_lease.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
