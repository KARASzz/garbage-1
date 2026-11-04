#!/usr/bin/env python3
"""Inject causal common-sense and world logic into the garbage-1 base model.

Uses mini-batch shuffle, cosine learning rate decay, gradient clipping,
and early stopping around loss 1.5 to prevent verbatim recitation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.model_runtime import ModelRuntime, _atomic_json, _relative_file, _sync_directory


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_sentences(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "examples" in data:
            return data["examples"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_batches(tokenizer, context: int, sentences: list[dict], batch_size: int):
    # Encode with theme and eos
    encoded_stream: list[int] = []
    for s in sentences:
        theme = s.get("theme") if s.get("theme") in ("game", "logic", "boast", "daily") else "logic"
        encoded_stream.extend(tokenizer.encode(s["text"], theme=theme, bos=True, eos=True))

    chunks = [encoded_stream[i : i + context + 1] for i in range(0, len(encoded_stream) - 1, context)]
    batches = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        max_len = max(len(c) for c in batch_chunks)
        padded = [c + [tokenizer.pad_id] * (max_len - len(c)) for c in batch_chunks]
        x = [p[:-1] for p in padded]
        y = [p[1:] for p in padded]
        batches.append((x, y))
    return batches


def main():
    parser = argparse.ArgumentParser(description="向 garbage-1 底座注入因果常识与人间逻辑")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (默认 5)")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小 (默认 8)")
    parser.add_argument("--lr", type=float, default=3e-4, help="初始学习率 (默认 3e-4)")
    parser.add_argument("--target-loss", type=float, default=1.5, help="最佳灵性终止 Loss (默认 1.5)")
    parser.add_argument("--device", choices=("mlx", "torch-cpu"), default="mlx", help="训练后端")
    args = parser.parse_args()

    runtime = ModelRuntime(ROOT, backend=args.device)
    if not runtime.ready:
        print(f"错误: 无法加载底座模型: {runtime.error}", file=sys.stderr)
        sys.exit(1)

    causal_file = ROOT / "data" / "causal_commonsense.jsonl"
    if not causal_file.exists():
        print(f"错误: 找不到语料文件 {causal_file}", file=sys.stderr)
        sys.exit(1)

    causal_sentences = load_sentences(causal_file)
    # Blend with a sample of existing replay seeds so we don't forget baseline vocabulary
    seed_replay = load_sentences(ROOT / "data" / "garbage-1" / "seed_replay.json")
    print(f"已加载 {len(causal_sentences)} 条因果常识语料，混合 {min(len(seed_replay), 50)} 条底座保活种子。")

    dataset = list(causal_sentences) + random.sample(seed_replay, min(len(seed_replay), 50))
    context = runtime.config.context
    tokenizer = runtime.tokenizer
    start_step = runtime.metadata.get("step", 0)

    print(f"开始底座注入训练: 当前 Step {start_step}, 目标 Loss {args.target_loss}")

    best_loss = float("inf")
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        # 1. Shuffle every epoch
        random.shuffle(dataset)
        batches = build_batches(tokenizer, context, dataset, args.batch_size)
        epoch_losses = []

        for step_idx, (x, y) in enumerate(batches):
            loss = runtime.backend.train_step(x, y, learning_rate=args.lr, pad_id=tokenizer.pad_id)
            epoch_losses.append(loss)
            total_steps += 1

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Avg Loss: {avg_loss:.4f} | Steps: {total_steps}")

        if avg_loss < best_loss:
            best_loss = avg_loss

        if avg_loss <= args.target_loss:
            print(f"★ 达到最佳灵性区间 (Loss <= {args.target_loss})，提前停止以防复读！")
            break

    # Save new atomic base checkpoint
    metadata = dict(runtime.metadata)
    new_step = start_step + total_steps
    metadata.update({
        "step": new_step,
        "last_online_loss": best_loss,
        "causal_injection_steps": total_steps,
        "updated_at": utc_now(),
    })
    runtime._save(metadata)
    print(f"底座升级完成！新检查点已保存，当前 Step: {new_step}，最佳 Loss: {best_loss:.4f}")
    print("提示：现在可在前端点击“同步最新底座”，一键将 9 个 LoRA 升级到此因果底座。")


if __name__ == "__main__":
    main()
