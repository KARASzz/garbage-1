#!/usr/bin/env python3
"""Pre-train an exact 10,000,000 parameter base Transformer on Apple Silicon via MLX/PyTorch.

Integrates:
- Full grammar-expanded syntax corpus
- Causal world logic and survival/human-nature commonsense
- 8-layer deep causal Transformer with exact 10M FP32 budget
- Automatic initialization of 9 matching LoRA adapter slots
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import unicodedata
import uuid

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.garbage_model import (
    BYTE_TOKEN_COUNT,
    SPECIAL_TOKENS,
    CharacterTokenizer,
    ModelConfig,
    build_model,
    solve_config,
)
from trash_talk.lora import initial_adapter, STYLE_NAMES
from trash_talk.model_runtime import _atomic_json, _sync_directory, prune_checkpoints


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def encode_stream(rows: list[dict], tokenizer: CharacterTokenizer) -> torch.Tensor:
    stream: list[int] = []
    for row in rows:
        theme = row.get("theme") if row.get("theme") in ("game", "logic", "boast", "daily") else "daily"
        stream.extend(tokenizer.encode(row["text"], theme=theme, bos=True, eos=True))
    return torch.tensor(stream, dtype=torch.long, device="cpu")


def packed_batch(
    stream: torch.Tensor,
    cursor: int,
    batch_size: int,
    context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    length = stream.numel()
    count = batch_size * context
    offsets = (torch.arange(count, dtype=torch.long) + cursor) % length
    inputs = stream[offsets].view(batch_size, context).to(device=device)
    targets = stream[(offsets + 1) % length].view(batch_size, context).to(device=device)
    return inputs, targets, (cursor + count) % length


def main():
    parser = argparse.ArgumentParser(description="从零预训练 1000 万参数 (10M) 底座模型")
    parser.add_argument("--steps", type=int, default=600, help="预训练步数 (默认 600 步)")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小 (默认 8)")
    parser.add_argument("--lr", type=float, default=6e-4, help="初始学习率 (默认 6e-4)")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps", help="训练设备")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "garbage-10m", help="检查点输出目录")
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("【嘴硬终端 · 10M 底座预训练】")
    print(f"目标参数量: 10,000,000 (FP32)")
    print(f"训练设备: {args.device} | 批次: {args.batch_size} | 步数: {args.steps} | 初始学习率: {args.lr}")
    print("=" * 60)

    # 1. 语料融合：语法抽卡库 + 因果常识库 (5x 权重强化)
    train_rows = read_jsonl(ROOT / "data" / "garbage-1" / "train.jsonl")
    causal_file = ROOT / "data" / "causal_commonsense.jsonl"
    causal_rows = read_jsonl(causal_file) if causal_file.exists() else []

    all_train = list(train_rows) + causal_rows * 5
    random.seed(20261103)
    random.shuffle(all_train)
    print(f"训练集语料规模: {len(all_train)} 句 (含 {len(causal_rows)} 条 5x 强化的因果常识)")

    # 2. 构建包含所有汉字与 UTF-8 回退的统一分词器
    tokenizer = CharacterTokenizer.fit([r["text"] for r in all_train], byte_fallback=True)
    print(f"分词器拟合完成: 词表大小 = {tokenizer.vocab_size} (含单字、控制符与 256 字节回退)")

    # 3. 求解精确 10,000,000 参数配置
    config = solve_config(tokenizer.vocab_size, target=10_000_000)
    print(f"解算精确 10M 架构: 层数={len(config.ffn_dims)}, 隐藏宽度={config.d_model}, 头数={config.heads}, 上下文={config.context}")
    print(f"每层 FFN 宽度: {config.ffn_dims[:3]}...{config.ffn_dims[-1]}")

    # 4. 构建模型
    device = torch.device(args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = build_model(config).to(device=device, dtype=torch.float32)

    # UTF-8 字节 token 初始偏置设低，优先输出真实汉字
    byte_start = len(SPECIAL_TOKENS)
    with torch.no_grad():
        model.lm_bias[byte_start : byte_start + BYTE_TOKEN_COUNT].fill_(-4.0)

    # 5. 编码为 Packed Stream
    print("正在编码 Token 数据流...")
    train_stream = encode_stream(all_train, tokenizer)
    val_rows = read_jsonl(ROOT / "data" / "garbage-1" / "validation.jsonl")
    val_stream = encode_stream(val_rows, tokenizer)
    print(f"Token 流总量: 训练集 {train_stream.numel()} tokens, 验证集 {val_stream.numel()} tokens")

    # 6. 优化器与余弦退火
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    cursor = 0
    t0 = time.time()
    best_loss = float("inf")

    print("\n>>> 开始预训练迭代...")
    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets, cursor = packed_batch(train_stream, cursor, args.batch_size, config.context, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        cur_loss = float(loss.detach().cpu())
        if cur_loss < best_loss:
            best_loss = cur_loss

        if step % 50 == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - t0
            speed = step / max(1e-5, elapsed)
            lr_now = scheduler.get_last_lr()[0]
            print(f"Step {step:04d}/{args.steps:04d} | Loss: {cur_loss:.4f} (Best: {best_loss:.4f}) | LR: {lr_now:.6f} | {speed:.1f} steps/s")

    # 7. 保存原子 NPZ 与 JSON 检查点
    print("\n正在生成 10M 权重快照...")
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    generation_id = f"10m-base-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    named_params = {name: p.detach().cpu().numpy().astype(np.float32) for name, p in model.named_parameters()}
    param_count = sum(p.size for p in named_params.values())
    assert param_count == 10_000_000, f"参数量异常: {param_count}"

    digest = hashlib.sha256()
    for name in sorted(named_params):
        digest.update(np.asarray(named_params[name], dtype="<f4").tobytes())
    weight_sha = digest.hexdigest()

    weights_file = checkpoint_dir / f"{generation_id}.npz"
    with weights_file.open("wb") as handle:
        np.savez(handle, **named_params)

    metadata = {
        "format": "garbage-1-checkpoint-v1",
        "generated_at_utc": utc_now(),
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "weights": weights_file.name,
        "weight_sha256": weight_sha,
        "step": args.steps,
        "run_id": f"garbage-10m-{uuid.uuid4().hex[:12]}",
        "parameter_count": 10_000_000,
        "dtype": "float32",
        "training": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "steps": args.steps,
            "best_loss": best_loss,
            "device": device.type,
        },
    }

    meta_file = checkpoint_dir / f"{generation_id}.json"
    _atomic_json(meta_file, metadata)
    _atomic_json(output_dir / "current.json", {"checkpoint": f"checkpoints/{meta_file.name}"})
    prune_checkpoints(checkpoint_dir)

    # 保存与此底座配套的 prefixes 与 seed_replay
    prefixes = [{"text": r["text"][:4], "theme": r.get("theme", "daily")} for r in all_train[:200]]
    _atomic_json(output_dir / "prefixes.json", {"format_version": 1, "count": len(prefixes), "examples": prefixes})
    replays = [{"text": r["text"], "theme": r.get("theme", "daily")} for r in all_train[:500]]
    _atomic_json(output_dir / "seed_replay.json", {"format_version": 1, "count": len(replays), "examples": replays})

    print(f"✓ 10M 底座检查点已保存至: {output_dir}")
    print(f"  权重文件: {weights_file.name} (FP32, 10,000,000 参数, SHA256: {weight_sha[:12]}...)")

    # 8. 为 9 种风格准备初始 10M LoRA 槽位 (Rank 8, Alpha 16)
    print("\n正在为 9 种风格初始化配套的 10M LoRA 适配器槽位...")
    lora_root = output_dir / "lora"
    lora_root.mkdir(parents=True, exist_ok=True)

    # 固定一份 base 快照
    base_target = lora_root / "base"
    base_target.mkdir(parents=True, exist_ok=True)
    with (base_target / "base.npz").open("wb") as handle:
        np.savez(handle, **named_params)
    _atomic_json(base_target / "base.json", {**metadata, "weights": "base.npz"})
    _atomic_json(base_target / "current.json", {"checkpoint": "base.json"})

    for style_id, style_name in STYLE_NAMES.items():
        style_dir = lora_root / style_id
        style_dir.mkdir(parents=True, exist_ok=True)
        adapter_arrays = initial_adapter(config, rank=8, include_ffn=True)
        adapter_weights = style_dir / "adapter_init.npz"
        with adapter_weights.open("wb") as handle:
            np.savez(handle, **adapter_arrays)

        adapter_digest = hashlib.sha256()
        for k in sorted(adapter_arrays):
            adapter_digest.update(k.encode())
            adapter_digest.update(np.asarray(adapter_arrays[k], dtype="<f4").tobytes())

        style_meta = {
            "format": "garbage-1-lora-v1",
            "style": style_id,
            "base_sha256": weight_sha,
            "base_step": args.steps,
            "step": 0,
            "last_job_id": 0,
            "run_id": f"lora-10m-{style_id}-{uuid.uuid4().hex[:8]}",
            "rank": 8,
            "alpha": 16,
            "weights": adapter_weights.name,
            "weight_sha256": adapter_digest.hexdigest(),
        }
        _atomic_json(style_dir / "adapter_init.json", style_meta)
        _atomic_json(style_dir / "current.json", {"checkpoint": "adapter_init.json"})

    print(f"✓ 9 种 LoRA 槽位全部初始化完成（适配 10M 底座，支持全部 8 层 Transformer）！")
    print("=" * 60)
    print("【完成】10M 底座预训练大功告成！你可以随时投喂语料或点赞点踩观察各自的生长变化。")
    print("=" * 60)


if __name__ == "__main__":
    main()
