#!/usr/bin/env python3
"""Train the exact-budget garbage-1 model and write an atomic NPZ checkpoint."""

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
import uuid
import unicodedata

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.garbage_model import (
    BYTE_TOKEN_COUNT,
    SPECIAL_TOKENS,
    CharacterTokenizer,
    build_model,
    solve_config,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name} 第 {line_no} 行不是有效 JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name} 第 {line_no} 行必须是 JSON 对象")
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError(f"{path.name} 第 {line_no} 行缺少 text")
        if row.get("theme") not in {"game", "logic", "boast", "daily"}:
            raise ValueError(f"{path.name} 第 {line_no} 行 theme 无效")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} 为空")
    return rows


def text_digest(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def encode_stream(rows: list[dict], tokenizer: CharacterTokenizer) -> torch.Tensor:
    stream: list[int] = []
    for row in rows:
        stream.extend(
            tokenizer.encode(row["text"], theme=row["theme"], bos=True, eos=True)
        )
    if len(stream) < 2:
        raise ValueError("编码后的 packed token stream 至少需要两个 token")
    return torch.tensor(stream, dtype=torch.long, device="cpu")


def packed_batch(
    stream: torch.Tensor,
    cursor: int,
    batch_size: int,
    context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Read a full-context batch from the circular packed stream.

    The cursor advances by every input position used. Across a stream cycle,
    each packed training position participates as an input token.
    """
    length = stream.numel()
    count = batch_size * context
    offsets = (torch.arange(count, dtype=torch.long) + cursor) % length
    inputs = stream[offsets].view(batch_size, context).to(device=device)
    targets = stream[(offsets + 1) % length].view(batch_size, context).to(device=device)
    return inputs, targets, (cursor + count) % length


def loss_for_batch(model, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.inference_mode()
def evaluate(
    model,
    stream: torch.Tensor,
    *,
    batch_size: int,
    context: int,
    device: torch.device,
    batches: int,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    cursor = 0
    try:
        for _ in range(batches):
            inputs, targets, cursor = packed_batch(
                stream, cursor, batch_size, context, device
            )
            losses.append(float(loss_for_batch(model, inputs, targets).cpu()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def save_checkpoint(
    *,
    model,
    config,
    tokenizer: CharacterTokenizer,
    output_dir: Path,
    source_sha256: str,
    seed: int,
    steps: int,
    training: dict,
) -> dict:
    named_parameters = list(model.named_parameters())
    arrays = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        for name, parameter in named_parameters
    }
    parameter_count = sum(array.size for array in arrays.values())
    if parameter_count != config.target_parameters:
        raise RuntimeError(
            f"检查点参数数错误：{parameter_count}，预期 {config.target_parameters}"
        )
    if any(array.dtype != np.float32 for array in arrays.values()):
        raise RuntimeError("检查点必须全部使用 float32")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name], dtype=np.dtype("<f4"))
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    weight_sha256 = digest.hexdigest()

    run_id = str(uuid.uuid4())
    generation = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{run_id[:12]}"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_name = f"{generation}.npz"
    metadata_name = f"{generation}.json"
    weights_path = checkpoint_dir / weights_name
    temp_weights = checkpoint_dir / f".{generation}.{uuid.uuid4().hex}.npz.tmp"
    try:
        with temp_weights.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_weights, weights_path)
    finally:
        temp_weights.unlink(missing_ok=True)

    metadata = {
        "format": "garbage-1-checkpoint-v1",
        "generated_at_utc": utc_now(),
        "config": config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "weights": weights_name,
        "weight_sha256": weight_sha256,
        "step": steps,
        "run_id": run_id,
        "source_sha256": source_sha256,
        "seed": seed,
        "last_job_id": 0,
        "training": training,
        "parameter_count": parameter_count,
        "dtype": "float32",
    }
    metadata_path = checkpoint_dir / metadata_name
    atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
    )
    current = {"checkpoint": f"checkpoints/{metadata_name}"}
    atomic_write(
        output_dir / "current.json",
        json.dumps(current, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
    )
    return {
        "generation": generation,
        "weights": str(weights_path),
        "metadata": str(metadata_path),
        "current": str(output_dir / "current.json"),
        "parameter_count": parameter_count,
        "run_id": run_id,
    }


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已请求 CUDA，但当前 PyTorch/CUDA 设备不可用")
        return torch.device("cuda")
    if name == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("已请求 MPS，但当前 PyTorch/MPS 设备不可用")
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> dict:
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        # PyTorch only allows this before the first parallel operation. The
        # intra-op thread count above remains enforced for this run.
        pass
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    data_dir = args.root / "data/garbage-1"
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "garbage-1-dataset-v1":
        raise ValueError("数据集 manifest 格式不匹配，请先运行 prepare_garbage.py")
    train_rows = read_jsonl(data_dir / "train.jsonl")
    validation_rows = read_jsonl(data_dir / "validation.jsonl")

    train_hashes = {text_digest(row["text"]) for row in train_rows}
    validation_hashes = {text_digest(row["text"]) for row in validation_rows}
    if len(train_hashes) != len(train_rows) or len(validation_hashes) != len(validation_rows):
        raise ValueError("train 或 validation 内存在重复完整句")
    overlap = train_hashes & validation_hashes
    if overlap:
        raise ValueError(f"train/validation 完整句泄漏：{len(overlap)} 条")

    # Fit vocabulary strictly from train.jsonl. Held-out text is encoded only
    # after fitting and can map unseen characters to the reserved UNK token.
    byte_fallback = args.target_parameters >= 1_000_000
    tokenizer = CharacterTokenizer.fit(
        [row["text"] for row in train_rows], byte_fallback=byte_fallback
    )
    config = solve_config(tokenizer.vocab_size, target=args.target_parameters)
    if args.target_parameters == 100_000:
        expected_config = manifest.get("training_only_tokenizer", {}).get(
            "exact_100k_parameter_config"
        )
        if expected_config and expected_config != config.to_dict():
            raise ValueError("manifest 中的精确 100K 配置与 train-only tokenizer 不一致")
    manifest_vocab_size = manifest.get("training_only_tokenizer", {}).get("vocab_size")
    expected_vocab_size = manifest_vocab_size + (256 if byte_fallback else 0)
    if tokenizer.vocab_size != expected_vocab_size:
        raise ValueError("manifest 中的 tokenizer 詞表大小與 train-only tokenizer 不一致")

    train_stream = encode_stream(train_rows, tokenizer)
    validation_stream = encode_stream(validation_rows, tokenizer)
    steps_per_epoch = max(
        1, math.ceil(train_stream.numel() / (args.batch_size * config.context))
    )
    if args.epochs is not None:
        total_steps = args.epochs * steps_per_epoch
    else:
        total_steps = args.steps if args.steps is not None else 3000
    model = build_model(config).to(device=device, dtype=torch.float32)
    if byte_fallback:
        byte_start = len(SPECIAL_TOKENS)
        with torch.no_grad():
            model.lm_bias[byte_start:byte_start + BYTE_TOKEN_COUNT].fill_(-4.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    cursor = 0
    window_losses = []
    final_train_loss = float("nan")
    final_validation_loss = float("nan")
    model.train()
    for step in range(1, total_steps + 1):
        inputs, targets, cursor = packed_batch(
            train_stream, cursor, args.batch_size, config.context, device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, inputs, targets)
        if model.aux_loss is not None:
            loss = loss + model.aux_loss
        loss.backward()
        optimizer.step()
        final_train_loss = float(loss.detach().cpu())
        window_losses.append(final_train_loss)

        if step % args.log_every == 0 or step == 1 or step == total_steps:
            final_validation_loss = evaluate(
                model,
                validation_stream,
                batch_size=args.batch_size,
                context=config.context,
                device=device,
                batches=args.eval_batches,
            )
            logged_train_loss = sum(window_losses) / len(window_losses)
            print(
                json.dumps(
                    {
                        "step": step,
                        "steps": total_steps,
                        "epoch": min(args.epochs, math.ceil(step / steps_per_epoch))
                        if args.epochs is not None else None,
                        "train_loss": logged_train_loss,
                        "validation_loss": final_validation_loss,
                        "device": device.type,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            window_losses.clear()

    dataset_source_sha = manifest.get("dataset_source_sha256") or manifest.get("source_sha256")
    if not isinstance(dataset_source_sha, str) or len(dataset_source_sha) != 64:
        raise ValueError("manifest 缺少有效 dataset_source_sha256")
    training_metadata = {
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "device": device.type,
        "cpu_threads": args.threads,
        "steps": total_steps,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "target_parameters": config.target_parameters,
        "byte_fallback": tokenizer.byte_fallback,
        "byte_output_bias_init": -4.0 if byte_fallback else None,
        "log_every": args.log_every,
        "eval_batches": args.eval_batches,
        "float32_matmul_precision": "highest",
        "allow_tf32": False,
        "latest_train_loss": final_train_loss,
        "latest_validation_loss": final_validation_loss,
        "precision": "float32",
        "autocast": False,
        "quantization": None,
        "dropout": 0.0,
        "input_stream_tokens": train_stream.numel(),
        "validation_stream_tokens": validation_stream.numel(),
        "stream_policy": "circular packed stream; full context; cursor advances by every input position",
    }
    output_dir = args.output or (args.root / "models/garbage-1")
    checkpoint = save_checkpoint(
        model=model,
        config=config,
        tokenizer=tokenizer,
        output_dir=output_dir,
        source_sha256=dataset_source_sha,
        seed=args.seed,
        steps=total_steps,
        training=training_metadata,
    )
    return {
        "status": "trained",
        "step": total_steps,
        "epochs": args.epochs,
        "train_loss": final_train_loss,
        "validation_loss": final_validation_loss,
        "source_sha256": dataset_source_sha,
        "vocab_size": tokenizer.vocab_size,
        "context": config.context,
        "parameter_count": checkpoint["parameter_count"],
        "device": device.type,
        "checkpoint": checkpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None, help="输出目录；默认 <root>/models/garbage-1")
    parser.add_argument("--steps", type=int, default=None, help="固定训练步数；与 --epochs 二选一")
    parser.add_argument("--epochs", type=int, default=None, help="按 packed train stream 完整覆盖次数训练")
    parser.add_argument("--target-parameters", type=int, choices=(100_000, 1_000_000), default=100_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=4)
    args = parser.parse_args()
    if args.steps is not None and args.epochs is not None:
        parser.error("--steps 和 --epochs 只能指定一个")
    if (args.steps is not None and args.steps <= 0) or (args.epochs is not None and args.epochs <= 0) or args.batch_size <= 0 or args.threads <= 0:
        parser.error("steps、epochs、batch-size 和 threads 必须为正数")
    if args.log_every <= 0 or args.eval_batches <= 0:
        parser.error("log-every 和 eval-batches 必须为正数")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate 必须为正数，weight-decay 不能为负数")
    args.root = args.root.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
