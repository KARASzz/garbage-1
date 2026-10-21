#!/usr/bin/env python3
"""Evaluate the current garbage-1 checkpoint with the native MLX runtime.

The held-out metrics cover every next-token target in each packed split. Since
grammar templates and fragments are shared across splits, these are in-domain
metrics and do not measure generalization to unseen templates.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trash_talk.model_runtime import ModelRuntime


THEMES = ("game", "logic", "boast", "daily")
EVALUATION_SCOPE = (
    "validation/test are sentence-disjoint, but templates and grammar fragments "
    "are shared across splits; metrics describe in-domain next-token prediction, "
    "not unseen-template generalization."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


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
        text, theme = row.get("text"), row.get("theme")
        if not isinstance(text, str) or not text or theme not in THEMES:
            raise ValueError(f"{path.name} 第 {line_no} 行缺少有效 text/theme")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name} 为空")
    return rows


def checkpoint_metadata(model_dir: Path) -> tuple[Path, bytes, dict]:
    pointer_path = model_dir / "current.json"
    pointer_raw = pointer_path.read_bytes()
    pointer = json.loads(pointer_raw)
    relative = pointer.get("checkpoint")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("current.json 的 checkpoint 必须是相对路径")
    metadata_path = (model_dir / relative).resolve()
    if not metadata_path.is_relative_to(model_dir.resolve()):
        raise ValueError("current.json 的 checkpoint 路径越界")
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw)
    if metadata.get("format") != "garbage-1-checkpoint-v1":
        raise ValueError("当前 checkpoint metadata 格式不受支持")
    return metadata_path, metadata_raw, {**metadata, "_pointer_sha256": sha256_bytes(pointer_raw)}


def canonical_weight_digest(weights: dict[str, np.ndarray]) -> str:
    """Match MLXBackend.weight_digest over unique canonical FP32 arrays."""
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = np.asarray(weights[name], dtype=np.dtype("<f4"))
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def load_canonical_weights(metadata_path: Path, metadata: dict) -> tuple[dict, dict]:
    weight_name = metadata.get("weights")
    if not isinstance(weight_name, str) or Path(weight_name).is_absolute():
        raise ValueError("checkpoint metadata.weights 必须是相对路径")
    weights_path = (metadata_path.parent / weight_name).resolve()
    if not weights_path.is_relative_to(metadata_path.parent.resolve()):
        raise ValueError("checkpoint weights 路径越界")
    weights_sha256 = sha256_bytes(weights_path.read_bytes())
    with np.load(weights_path, allow_pickle=False) as archive:
        names = list(archive.files)
        if len(names) != len(set(names)):
            raise ValueError("NPZ 包含重复参数名")
        weights = {name: archive[name] for name in names}
    parameter_count = sum(array.size for array in weights.values())
    raw_array_bytes = sum(array.nbytes for array in weights.values())
    if parameter_count != 100_000:
        raise ValueError(f"NPZ unique 参数数为 {parameter_count}，预期 100000")
    if any(array.dtype != np.float32 for array in weights.values()):
        raise ValueError("NPZ canonical 参数必须全部是 FP32")
    if raw_array_bytes != 400_000:
        raise ValueError(f"NPZ raw array bytes 为 {raw_array_bytes}，预期 400000")
    if "lm_head.weight" in weights:
        raise ValueError("canonical NPZ 不应重复存储 tied lm_head.weight")
    digest = canonical_weight_digest(weights)
    expected = metadata.get("weight_sha256")
    if expected and expected != digest:
        raise ValueError("checkpoint metadata.weight_sha256 与 NPZ 权重不匹配")
    return weights, {
        "weights_path": str(weights_path),
        "weights_path_relative": weights_path.relative_to(metadata_path.parent).as_posix(),
        "weights_file_sha256": weights_sha256,
        "weight_sha256": digest,
        "unique_parameter_names": len(names),
        "parameter_count": parameter_count,
        "raw_array_bytes": raw_array_bytes,
        "dtype": "float32",
        "tied_lm_head_duplicate_present": False,
    }


def encode_packed_stream(rows: list[dict], tokenizer) -> tuple[np.ndarray, dict]:
    token_ids: list[int] = []
    oov = Counter()
    known = set(tokenizer.characters)
    total_characters = 0
    for row in rows:
        text = unicodedata.normalize("NFC", row["text"])
        total_characters += len(text)
        oov.update(char for char in text if char not in known)
        token_ids.extend(tokenizer.encode(text, theme=row["theme"], bos=True, eos=True))
    if len(token_ids) < 2:
        raise ValueError("packed token stream 至少需要两个 token")
    return np.asarray(token_ids, dtype=np.int32), {
        "character_count": total_characters,
        "oov_character_count": sum(oov.values()),
        "oov_unique_character_count": len(oov),
        "oov_ratio": sum(oov.values()) / total_characters if total_characters else 0.0,
        "oov_characters": dict(sorted(oov.items())),
    }


def packed_batches(
    stream: np.ndarray,
    *,
    context: int,
    batch_size: int,
    pad_id: int,
):
    """Yield contiguous, non-overlapping next-token targets with ignored tail PADs."""
    inputs = stream[:-1]
    targets = stream[1:]
    tokens_per_batch = context * batch_size
    for start in range(0, len(targets), tokens_per_batch):
        valid = min(tokens_per_batch, len(targets) - start)
        x = np.full(tokens_per_batch, pad_id, dtype=np.int32)
        y = np.full(tokens_per_batch, pad_id, dtype=np.int32)
        x[:valid] = inputs[start : start + valid]
        y[:valid] = targets[start : start + valid]
        yield x.reshape(batch_size, context), y.reshape(batch_size, context), valid


def token_cross_entropy(logits: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(target_ids, dtype=np.int64).reshape(-1)
    values = values.reshape(-1, values.shape[-1])
    if len(values) != len(targets):
        raise ValueError("logits 和 target 数量不匹配")
    maxima = np.max(values, axis=1)
    logsumexp = maxima + np.log(np.exp(values - maxima[:, None]).sum(axis=1))
    selected = values[np.arange(len(targets)), targets]
    return logsumexp - selected


def evaluate_split(
    runtime: ModelRuntime,
    rows: list[dict],
    *,
    context: int,
    batch_size: int,
) -> dict:
    stream, oov = encode_packed_stream(rows, runtime.tokenizer)
    target_count = int(stream.size - 1)
    pad_id = runtime.tokenizer.pad_id
    loss_sum = 0.0
    counted = 0
    padded_targets = 0
    start_time = time.perf_counter()
    for x, y, valid in packed_batches(
        stream, context=context, batch_size=batch_size, pad_id=pad_id
    ):
        flat_y = y.reshape(-1)
        mask = flat_y != pad_id
        if int(mask.sum()) != valid:
            raise AssertionError("真实 target 被误当作 PAD 或尾部 padding 未忽略")
        logits = np.asarray(runtime.backend.logits(x.tolist()), dtype=np.float32)
        losses = token_cross_entropy(
            logits.reshape(-1, logits.shape[-1])[:valid], flat_y[mask]
        )
        loss_sum += float(losses.sum(dtype=np.float64))
        counted += len(losses)
        padded_targets += int((~mask).sum())
    if counted != target_count:
        raise AssertionError(f"next-token target 计数 {counted} != 预期 {target_count}")
    mean_loss = loss_sum / counted
    return {
        "sentence_count": len(rows),
        "packed_stream_token_count": int(stream.size),
        "next_token_target_count": target_count,
        "counted_target_count": counted,
        "ignored_pad_target_count": padded_targets,
        "context": context,
        "batch_size": batch_size,
        "average_next_token_cross_entropy_nats": mean_loss,
        "perplexity": math.exp(mean_loss),
        "oov": oov,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def choose_prompts(prefix_doc: dict, seed: int) -> list[dict]:
    examples = prefix_doc.get("examples")
    if not isinstance(examples, list):
        raise ValueError("prefixes.json 缺少 examples 列表")
    rng = random.Random(seed)
    chosen = []
    for theme in THEMES:
        by_prefix = {}
        for example in examples:
            if example.get("theme") == theme:
                prefix = example.get("prefix")
                if isinstance(prefix, str) and prefix:
                    by_prefix.setdefault(prefix, example)
        candidates = [by_prefix[key] for key in sorted(by_prefix)]
        if len(candidates) < 4:
            raise ValueError(f"主题 {theme} 的训练 prefix 不足 4 个唯一开场")
        for example in rng.sample(candidates, 4):
            chosen.append(
                {
                    "theme": theme,
                    "prefix": example["prefix"],
                    "source_id": example.get("source_id"),
                }
            )
    return chosen


def generate_raw_samples(
    runtime: ModelRuntime,
    prompts: list[dict],
    *,
    max_new_tokens: int,
    seed: int,
) -> dict:
    rows = []
    eos_count = 0
    lengths = []
    for index, prompt in enumerate(prompts, 1):
        first = runtime.generate(
            prefix=prompt["prefix"],
            theme=prompt["theme"],
            max_new_tokens=max_new_tokens,
        )
        second = runtime.generate(
            prefix=prompt["prefix"],
            theme=prompt["theme"],
            max_new_tokens=max_new_tokens,
        )
        deterministic = (
            first["text"] == second["text"]
            and first["generated_tokens"] == second["generated_tokens"]
        )
        if not deterministic:
            raise AssertionError(f"固定 prompt 第 {index} 条重复生成不一致")

        generated_count = int(first["generated_tokens"])
        prompt_ids = runtime.tokenizer.encode(
            prompt["prefix"], theme=prompt["theme"], bos=True
        )
        if generated_count < max_new_tokens:
            ended_eos = True
        elif generated_count == max_new_tokens:
            # ModelRuntime exposes decoded text and length. On the limit edge,
            # make one backend call to distinguish EOS-at-limit from truncation.
            with runtime.lock:
                replay_ids = runtime.backend.generate(
                    prompt_ids, max_new_tokens, runtime.tokenizer.eos_id
                )
            replay_text = runtime.tokenizer.decode(replay_ids)
            if replay_text != first["text"] or len(replay_ids) - len(prompt_ids) != generated_count:
                raise AssertionError("MLX raw token replay disagrees with ModelRuntime output")
            ended_eos = bool(
                len(replay_ids) > len(prompt_ids)
                and replay_ids[-1] == runtime.tokenizer.eos_id
            )
        else:
            raise AssertionError("runtime generated token count exceeds its configured limit")

        eos_count += int(ended_eos)
        lengths.append(generated_count)
        rows.append(
            {
                "index": index,
                "theme": prompt["theme"],
                "prompt": prompt["prefix"],
                "source_train_id": prompt["source_id"],
                "raw_text": first["text"],
                "generated_token_count": generated_count,
                "ended_with_eos": ended_eos,
                "deterministic_repeat_match": deterministic,
            }
        )
    return {
        "selection_seed": seed,
        "max_new_tokens": max_new_tokens,
        "sample_count": len(rows),
        "samples_per_theme": {theme: sum(row["theme"] == theme for row in rows) for theme in THEMES},
        "eos_count": eos_count,
        "eos_rate": eos_count / len(rows) if rows else 0.0,
        "generated_token_length": {
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "values": lengths,
        },
        "all_deterministic_repeats_match": all(
            row["deterministic_repeat_match"] for row in rows
        ),
        "raw_samples": rows,
    }


def torch_cpu_parity(runtime: ModelRuntime, weights: dict[str, np.ndarray], stream: np.ndarray) -> dict:
    import torch

    from trash_talk.torch_backend import TorchBackend

    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ids = stream[:16]
    if ids.size < 16:
        raise ValueError("validation packed stream 不足 16 个 token，无法计算 parity")
    input_ids = ids.reshape(1, 16).tolist()
    mlx_logits = np.asarray(runtime.backend.logits(input_ids), dtype=np.float32)
    reference = TorchBackend(runtime.config, weights, device="cpu")
    with torch.inference_mode():
        torch_logits = (
            reference.model(torch.tensor(input_ids, dtype=torch.long))
            .detach()
            .cpu()
            .numpy()
        )
    absolute = np.abs(mlx_logits.astype(np.float64) - torch_logits.astype(np.float64))
    mlx_argmax = mlx_logits.argmax(axis=-1)
    torch_argmax = torch_logits.argmax(axis=-1)
    return {
        "backend_pair": "MLX runtime vs PyTorch CPU reference",
        "input_shape": [1, 16],
        "torch_threads": torch.get_num_threads(),
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "max_absolute_logit_error": float(absolute.max()),
        "mean_absolute_logit_error": float(absolute.mean()),
        "all_logits_within_rtol_1e-4_atol_1e-4": bool(
            np.allclose(mlx_logits, torch_logits, rtol=1e-4, atol=1e-4)
        ),
        "all_argmax_match": bool(np.array_equal(mlx_argmax, torch_argmax)),
        "last_position_argmax_match": bool(
            mlx_argmax[0, -1] == torch_argmax[0, -1]
        ),
    }


def mlx_device_info(mx) -> dict:
    try:
        default_device = str(mx.default_device())
    except Exception as error:
        default_device = f"unavailable: {type(error).__name__}: {error}"
    try:
        metal_available = bool(mx.metal.is_available())
    except Exception as error:
        metal_available = f"unavailable: {type(error).__name__}: {error}"
    return {
        "backend": "mlx",
        "default_device": default_device,
        "metal_available": metal_available,
    }


def evaluate(args: argparse.Namespace) -> dict:
    import mlx.core as mx

    root = args.root.resolve()
    model_dir = root / "models/garbage-1"
    data_dir = root / "data/garbage-1"
    metadata_path, metadata_raw, metadata = checkpoint_metadata(model_dir)
    metadata_sha256 = sha256_bytes(metadata_raw)
    weights, weight_info = load_canonical_weights(metadata_path, metadata)

    runtime = ModelRuntime(root, backend="mlx")
    if not runtime.ready:
        raise RuntimeError(f"ModelRuntime MLX 未就绪：{runtime.error}")
    if runtime.metadata.get("run_id") != metadata.get("run_id"):
        raise RuntimeError("current.json 在检查点解析期间发生变化")
    if runtime.backend_name != "mlx":
        raise AssertionError("评估必须通过 MLX 实际后端")
    if sum(array.size for array in runtime.backend.parameters.values()) != 100_000:
        raise AssertionError("MLX runtime canonical 参数数量不是 100000")
    if any(array.dtype != mx.float32 for array in runtime.backend.parameters.values()):
        raise AssertionError("MLX runtime 参数存在非 float32 dtype")
    if runtime.backend.weight_digest() != weight_info["weight_sha256"]:
        raise AssertionError("MLX runtime digest 与 checkpoint canonical NPZ 不一致")

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    validation_rows = read_jsonl(data_dir / "validation.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    validation = evaluate_split(
        runtime,
        validation_rows,
        context=runtime.config.context,
        batch_size=args.batch_size,
    )
    test = evaluate_split(
        runtime,
        test_rows,
        context=runtime.config.context,
        batch_size=args.batch_size,
    )
    parity_stream, _ = encode_packed_stream(validation_rows, runtime.tokenizer)
    parity = torch_cpu_parity(runtime, weights, parity_stream)

    prefix_doc = json.loads((data_dir / "prefixes.json").read_text(encoding="utf-8"))
    prompts = choose_prompts(prefix_doc, args.seed)
    samples = generate_raw_samples(
        runtime, prompts, max_new_tokens=args.max_new_tokens, seed=args.seed
    )

    # Verify the pointer and metadata still identify the checkpoint used by
    # ModelRuntime after the complete evaluation and generation pass.
    final_path, final_raw, final_metadata = checkpoint_metadata(model_dir)
    if (
        final_path != metadata_path
        or sha256_bytes(final_raw) != metadata_sha256
        or final_metadata.get("_pointer_sha256") != metadata.get("_pointer_sha256")
    ):
        raise RuntimeError("评估过程中 current checkpoint 发生变化；拒绝写出混合结果")

    source_sha = metadata.get("source_sha256")
    result = {
        "format": "garbage-1-evaluation-v1",
        "evaluated_at_utc": utc_now(),
        "scope": EVALUATION_SCOPE,
        "checkpoint": {
            "metadata_path": str(metadata_path),
            "metadata_path_relative": metadata_path.relative_to(model_dir).as_posix(),
            "metadata_sha256": metadata_sha256,
            "weights_path": weight_info["weights_path"],
            "weights_file_sha256": weight_info["weights_file_sha256"],
            "weight_sha256": weight_info["weight_sha256"],
            "run_id": metadata.get("run_id"),
            "step": metadata.get("step"),
            "source_sha256": source_sha,
            "dataset_source_sha256": manifest.get("dataset_source_sha256"),
            "source_matches_dataset": source_sha == manifest.get("dataset_source_sha256"),
            "parameter_storage": weight_info,
            "config": metadata.get("config"),
            "dtype": metadata.get("dtype"),
        },
        "runtime": {
            "ready": runtime.ready,
            "status": runtime.status(),
            "device": mlx_device_info(mx),
            "backend_weight_digest": runtime.backend.weight_digest(),
        },
        "evaluation": {
            "packing": "all split rows encoded as BOS/theme/text/EOS and concatenated; every adjacent next-token target is counted once; final partial context is PAD-filled and PAD targets are ignored",
            "batch_size": args.batch_size,
            "context": runtime.config.context,
            "validation": validation,
            "test": test,
        },
        "torch_cpu_reference": parity,
        "generation": samples,
    }
    output = args.output or root / "output/garbage-1-evaluation.json"
    output = output.resolve()
    atomic_write(
        output,
        json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
    )
    return {"status": "evaluated", "output": str(output), "checkpoint": result["checkpoint"],
            "validation": validation, "test": test, "generation": {
                "sample_count": samples["sample_count"], "eos_rate": samples["eos_rate"],
                "all_deterministic_repeats_match": samples["all_deterministic_repeats_match"]},
            "torch_cpu_reference": parity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正数")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须为正数")
    args.root = args.root.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
