"""Pure MLX FP32 backend for the local dense character Transformer.

Parameters use the same names and matrix orientation as
``trash_talk.garbage_model``. The tied token/output embedding is represented
once as ``token_embedding.weight``; ``lm_head.weight`` is accepted only as a
validated input alias and is never exported as a second parameter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
from typing import Any

import mlx.core as mx
import numpy as np


Array = mx.array
FFNFunction = Callable[[Array, Mapping[str, Array], str], Array]


def _dense_ffn(value: Array, parameters: Mapping[str, Array], prefix: str) -> Array:
    up = value @ parameters[f"{prefix}.up.weight"].T
    up = up + parameters[f"{prefix}.up.bias"]
    # PyTorch nn.GELU() uses the exact erf form by default.
    activated = 0.5 * up * (1.0 + mx.erf(up * 0.7071067811865476))
    down = activated @ parameters[f"{prefix}.down.weight"].T
    return down + parameters[f"{prefix}.down.bias"]


FFN_REGISTRY: dict[str, FFNFunction] = {"dense": _dense_ffn}


def register_ffn_backend(name: str, function: FFNFunction) -> None:
    """Register a future MLX FFN implementation under an explicit name."""
    if not isinstance(name, str) or not name or not callable(function):
        raise ValueError("FFN 注册名和函数无效")
    if name in FFN_REGISTRY:
        raise ValueError(f"FFN 后端已注册: {name}")
    FFN_REGISTRY[name] = function


def _linear(value: Array, weight: Array, bias: Array) -> Array:
    return value @ weight.T + bias


def _layer_norm(value: Array, weight: Array, bias: Array) -> Array:
    mean = mx.mean(value, axis=-1, keepdims=True)
    centered = value - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    normalized = centered / mx.sqrt(variance + 1e-5)
    return normalized * weight + bias


def _linear_parameter_shapes(config: Any) -> dict[str, tuple[int, ...]]:
    vocab = config.vocab_size
    context = config.context
    width = config.d_model
    shapes: dict[str, tuple[int, ...]] = {
        "token_embedding.weight": (vocab, width),
        "position_embedding.weight": (context, width),
        "final_ln.weight": (width,),
        "final_ln.bias": (width,),
        "lm_bias": (vocab,),
    }
    for index, hidden in enumerate(config.ffn_dims):
        prefix = f"blocks.{index}"
        shapes.update(
            {
                f"{prefix}.qkv.weight": (3 * width, width),
                f"{prefix}.qkv.bias": (3 * width,),
                f"{prefix}.attn_out.weight": (width, width),
                f"{prefix}.attn_out.bias": (width,),
                f"{prefix}.ln1.weight": (width,),
                f"{prefix}.ln1.bias": (width,),
                f"{prefix}.ln2.weight": (width,),
                f"{prefix}.ln2.bias": (width,),
                f"{prefix}.ffn.up.weight": (hidden, width),
                f"{prefix}.ffn.up.bias": (hidden,),
                f"{prefix}.ffn.down.weight": (width, hidden),
                f"{prefix}.ffn.down.bias": (width,),
            }
        )
    return shapes


def _numpy_value(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        array = np.asarray(value)
    except Exception as error:
        raise TypeError(f"参数 {name} 无法转换为 NumPy 数组") from error
    if array.dtype.kind not in "fiu":
        raise TypeError(f"参数 {name} 必须是实数数组")
    if not np.isfinite(array).all():
        raise ValueError(f"参数 {name} 包含非有限值")
    return np.asarray(array, dtype=np.float32, order="C")


def _token_array(ids: Any, config: Any, *, name: str = "ids") -> Array:
    integer_mlx_types = tuple(
        getattr(mx, type_name)
        for type_name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
        if hasattr(mx, type_name)
    )
    if type(ids).__module__.startswith("mlx.core"):
        if ids.dtype not in integer_mlx_types:
            raise TypeError(f"{name} 必须是整数数组")
        shape = tuple(ids.shape)
        if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
            raise ValueError(f"{name} 必须是非空的 [B,T] 数组")
        if shape[1] > config.context:
            raise ValueError(f"{name} 长度超过模型 context")
        lower = int(mx.min(ids).item())
        upper = int(mx.max(ids).item())
        if lower < 0 or upper >= config.vocab_size:
            raise ValueError(f"{name} token ID 超出模型词表范围")
        return mx.array(ids, dtype=mx.int32)

    try:
        host = np.asarray(ids)
    except Exception as error:
        raise TypeError(f"{name} 必须是整数 [B,T] 数组或列表") from error
    if host.dtype.kind not in "iu" or host.dtype.kind == "b":
        raise TypeError(f"{name} 必须是整数数组")
    if host.ndim != 2 or host.shape[0] == 0 or host.shape[1] == 0:
        raise ValueError(f"{name} 必须是非空的 [B,T] 数组")
    if host.shape[1] > config.context:
        raise ValueError(f"{name} 长度超过模型 context")
    if int(host.min()) < 0 or int(host.max()) >= config.vocab_size:
        raise ValueError(f"{name} token ID 超出模型词表范围")
    return mx.array(host.astype(np.int32, copy=False))


class MLXBackend:
    """FP32 dense Transformer inference and stateless SGD training backend."""

    def __init__(self, config: Any, weights: Mapping[str, Any]):
        self.config = config
        if not isinstance(weights, Mapping):
            raise TypeError("weights 必须是参数名到数组的映射")
        architecture = getattr(config, "architecture", None)
        if architecture != "dense_transformer_v1":
            if isinstance(architecture, str) and architecture.startswith("moe"):
                raise ValueError(f"MoE 架构尚未实现: {architecture}")
            raise ValueError(f"MLX 后端不支持模型架构: {architecture}")
        ffn_type = getattr(config, "ffn_type", None)
        if ffn_type not in FFN_REGISTRY:
            if isinstance(ffn_type, str) and "moe" in ffn_type.lower():
                raise ValueError(f"MoE FFN 尚未实现: {ffn_type}")
            raise ValueError(f"MLX 后端不支持 FFN 类型: {ffn_type}")
        for key in ("vocab_size", "context", "d_model", "heads"):
            value = getattr(config, key, None)
            if type(value) is not int or value <= 0:
                raise ValueError(f"config.{key} 必须是正整数")
        if config.vocab_size < 1 or config.d_model % config.heads:
            raise ValueError("模型词表或 heads 配置无效")
        ffn_dims = getattr(config, "ffn_dims", None)
        if not isinstance(ffn_dims, (list, tuple)) or len(ffn_dims) != 2:
            raise ValueError("当前 dense 架构要求两层 ffn_dims")
        if any(type(width) is not int or width <= 0 for width in ffn_dims):
            raise ValueError("ffn_dims 必须是两层正整数列表")
        self._ffn = FFN_REGISTRY[ffn_type]

        expected = _linear_parameter_shapes(config)
        supplied_names = set(weights)
        allowed_names = set(expected) | {"lm_head.weight"}
        unknown = supplied_names - allowed_names
        if unknown:
            raise ValueError(f"存在未知模型参数: {', '.join(sorted(unknown))}")
        missing = set(expected) - supplied_names
        if missing:
            raise ValueError(f"缺少模型参数: {', '.join(sorted(missing))}")

        canonical: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            value = _numpy_value(weights[name], name)
            if value.shape != shape:
                raise ValueError(f"参数 {name} 形状应为 {shape}，实际为 {value.shape}")
            canonical[name] = value
        if "lm_head.weight" in weights:
            alias = _numpy_value(weights["lm_head.weight"], "lm_head.weight")
            if alias.shape != canonical["token_embedding.weight"].shape:
                raise ValueError("lm_head.weight 绑定别名的形状与 token embedding 不匹配")
            if not np.array_equal(alias, canonical["token_embedding.weight"]):
                raise ValueError("lm_head.weight 必须与 token_embedding.weight 完全相同")

        # This dictionary is the sole canonical parameter store. Tied output
        # projection reads token_embedding.weight directly.
        self._parameters: dict[str, Array] = {
            name: mx.array(value, dtype=mx.float32)
            for name, value in canonical.items()
        }
        mx.eval(*self._parameters.values())

    @property
    def parameters(self) -> dict[str, Array]:
        """Return the one canonical dictionary of trainable FP32 parameters."""
        return self._parameters

    def _forward(self, parameters: Mapping[str, Array], ids: Array) -> Array:
        batch, length = ids.shape
        width = self.config.d_model
        heads = self.config.heads
        head_dim = width // heads
        value = parameters["token_embedding.weight"][ids]
        positions = mx.arange(length, dtype=mx.int32)
        value = value + parameters["position_embedding.weight"][positions][None, :, :]

        position_ids = mx.arange(length, dtype=mx.int32)
        causal_mask = position_ids[None, :] <= position_ids[:, None]
        for index in range(2):
            prefix = f"blocks.{index}"
            normalized = _layer_norm(
                value,
                parameters[f"{prefix}.ln1.weight"],
                parameters[f"{prefix}.ln1.bias"],
            )
            qkv = _linear(
                normalized,
                parameters[f"{prefix}.qkv.weight"],
                parameters[f"{prefix}.qkv.bias"],
            )
            qkv = qkv.reshape(batch, length, 3, heads, head_dim)
            query = qkv[:, :, 0, :, :].transpose(0, 2, 1, 3)
            key = qkv[:, :, 1, :, :].transpose(0, 2, 1, 3)
            val = qkv[:, :, 2, :, :].transpose(0, 2, 1, 3)
            scores = (query @ key.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(head_dim))
            scores = mx.where(causal_mask[None, None, :, :], scores, float("-inf"))
            probabilities = mx.softmax(scores, axis=-1)
            attended = probabilities @ val
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
            value = value + _linear(
                attended,
                parameters[f"{prefix}.attn_out.weight"],
                parameters[f"{prefix}.attn_out.bias"],
            )

            normalized = _layer_norm(
                value,
                parameters[f"{prefix}.ln2.weight"],
                parameters[f"{prefix}.ln2.bias"],
            )
            value = value + self._ffn(normalized, parameters, f"{prefix}.ffn")

        value = _layer_norm(
            value, parameters["final_ln.weight"], parameters["final_ln.bias"]
        )
        return value @ parameters["token_embedding.weight"].T + parameters["lm_bias"]

    def logits(self, ids: Any) -> Array:
        """Return full-vocabulary logits with shape ``[B,T,V]``."""
        token_ids = _token_array(ids, self.config)
        result = self._forward(self._parameters, token_ids)
        mx.eval(result)
        return result

    def generate(
        self, prompt_ids: list[int], max_new_tokens: int, eos_id: int
    ) -> list[int]:
        """Greedily append full-vocabulary argmax tokens until EOS or limit."""
        if not isinstance(prompt_ids, list) or not prompt_ids:
            raise ValueError("prompt_ids 必须是非空列表")
        if any(type(token_id) is not int or token_id < 0 for token_id in prompt_ids):
            raise ValueError("prompt_ids 必须只包含非负整数")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须是非负整数")
        if type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size:
            raise ValueError("eos_id 超出模型词表范围")
        if any(token_id >= self.config.vocab_size for token_id in prompt_ids):
            raise ValueError("prompt_ids 超出模型词表范围")
        result = list(prompt_ids)
        if eos_id in result:
            return result
        for _ in range(max_new_tokens):
            active = result[-self.config.context :]
            next_id = int(self.logits([active])[0, -1].argmax(axis=-1).item())
            result.append(next_id)
            if next_id == eos_id:
                break
        return result

    def train_step(
        self,
        x: list[list[int]],
        y: list[list[int]],
        learning_rate: float = 0.005,
        pad_id: int = 0,
    ) -> float:
        """Run one FP32 cross-entropy step with global-norm clipping and SGD."""
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate 必须是正有限数")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate 必须是正有限数")
        if type(pad_id) is not int or not 0 <= pad_id < self.config.vocab_size:
            raise ValueError("pad_id 超出模型词表范围")
        input_ids = _token_array(x, self.config, name="x")
        target_ids = _token_array(y, self.config, name="y")
        if input_ids.shape != target_ids.shape:
            raise ValueError("x 和 y 必须具有相同形状")

        def loss_function(parameters: Mapping[str, Array]) -> Array:
            scores = self._forward(parameters, input_ids)
            flat_scores = scores.reshape(-1, self.config.vocab_size)
            flat_targets = target_ids.reshape(-1)
            selected = mx.take_along_axis(
                flat_scores, flat_targets[:, None], axis=-1
            ).reshape(-1)
            token_losses = mx.logsumexp(flat_scores, axis=-1) - selected
            mask = (flat_targets != pad_id).astype(mx.float32)
            denominator = mx.maximum(mx.sum(mask), mx.array(1.0, dtype=mx.float32))
            return mx.sum(token_losses * mask) / denominator

        loss, gradients = mx.value_and_grad(loss_function)(self._parameters)
        squared_norm = sum(mx.sum(gradient * gradient) for gradient in gradients.values())
        global_norm = mx.sqrt(squared_norm)
        scale = mx.minimum(1.0, 1.0 / (global_norm + 1e-12))
        updated = {
            name: parameter - float(learning_rate) * scale * gradients[name]
            for name, parameter in self._parameters.items()
        }
        mx.eval(loss, global_norm, *updated.values())
        self._parameters.update(updated)
        return float(loss.item())

    def export_weights(self) -> dict[str, np.ndarray]:
        """Return detached canonical parameter arrays in FP32."""
        mx.eval(*self._parameters.values())
        return {
            name: np.array(value, dtype=np.float32, copy=True)
            for name, value in self._parameters.items()
        }

    def weight_digest(self) -> str:
        """Hash canonical FP32 parameter bytes in stable name order."""
        mx.eval(*self._parameters.values())
        digest = hashlib.sha256()
        for name in sorted(self._parameters):
            value = np.asarray(self._parameters[name], dtype=np.dtype("<f4"))
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        return digest.hexdigest()
