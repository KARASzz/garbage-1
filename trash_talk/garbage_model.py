"""Small, exact-budget character Transformer for the local trash-talk app.

The dense model has a stable state-dict layout intended to be mirrored by
other runtimes. The vocabulary is fitted only from the caller's training
texts; reserved controls and theme tokens always occupy the first eight IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
import unicodedata
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TOKENIZER_FORMAT_VERSION = 1
TOKENIZER_BYTE_FALLBACK_FORMAT_VERSION = 2
MODEL_FORMAT_VERSION = 1
BYTE_TOKEN_COUNT = 256
SPECIAL_TOKENS = (
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "<THEME_GAME>",
    "<THEME_LOGIC>",
    "<THEME_BOAST>",
    "<THEME_DAILY>",
)
THEMES = ("game", "logic", "boast", "daily")
THEME_TO_TOKEN = dict(zip(THEMES, SPECIAL_TOKENS[4:], strict=True))
_SPECIAL_TO_ID = {token: index for index, token in enumerate(SPECIAL_TOKENS)}


class CharacterTokenizer:
    """NFC character tokenizer with optional lossless UTF-8 byte fallback."""

    def __init__(self, characters: Sequence[str], *, byte_fallback: bool = False):
        if type(byte_fallback) is not bool:
            raise ValueError("byte_fallback 必须是布尔值")
        if any(not isinstance(char, str) for char in characters):
            raise ValueError("字符表必须只包含单个 Unicode 字符")
        normalized = [unicodedata.normalize("NFC", char) for char in characters]
        if any(len(char) != 1 for char in normalized):
            raise ValueError("字符表必须只包含单个 Unicode 字符")
        if len(set(normalized)) != len(normalized):
            raise ValueError("字符表包含重复字符")
        if any(char in _SPECIAL_TO_ID for char in normalized):
            raise ValueError("字符表不能包含特殊 token 名称")
        self.characters = tuple(normalized)
        self.byte_fallback = byte_fallback
        self._byte_start = len(SPECIAL_TOKENS) if byte_fallback else None
        self._char_start = len(SPECIAL_TOKENS) + (BYTE_TOKEN_COUNT if byte_fallback else 0)
        self._char_to_id = {
            char: self._char_start + index
            for index, char in enumerate(self.characters)
        }
        self._id_to_char = {
            token_id: char for char, token_id in self._char_to_id.items()
        }
        self._theme_to_id = {
            theme: _SPECIAL_TO_ID[token]
            for theme, token in THEME_TO_TOKEN.items()
        }

    @classmethod
    def fit(cls, texts: list[str], *, byte_fallback: bool = False) -> CharacterTokenizer:
        """Fit only on the supplied training texts, never on held-out text."""
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("texts 必须是字符串列表")
        characters = {
            char
            for text in texts
            for char in unicodedata.normalize("NFC", text)
        }
        return cls(sorted(characters), byte_fallback=byte_fallback)

    @property
    def vocab_size(self) -> int:
        return len(SPECIAL_TOKENS) + (BYTE_TOKEN_COUNT if self.byte_fallback else 0) + len(self.characters)

    @property
    def pad_id(self) -> int:
        return _SPECIAL_TO_ID["<PAD>"]

    @property
    def bos_id(self) -> int:
        return _SPECIAL_TO_ID["<BOS>"]

    @property
    def eos_id(self) -> int:
        return _SPECIAL_TO_ID["<EOS>"]

    @property
    def unk_id(self) -> int:
        return _SPECIAL_TO_ID["<UNK>"]

    def encode(
        self,
        text: str,
        theme: str | None = None,
        bos: bool = True,
        eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text 必须是字符串")
        if theme is not None and theme not in self._theme_to_id:
            raise ValueError(f"未知主题: {theme}")
        if type(bos) is not bool or type(eos) is not bool:
            raise TypeError("bos 和 eos 必须是布尔值")

        result: list[int] = []
        if bos:
            result.append(self.bos_id)
        if theme is not None:
            result.append(self._theme_to_id[theme])
        for char in unicodedata.normalize("NFC", text):
            token_id = self._char_to_id.get(char)
            if token_id is not None:
                result.append(token_id)
            elif self.byte_fallback:
                result.extend(self._byte_start + value for value in char.encode("utf-8"))
            else:
                result.append(self.unk_id)
        if eos:
            result.append(self.eos_id)
        return result

    def decode(self, ids: Sequence[int]) -> str:
        """Decode character IDs and omit every reserved special token."""
        pieces: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                pieces.append(bytes(pending_bytes).decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token ID 必须是整数")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"token ID 超出词表范围: {token_id}")
            if self.byte_fallback and self._byte_start <= token_id < self._char_start:
                pending_bytes.append(token_id - self._byte_start)
                continue
            flush_bytes()
            char = self._id_to_char.get(token_id)
            if char is not None:
                pieces.append(char)
        flush_bytes()
        return "".join(pieces)

    def to_dict(self) -> dict:
        data = {
            "format_version": TOKENIZER_BYTE_FALLBACK_FORMAT_VERSION if self.byte_fallback else TOKENIZER_FORMAT_VERSION,
            "normalization": "NFC",
            "special_tokens": list(SPECIAL_TOKENS),
            "characters": list(self.characters),
        }
        if self.byte_fallback:
            data["byte_fallback"] = True
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> CharacterTokenizer:
        if not isinstance(data, Mapping):
            raise TypeError("tokenizer 数据必须是映射")
        expected_v1 = {"format_version", "normalization", "special_tokens", "characters"}
        expected_v2 = expected_v1 | {"byte_fallback"}
        version = data.get("format_version")
        if version == TOKENIZER_FORMAT_VERSION:
            expected = expected_v1
            byte_fallback = False
        elif version == TOKENIZER_BYTE_FALLBACK_FORMAT_VERSION:
            expected = expected_v2
            byte_fallback = data.get("byte_fallback") is True
            if not byte_fallback:
                raise ValueError("tokenizer v2 必须启用 byte_fallback")
        else:
            raise ValueError("不支持的 tokenizer 格式版本")
        if set(data) != expected:
            raise ValueError("tokenizer 配置字段不匹配")
        if data["normalization"] != "NFC":
            raise ValueError("只支持 NFC 字符规范化")
        if data["special_tokens"] != list(SPECIAL_TOKENS):
            raise ValueError("特殊 token 列表或顺序不匹配")
        characters = data["characters"]
        if not isinstance(characters, list) or any(not isinstance(char, str) for char in characters):
            raise ValueError("characters 必须是字符串列表")
        if characters != sorted(characters):
            raise ValueError("characters 必须按 Unicode 顺序排列")
        return cls(characters, byte_fallback=byte_fallback)


FFNFactory = Callable[[int, int], nn.Module]
ArchitectureFactory = Callable[["ModelConfig"], nn.Module]


def _dense_ffn_factory(d_model: int, hidden_dim: int) -> nn.Module:
    return _DenseFeedForward(d_model, hidden_dim)


FFN_REGISTRY: dict[str, FFNFactory] = {"dense": _dense_ffn_factory}
ARCHITECTURE_REGISTRY: dict[str, ArchitectureFactory] = {}


def register_ffn(name: str, factory: FFNFactory) -> None:
    """Register an explicit FFN implementation for a future architecture."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("FFN 注册名和 factory 无效")
    if name in FFN_REGISTRY:
        raise ValueError(f"FFN 类型已注册: {name}")
    FFN_REGISTRY[name] = factory


def register_architecture(name: str, factory: ArchitectureFactory) -> None:
    """Register an explicit architecture; no placeholder experts are created."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("架构注册名和 factory 无效")
    if name in ARCHITECTURE_REGISTRY:
        raise ValueError(f"架构已注册: {name}")
    ARCHITECTURE_REGISTRY[name] = factory


@dataclass
class ModelConfig:
    vocab_size: int
    context: int
    d_model: int
    heads: int
    ffn_dims: list[int]
    architecture: str = "dense_transformer_v1"
    ffn_type: str = "dense"
    target_parameters: int = 100_000

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, str):
            raise ValueError("architecture 必须是字符串")
        if not isinstance(self.ffn_type, str):
            raise ValueError("ffn_type 必须是字符串")
        if self.architecture not in ARCHITECTURE_REGISTRY:
            if self.architecture.startswith("moe"):
                raise ValueError(f"MoE 架构尚未实现: {self.architecture}")
            raise ValueError(f"未知模型架构: {self.architecture}")
        if self.ffn_type not in FFN_REGISTRY:
            if "moe" in self.ffn_type.lower():
                raise ValueError(f"MoE FFN 尚未实现: {self.ffn_type}")
            raise ValueError(f"未知 FFN 类型: {self.ffn_type}")
        for key in ("vocab_size", "context", "d_model", "heads", "target_parameters"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} 必须是正整数")
        if self.vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(f"vocab_size 至少为 {len(SPECIAL_TOKENS)}")
        if self.d_model % self.heads:
            raise ValueError("d_model 必须能被 heads 整除")
        if not isinstance(self.ffn_dims, list) or len(self.ffn_dims) != 2:
            raise ValueError("ffn_dims 必须是两层正整数列表")
        if any(type(width) is not int or width <= 0 for width in self.ffn_dims):
            raise ValueError("ffn_dims 必须是两层正整数列表")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["ffn_dims"] = list(self.ffn_dims)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelConfig:
        if not isinstance(data, Mapping):
            raise TypeError("模型配置必须是映射")
        allowed = {
            "vocab_size", "context", "d_model", "heads", "ffn_dims",
            "architecture", "ffn_type", "target_parameters",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"未知模型配置字段: {', '.join(sorted(unknown))}")
        required = {"vocab_size", "context", "d_model", "heads", "ffn_dims"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"缺少模型配置字段: {', '.join(sorted(missing))}")
        return cls(**dict(data))


def _parameter_count_for(
    vocab_size: int,
    context: int,
    d_model: int,
    ffn_dims: Sequence[int],
) -> int:
    """Count unique trainable values for the tied, two-layer dense model."""
    if len(ffn_dims) != 2:
        raise ValueError("当前 dense 架构固定为两层")
    # Tied token/output matrix is counted once; include its vocabulary bias.
    # Each block has QKV and output projections with biases, two affine
    # LayerNorms, and two biased FFN linears; the model ends with affine LN.
    return (
        vocab_size * (d_model + 1)
        + context * d_model
        + 8 * d_model * d_model
        + 20 * d_model
        + (2 * d_model + 1) * sum(ffn_dims)
    )


def _candidate_architectures() -> tuple[tuple[int, int], ...]:
    # Prefer a conventional 48-wide, six-head model for ordinary corpora;
    # larger widths let the same exact-budget builder support the 1M variant.
    return (
        (48, 6), (56, 7), (40, 5), (32, 4), (64, 8), (49, 7), (24, 4),
        (128, 8), (160, 8), (192, 8), (256, 8),
    )


def solve_config(
    vocab_size: int,
    target: int = 100_000,
    min_context: int = 96,
    max_context: int = 512,
) -> ModelConfig:
    """Find an exact two-layer dense configuration without spare parameters."""
    for name, value in (("vocab_size", vocab_size), ("target", target), ("min_context", min_context), ("max_context", max_context)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size 至少为 {len(SPECIAL_TOKENS)}")
    if min_context > max_context:
        raise ValueError("min_context 不能大于 max_context")

    for d_model, heads in _candidate_architectures():
        if d_model % heads:
            continue
        coefficient = 2 * d_model + 1
        for context in range(min_context, max_context + 1):
            fixed = (
                vocab_size * (d_model + 1)
                + context * d_model
                + 8 * d_model * d_model
                + 20 * d_model
            )
            remainder = target - fixed
            if remainder <= 0 or remainder % coefficient:
                continue
            total_ffn = remainder // coefficient
            first = total_ffn // 2
            second = total_ffn - first
            if first < d_model or second < d_model:
                continue
            if max(first, second) > 8 * d_model:
                continue
            config = ModelConfig(
                vocab_size=vocab_size,
                context=context,
                d_model=d_model,
                heads=heads,
                ffn_dims=[first, second],
                target_parameters=target,
            )
            assert _parameter_count_for(vocab_size, context, d_model, config.ffn_dims) == target
            return config
    raise ValueError(
        f"无法在 {min_context}–{max_context} 上下文内为词表 {vocab_size} 找到 "
        f"FFN 宽度不超过 8×d_model 的精确 {target} 参数配置"
    )


class _DenseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim, bias=True)
        self.activation = nn.GELU()
        self.down = nn.Linear(hidden_dim, d_model, bias=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.down(self.activation(self.up(value)))


def _make_ffn(ffn_type: str, d_model: int, hidden_dim: int) -> nn.Module:
    try:
        factory = FFN_REGISTRY[ffn_type]
    except KeyError as error:
        if "moe" in ffn_type.lower():
            raise ValueError(f"MoE FFN 尚未实现: {ffn_type}") from error
        raise ValueError(f"未知 FFN 类型: {ffn_type}") from error
    return factory(d_model, hidden_dim)


class _DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, hidden_dim: int):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=True)
        self.ln1 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.ffn = _make_ffn(config.ffn_type, config.d_model, hidden_dim)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor | None]:
        batch, length, width = value.shape
        qkv = self.qkv(self.ln1(value)).view(
            batch, length, 3, self.heads, self.head_dim
        )
        query, key, val = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        value = value + self.attn_out(attention)

        ffn_result = self.ffn(self.ln2(value))
        auxiliary_loss: Tensor | None = None
        if isinstance(ffn_result, tuple):
            if len(ffn_result) != 2:
                raise ValueError("FFN tuple 返回值必须是 (hidden, aux_loss)")
            hidden, auxiliary_loss = ffn_result
            if auxiliary_loss is not None and not isinstance(auxiliary_loss, Tensor):
                raise ValueError("FFN aux_loss 必须是 Tensor 或 None")
        else:
            hidden = ffn_result
        if not isinstance(hidden, Tensor) or hidden.shape != value.shape:
            raise ValueError("FFN 必须返回与输入同形状的 hidden tensor")
        return value + hidden, auxiliary_loss


class _DenseTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.ffn_type not in FFN_REGISTRY:
            raise ValueError(f"未知 FFN 类型: {config.ffn_type}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        self.blocks = nn.ModuleList(
            _DecoderBlock(config, hidden_dim)
            for hidden_dim in config.ffn_dims
        )
        self.final_ln = nn.LayerNorm(config.d_model, elementwise_affine=True)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.aux_loss: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Keep embedding/output scale moderate at initialization; all modules
        # are real forward-path parameters, including every position row.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_bias)

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("ids 必须是非空的 [B,T] token ID 张量")
        if ids.shape[1] > self.config.context:
            raise ValueError("输入长度超过模型 context")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("ids 必须是整数 tensor")
        precision_guard = (
            torch.autocast(device_type=ids.device.type, enabled=False)
            if ids.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with precision_guard:
            positions = torch.arange(ids.shape[1], device=ids.device)
            value = self.token_embedding(ids) + self.position_embedding(positions)[None, :, :]
            auxiliary_losses: list[Tensor] = []
            for block in self.blocks:
                value, auxiliary_loss = block(value)
                if auxiliary_loss is not None:
                    auxiliary_losses.append(auxiliary_loss)
            self.aux_loss = torch.stack(auxiliary_losses).sum() if auxiliary_losses else None
            value = self.final_ln(value)
            return F.linear(value, self.lm_head.weight, self.lm_bias)


def _build_dense_transformer(config: ModelConfig) -> nn.Module:
    return _DenseTransformer(config)


ARCHITECTURE_REGISTRY["dense_transformer_v1"] = _build_dense_transformer


def build_model(config: ModelConfig) -> nn.Module:
    if not isinstance(config, ModelConfig):
        raise TypeError("config 必须是 ModelConfig")
    try:
        factory = ARCHITECTURE_REGISTRY[config.architecture]
    except KeyError as error:
        if config.architecture.startswith("moe"):
            raise ValueError(f"MoE 架构尚未实现: {config.architecture}") from error
        raise ValueError(f"未知模型架构: {config.architecture}") from error
    model = factory(config)
    # Keep this module's checkpoint dtype explicit; callers may move devices,
    # but must opt in if they later want a non-FP32 representation.
    model.to(dtype=torch.float32)
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != config.target_parameters:
        raise ValueError(
            f"配置声明 {config.target_parameters} 个参数，实际模型有 {actual} 个"
        )
    return model


def greedy_tokens(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_id: int,
) -> list[int]:
    """Append deterministic full-vocabulary argmax tokens until EOS or limit."""
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("prompt_ids 必须是非空列表")
    if any(type(token_id) is not int or token_id < 0 for token_id in prompt_ids):
        raise ValueError("prompt_ids 必须只包含非负整数")
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须是非负整数")
    if type(eos_id) is not int or eos_id < 0:
        raise ValueError("eos_id 必须是非负整数")

    config = getattr(model, "config", None)
    context = getattr(config, "context", None)
    vocab_size = getattr(config, "vocab_size", None)
    if context is not None and (type(context) is not int or context <= 0):
        raise ValueError("model.config.context 无效")
    if vocab_size is not None:
        if eos_id >= vocab_size or any(token_id >= vocab_size for token_id in prompt_ids):
            raise ValueError("prompt 或 eos_id 超出模型词表范围")
    if eos_id in prompt_ids:
        return list(prompt_ids)

    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("model 必须至少包含一个参数")
    device = parameters[0].device
    result = list(prompt_ids)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                active = result[-context:] if context is not None else result
                ids = torch.tensor([active], dtype=torch.long, device=device)
                logits = model(ids)
                if not isinstance(logits, Tensor) or logits.ndim != 3:
                    raise ValueError("model.forward 必须返回 [B,T,V] logits")
                next_id = int(logits[0, -1].argmax(dim=-1).item())
                result.append(next_id)
                if next_id == eos_id:
                    break
    finally:
        model.train(was_training)
    return result
