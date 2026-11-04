"""Sampling utilities supporting greedy argmax, temperature, top-p, and repetition penalty."""

from __future__ import annotations

from typing import Sequence
import numpy as np


def sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    history: Sequence[int] | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample or argmax next token from logits array of shape (vocab_size,)."""
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    elif hasattr(logits, "__array__"):
        logits = np.asarray(logits)

    if logits.ndim != 1:
        logits = logits.squeeze()
        if logits.ndim != 1:
            raise ValueError(f"logits 必须是一维向量，实际形状为 {logits.shape}")

    scores = logits.astype(np.float64, copy=True)

    # 1. Repetition penalty
    if history and repetition_penalty > 1.0:
        seen = set(history)
        for token_id in seen:
            if 0 <= token_id < len(scores):
                if scores[token_id] > 0:
                    scores[token_id] /= repetition_penalty
                else:
                    scores[token_id] *= repetition_penalty

    # 2. Greedy if temperature <= 0
    if temperature <= 1e-6:
        return int(np.argmax(scores))

    # 3. Temperature scaling
    scores = scores / temperature

    # 4. Top-p (nucleus) filtering
    sorted_indices = np.argsort(scores)[::-1]
    sorted_logits = scores[sorted_indices]
    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
    probs = exp_logits / np.sum(exp_logits)
    cum_probs = np.cumsum(probs)

    cutoff_idx = int(np.searchsorted(cum_probs, top_p))
    valid_indices = sorted_indices[:cutoff_idx + 1]
    valid_probs = probs[:cutoff_idx + 1]
    prob_sum = np.sum(valid_probs)
    if prob_sum > 0:
        valid_probs = valid_probs / prob_sum
    else:
        return int(sorted_indices[0])

    if rng is None:
        return int(np.random.choice(valid_indices, p=valid_probs))
    return int(rng.choice(valid_indices, p=valid_probs))
