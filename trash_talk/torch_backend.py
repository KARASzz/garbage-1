"""Canonical-weight adapter for CPU reference and future CUDA hosts."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from .garbage_model import build_model, greedy_tokens


class TorchBackend:
    def __init__(self, config, weights, *, device="cuda"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；没有自动改用其他设备")
        if device not in ("cpu", "cuda"):
            raise ValueError("Torch 兼容后端仅接受 cpu 或 cuda")
        self.device = torch.device(device)
        self.model = build_model(config).to(self.device)
        expected = dict(self.model.named_parameters())
        if set(weights) != set(expected):
            raise ValueError("checkpoint 的 canonical 参数名与架构不匹配")
        with torch.no_grad():
            for name, parameter in expected.items():
                array = np.asarray(weights[name])
                if array.dtype != np.float32 or tuple(array.shape) != tuple(parameter.shape) or not np.isfinite(array).all():
                    raise ValueError(f"参数格式无效: {name}")
                parameter.copy_(torch.from_numpy(array.copy()).to(self.device))
        self.model.eval()

    def generate(self, prompt_ids, max_new_tokens, eos_id):
        return greedy_tokens(self.model, prompt_ids, max_new_tokens, eos_id)

    def logits(self, ids):
        with torch.no_grad():
            return self.model(torch.tensor(ids, dtype=torch.long, device=self.device))

    def train_step(self, x, y, learning_rate=0.005, pad_id=0):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        ids = torch.tensor(x, dtype=torch.long, device=self.device)
        targets = torch.tensor(y, dtype=torch.long, device=self.device)
        scores = self.model(ids)
        if not bool((targets != pad_id).any()):
            self.model.eval()
            return 0.0
        loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), targets.reshape(-1), ignore_index=pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(parameter.grad, alpha=-learning_rate)
        self.model.eval()
        return float(loss.detach())

    def export_weights(self):
        return {name: parameter.detach().cpu().numpy().copy() for name, parameter in self.model.named_parameters()}

    def weight_digest(self):
        digest = hashlib.sha256()
        for name, array in sorted(self.export_weights().items()):
            digest.update(array.astype("<f4", copy=False).tobytes())
        return digest.hexdigest()
