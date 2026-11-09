"""Optimisation utilities: AdamW, global-norm gradient clipping and a cosine schedule.

Weight decay is decoupled (AdamW) and applied only to 2-D weight matrices;
biases, LayerNorm parameters and other 1-D tensors are never decayed. Updates
are written into the existing parameter buffers so references stay valid. The
per-element update and the gradient norm run in the active backend's kernels
(``adamw``, ``sumsq``, ``scale`` of :func:`genesis.nn.backend.ops`).
"""

from __future__ import annotations

import math
from array import array

from . import backend
from .tensor import Tensor


def _as_d(buf) -> array:
    return buf if type(buf) is array and buf.typecode == "d" else array("d", buf)


class AdamW:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params: list[Tensor] = list(params)
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("invalid AdamW hyper-parameters")
        self.lr, self.betas, self.eps, self.weight_decay = lr, tuple(betas), eps, weight_decay
        self.t = 0
        self.m = [array("d", [0.0]) * len(p.data) for p in self.params]
        self.v = [array("d", [0.0]) * len(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        lr, eps, K = self.lr, self.eps, backend.ops()
        step_size = lr / (1.0 - b1 ** self.t)
        inv_bc2 = 1.0 / (1.0 - b2 ** self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            keep = 1.0 - lr * self.weight_decay if len(p.shape) == 2 else 1.0
            K.adamw(p.data, _as_d(p.grad), m, v, b1, b2, step_size, inv_bc2, eps, keep)


def clip_grad_norm(params, max_norm: float) -> float:
    """Scale all grads so their global L2 norm is <= ``max_norm``; returns the pre-clip norm."""
    params, K = [p for p in params if p.grad is not None], backend.ops()
    total = math.sqrt(sum(K.sumsq(_as_d(p.grad)) for p in params))
    if max_norm > 0 and total > max_norm:
        s = max_norm / (total + 1e-6)
        for p in params:
            p.grad = K.scale(_as_d(p.grad), s)
    return total


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0, min_ratio: float = 0.1) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to ``min_ratio * base_lr``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
