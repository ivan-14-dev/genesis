"""Optimisation utilities: AdamW, global-norm gradient clipping and a cosine schedule.

Weight decay is decoupled (AdamW) and applied only to 2-D weight matrices;
biases, LayerNorm parameters and other 1-D tensors are never decayed. Updates
are written into the existing parameter buffers so references stay valid.
"""

from __future__ import annotations

import math
import operator
from array import array

from .tensor import Tensor

_dot = getattr(math, "sumprod", None) or (lambda x, y: sum(map(operator.mul, x, y)))


class AdamW:
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        self.params: list[Tensor] = list(params)
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("invalid AdamW hyper-parameters")
        self.lr, self.betas, self.eps, self.weight_decay = lr, tuple(betas), eps, weight_decay
        self.t = 0
        self.m = [array("d", bytes(8 * len(p.data))) for p in self.params]
        self.v = [array("d", bytes(8 * len(p.data))) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        c1, c2 = 1.0 - b1, 1.0 - b2
        lr, eps, sqrt = self.lr, self.eps, math.sqrt
        step_size = lr / (1.0 - b1 ** self.t)
        inv_bc2 = 1.0 / (1.0 - b2 ** self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if g is None:
                continue
            m[:] = array("d", [b1 * mi + c1 * gi for mi, gi in zip(m, g)])
            v[:] = array("d", [b2 * vi + c2 * gi * gi for vi, gi in zip(v, g)])
            keep = 1.0 - lr * self.weight_decay if len(p.shape) == 2 else 1.0
            p.data[:] = array("d", [w * keep - step_size * mi / (sqrt(vi * inv_bc2) + eps)
                                    for w, mi, vi in zip(p.data, m, v)])


def clip_grad_norm(params, max_norm: float) -> float:
    """Scale all grads so their global L2 norm is <= ``max_norm``; returns the pre-clip norm."""
    params = [p for p in params if p.grad is not None]
    total = math.sqrt(sum(_dot(p.grad, p.grad) for p in params))
    if max_norm > 0 and total > max_norm:
        s = max_norm / (total + 1e-6)
        for p in params:
            p.grad = array("d", [g * s for g in p.grad])
    return total


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0, min_ratio: float = 0.1) -> float:
    """Linear warmup over ``warmup`` steps, then cosine decay to ``min_ratio * base_lr``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
