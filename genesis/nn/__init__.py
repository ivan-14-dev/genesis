"""genesis.nn: a from-scratch neural-network toolkit (stdlib only, no pretrained anything).

* :mod:`.backend`   – matmul kernels: pure Python or an in-repo C kernel compiled on demand.
* :mod:`.tensor`    – flat float64 tensors with reverse-mode autograd and fused ops.
* :mod:`.optim`     – AdamW, gradient clipping, cosine learning-rate schedule.
* :mod:`.models`    – ``MLPClassifier`` and ``TinyGPT`` (decoder-only Transformer).
* :mod:`.serialize` – pickle-free, validated weight format (``GNNW1``).
* :mod:`.train`     – deterministic training / evaluation loops.

Every weight is initialised from ``random.Random(seed)`` and trained here.
"""

from __future__ import annotations

from . import backend, models, optim, serialize, tensor, train
from .backend import backend_info, backend_name, benchmark_matmul, set_backend
from .models import MLPClassifier, TinyGPT
from .optim import AdamW, clip_grad_norm, cosine_lr
from .serialize import digest, dumps, loads
from .tensor import Tensor, no_grad, randn, softmax_list, tensor_from_list, zeros
from .train import evaluate_classifier, evaluate_lm, train_classifier, train_lm

__all__ = ["backend", "models", "optim", "serialize", "tensor", "train", "backend_info", "backend_name",
           "benchmark_matmul", "set_backend", "MLPClassifier", "TinyGPT", "AdamW", "clip_grad_norm",
           "cosine_lr", "digest", "dumps", "loads", "Tensor", "no_grad", "randn", "softmax_list",
           "tensor_from_list", "zeros", "evaluate_classifier", "evaluate_lm", "train_classifier", "train_lm"]
