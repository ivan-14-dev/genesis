"""Minimal reverse-mode autograd on flat float64 buffers (stdlib only).

A :class:`Tensor` is a row-major ``array('d')`` plus a 1-D/2-D ``shape`` (``()``
for scalars). Every op computes its forward result eagerly and, when grad mode
is on and an input requires grad, records a closure mapping the output
gradient to input gradients. :meth:`Tensor.backward` walks the graph in
reverse topological order (iteratively, so deep graphs never hit the recursion
limit), sums gradients of tensors used several times, and accumulates the
result into ``.grad`` of leaf tensors only. Closures capture buffers, never
their output tensor, so graphs are freed by reference counting.

Heavy ops (embedding, bias add, GELU, layer norm, causal multi-head attention,
cross-entropy) are fused with hand-written backward passes, and their numeric
loops, like every matrix product and the gradient accumulation in
:meth:`Tensor.backward`, run through the kernel set of the active backend
(:func:`genesis.nn.backend.ops`: in-repo C library or its pure-Python twin).
An op's backward closure keeps the kernel set used by its forward pass.
"""

from __future__ import annotations

import builtins
import math
import operator
from array import array
from contextlib import contextmanager

from . import backend

_sum = builtins.sum
_mul = operator.mul
_grad_enabled = True


@contextmanager
def no_grad():
    """Disable graph construction (inference)."""
    global _grad_enabled
    prev, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled() -> bool:
    return _grad_enabled


def _zeros(n: int) -> array:
    return array("d", [0.0]) * n


def _prod(shape) -> int:
    r = 1
    for s in shape:
        r *= s
    return r


class Tensor:
    __slots__ = ("data", "shape", "grad", "requires_grad", "_parents", "_backward")

    def __init__(self, data, shape=None, requires_grad: bool = False):
        buf = data if isinstance(data, array) and data.typecode == "d" else array("d", data)
        shape = (len(buf),) if shape is None else tuple(int(s) for s in shape)
        if len(shape) > 2 or any(s < 0 for s in shape) or _prod(shape) != len(buf):
            raise ValueError(f"shape {shape} does not match {len(buf)} values (max 2-D)")
        self.data, self.shape, self.grad = buf, shape, None
        self.requires_grad = bool(requires_grad)
        self._parents, self._backward = (), None

    @property
    def size(self) -> int:
        return len(self.data)

    def item(self) -> float:
        if len(self.data) != 1:
            raise ValueError("item() needs a single-element tensor")
        return self.data[0]

    def tolist(self) -> list:
        if len(self.shape) == 2:
            c = self.shape[1]
            return [list(self.data[i * c:(i + 1) * c]) for i in range(self.shape[0])]
        return list(self.data)

    def detach(self) -> "Tensor":
        return Tensor(array("d", self.data), self.shape)

    def zero_grad(self) -> None:
        self.grad = None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return scale(self, other) if isinstance(other, (int, float)) else mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def relu(self):
        return relu(self)

    def tanh(self):
        return tanh(self)

    def gelu(self):
        return gelu(self)

    def sum(self):
        return sum(self)

    def mean(self):
        return mean(self)

    def backward(self, grad=None) -> None:
        """Backpropagate from this tensor; ``grad`` defaults to 1.0 for scalars."""
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        if grad is None:
            if len(self.data) != 1:
                raise ValueError("backward() needs an explicit grad for non-scalar tensors")
            seed = array("d", [1.0])
        else:
            seed = array("d", [float(grad)] * len(self.data)) if isinstance(grad, (int, float)) else array("d", grad)
            if len(seed) != len(self.data):
                raise ValueError("grad has the wrong size")
        order, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative post-order DFS: inputs before outputs
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        # id -> [gradient buffer, owned]; an owned buffer was allocated here and is referenced nowhere
        # else, so it can be accumulated into in place (closures may return the same buffer twice).
        K = backend.ops()
        grads = {id(self): [seed, True]}
        for node in reversed(order):
            entry = grads.pop(id(node), None)
            if entry is None:
                continue
            g, owned = entry
            if node._backward is None:
                if node.grad is None:
                    node.grad = g if owned else array("d", g)
                else:
                    node.grad = K.add(_as_d(node.grad), g)
                continue
            for p, pg in zip(node._parents, node._backward(g)):
                if pg is None or not p.requires_grad:
                    continue
                prev = grads.get(id(p))
                if prev is None:
                    grads[id(p)] = [pg, False]
                elif prev[1]:
                    K.add_(prev[0], pg)
                else:
                    grads[id(p)] = [K.add(prev[0], pg), True]


def _result(data: array, shape: tuple, parents: tuple, backward_fn) -> Tensor:
    out = Tensor.__new__(Tensor)
    out.data, out.shape, out.grad = data, shape, None
    if _grad_enabled and any(p.requires_grad for p in parents):
        out.requires_grad, out._parents, out._backward = True, parents, backward_fn
    else:
        out.requires_grad, out._parents, out._backward = False, (), None
    return out


def _need(*ts) -> bool:
    return _grad_enabled and any(t.requires_grad for t in ts)


def _as_d(buf) -> array:
    return buf if type(buf) is array and buf.typecode == "d" else array("d", buf)


# ---------------------------------------------------------------- factories
def zeros(shape, requires_grad: bool = False) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor(_zeros(_prod(shape)), shape, requires_grad)


def tensor_from_list(values, shape=None, requires_grad: bool = False) -> Tensor:
    """Build from a flat list (optionally with ``shape``) or a rectangular list of lists."""
    values = list(values)
    if values and isinstance(values[0], (list, tuple)):
        cols = len(values[0])
        if any(len(r) != cols for r in values):
            raise ValueError("ragged nested list")
        flat = array("d", [float(v) for r in values for v in r])
        return Tensor(flat, shape or (len(values), cols), requires_grad)
    return Tensor(array("d", [float(v) for v in values]), shape, requires_grad)


def randn(shape, rng, std: float = 1.0, requires_grad: bool = False) -> Tensor:
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    return Tensor(array("d", [rng.gauss(0.0, std) for _ in range(_prod(shape))]), shape, requires_grad)


# ---------------------------------------------------------------- elementwise / reductions
def add(a: Tensor, b: Tensor) -> Tensor:
    """``a + b`` for equal shapes, or 2-D ``a`` [T,D] plus 1-D bias ``b`` [D] broadcast over rows."""
    K = backend.ops()
    if a.shape == b.shape:
        return _result(K.add(a.data, b.data), a.shape, (a, b), lambda g: (g, g))
    if len(a.shape) == 2 and b.shape == (a.shape[1],):
        T, D = a.shape
        return _result(K.add_rows(a.data, b.data, T, D), a.shape, (a, b),
                       lambda g: (g, K.colsum(g, T, D) if b.requires_grad else None))
    raise ValueError(f"add: incompatible shapes {a.shape} and {b.shape}")


def mul(a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(f"mul: shape mismatch {a.shape} vs {b.shape}")
    ad, bd = a.data, b.data

    def bw(g):
        return (array("d", map(_mul, g, bd)) if a.requires_grad else None,
                array("d", map(_mul, g, ad)) if b.requires_grad else None)
    return _result(array("d", map(_mul, ad, bd)), a.shape, (a, b), bw)


def scale(a: Tensor, s: float) -> Tensor:
    s = float(s)
    return _result(array("d", [v * s for v in a.data]), a.shape, (a,), lambda g: (array("d", [v * s for v in g]),))


def relu(a: Tensor) -> Tensor:
    ad = a.data
    return _result(array("d", [v if v > 0.0 else 0.0 for v in ad]), a.shape, (a,),
                   lambda g: (array("d", [gv if v > 0.0 else 0.0 for gv, v in zip(g, ad)]),))


def tanh(a: Tensor) -> Tensor:
    t = math.tanh
    y = array("d", [t(v) for v in a.data])
    return _result(y, a.shape, (a,), lambda g: (array("d", [gv * (1.0 - v * v) for gv, v in zip(g, y)]),))


def gelu(a: Tensor) -> Tensor:
    """GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
    K, ad = backend.ops(), a.data
    y, th = K.gelu(ad)
    return _result(y, a.shape, (a,), lambda g: (K.gelu_backward(g, ad, th),))


def sum(a: Tensor) -> Tensor:  # noqa: A001 - tensor op mirrors the builtin's name
    n = len(a.data)
    return _result(array("d", [_sum(a.data)]), (), (a,), lambda g: (array("d", [g[0]]) * n,))


def mean(a: Tensor) -> Tensor:
    n = len(a.data)
    if n == 0:
        raise ValueError("mean of an empty tensor")
    return _result(array("d", [_sum(a.data) / n]), (), (a,), lambda g: (array("d", [g[0] / n]) * n,))


def mean_rows(x: Tensor) -> Tensor:
    """[T,D] -> [1,D] mean over rows (bag-of-tokens pooling)."""
    if len(x.shape) != 2 or x.shape[0] == 0:
        raise ValueError("mean_rows expects a non-empty 2-D tensor")
    T, D = x.shape
    return _result(array("d", [v / T for v in backend.ops().colsum(x.data, T, D)]), (1, D), (x,),
                   lambda g: (array("d", [v / T for v in g]) * T,))


# ---------------------------------------------------------------- linear algebra
def matmul(a: Tensor, b: Tensor, trans_b: bool = False) -> Tensor:
    """``a @ b`` for 2-D tensors; with ``trans_b`` computes ``a @ b.T`` (b stored [N,K])."""
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul expects 2-D tensors")
    m, k = a.shape
    n, kb = (b.shape[0], b.shape[1]) if trans_b else (b.shape[1], b.shape[0])
    if k != kb:
        raise ValueError(f"matmul: shapes {a.shape} and {b.shape} (trans_b={trans_b}) do not align")
    ad, bd, mm = a.data, b.data, backend.ops().matmul

    def bw(g):
        if trans_b:
            return (mm(g, bd, m, n, k) if a.requires_grad else None,
                    mm(g, ad, n, m, k, True) if b.requires_grad else None)
        return (mm(g, bd, m, n, k, False, True) if a.requires_grad else None,
                mm(ad, g, k, m, n, True) if b.requires_grad else None)
    return _result(mm(ad, bd, m, k, n, False, trans_b), (m, n), (a, b), bw)


def embedding(weight: Tensor, ids) -> Tensor:
    """Gather rows of ``weight`` [V,D] -> [T,D]; backward scatter-adds into the rows used."""
    if len(weight.shape) != 2:
        raise ValueError("embedding weight must be 2-D")
    V, D = weight.shape
    ids = list(ids)
    for i in ids:
        if type(i) is not int or not 0 <= i < V:
            raise ValueError(f"embedding id {i!r} out of range [0, {V})")
    K, idx = backend.ops(), array("i", ids)
    return _result(K.embedding(weight.data, idx, V, D), (len(ids), D), (weight,),
                   lambda g: (K.embedding_backward(g, idx, V, D),))


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    """Row-wise LayerNorm over the last dim of [T,D] with affine ``gamma``/``beta`` [D] (fused)."""
    if len(x.shape) != 2 or gamma.shape != (x.shape[1],) or beta.shape != (x.shape[1],):
        raise ValueError("layer_norm: expected x [T,D], gamma [D], beta [D]")
    T, D = x.shape
    K, gd = backend.ops(), gamma.data
    out, xhat, rstd = K.layer_norm(x.data, gd, beta.data, T, D, eps)
    return _result(out, (T, D), (x, gamma, beta), lambda g: K.layer_norm_backward(
        g, xhat, rstd, gd, T, D, x.requires_grad, gamma.requires_grad, beta.requires_grad))


def causal_self_attention(x: Tensor, w_qkv: Tensor, b_qkv: Tensor, w_o: Tensor, b_o: Tensor,
                          n_heads: int) -> Tensor:
    """Fused multi-head causal self-attention on x [T,D].

    qkv = x @ w_qkv + b_qkv laid out [Q | K | V] (heads contiguous inside each);
    per head P = softmax(Q K^T / sqrt(hd)) with position t attending to u <= t;
    out = concat_h(P V) @ w_o + b_o.
    """
    if len(x.shape) != 2:
        raise ValueError("attention input must be 2-D [T,D]")
    T, D = x.shape
    if (w_qkv.shape != (D, 3 * D) or b_qkv.shape != (3 * D,) or w_o.shape != (D, D) or b_o.shape != (D,)
            or n_heads <= 0 or D % n_heads):
        raise ValueError("causal_self_attention: inconsistent shapes or n_heads")
    H, D3, K = n_heads, 3 * D, backend.ops()
    mm, xd, wq, wo = K.matmul, x.data, w_qkv.data, w_o.data
    qkv = K.add_rows(mm(xd, wq, T, D, D3), b_qkv.data, T, D3)
    y, P = K.attention(qkv, T, D, H)
    out = K.add_rows(mm(y, wo, T, D, D), b_o.data, T, D)

    def bw(g):
        dw_o = mm(y, g, D, T, D, True) if w_o.requires_grad else None
        db_o = K.colsum(g, T, D) if b_o.requires_grad else None
        if not (x.requires_grad or w_qkv.requires_grad or b_qkv.requires_grad):
            return None, None, None, dw_o, db_o
        dqkv = K.attention_backward(qkv, P, mm(g, wo, T, D, D, False, True), T, D, H)
        return (mm(dqkv, wq, T, D3, D, False, True) if x.requires_grad else None,
                mm(xd, dqkv, D, T, D3, True) if w_qkv.requires_grad else None,
                K.colsum(dqkv, T, D3) if b_qkv.requires_grad else None, dw_o, db_o)
    return _result(out, (T, D), (x, w_qkv, b_qkv, w_o, b_o), bw)


def cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean negative log-likelihood of ``targets`` under row-wise softmax of logits [T,V] (fused)."""
    if len(logits.shape) != 2:
        raise ValueError("cross_entropy expects 2-D logits [T,V]")
    T, V = logits.shape
    targets = list(targets)
    if len(targets) != T or T == 0:
        raise ValueError(f"cross_entropy: need {T} > 0 targets, got {len(targets)}")
    for y in targets:
        if type(y) is not int or not 0 <= y < V:
            raise ValueError(f"target {y!r} out of range [0, {V})")
    K, tg = backend.ops(), array("i", targets)
    total, probs = K.cross_entropy(logits.data, tg, T, V)
    return _result(array("d", [total / T]), (), (logits,),
                   lambda g: (K.cross_entropy_backward(probs, tg, T, V, g[0] / T),))


def softmax_list(xs, temperature: float = 1.0) -> list[float]:
    """Plain-float softmax; ``temperature <= 0`` gives a one-hot on the (first) argmax."""
    xs = list(xs)
    if not xs:
        return []
    mx = max(xs)
    if temperature <= 0:
        i = xs.index(mx)
        return [1.0 if j == i else 0.0 for j in range(len(xs))]
    e = [math.exp((v - mx) / temperature) for v in xs]
    z = _sum(e)
    return [v / z for v in e]
