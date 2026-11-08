"""Compute backends for :mod:`genesis.nn`: pure Python and an in-repo C library.

Every numeric kernel used by the tensor ops and the optimiser exists twice, with the same
signature and semantics: a pure-Python version (always available) and a :mod:`ctypes` wrapper
around ``_kernels.c``. :func:`ops` returns the active set (a :class:`Kernels` namespace):

* ``matmul`` – dense ``op(A) @ op(B)`` (optional transposes / accumulation);
* ``add``, ``add_`` (in place), ``add_rows`` (bias broadcast), ``colsum`` (its reduction),
  ``scale``, ``sumsq`` (gradient norm);
* ``gelu`` / ``gelu_backward``, ``layer_norm`` / ``layer_norm_backward``;
* ``attention`` / ``attention_backward`` – all heads of causal softmax attention on a packed
  ``[T, 3D]`` QKV buffer (the C version only computes the causal blocks);
* ``cross_entropy`` / ``cross_entropy_backward``, ``embedding`` / ``embedding_backward``;
* ``adamw`` – in-place AdamW update.

Everything here is built from this repository: the C backend compiles only the shipped
``_kernels.c`` through :mod:`genesis.native` (private cache named after the SHA-256 of source and
flags, atomic rename, ownership checks), then declares the signatures and self-tests every kernel
against its Python twin before use. If anything fails, ``auto`` mode silently falls back to pure
Python and records the reason in :func:`backend_info`. The C functions are called with the GIL
held and every buffer's type and length is checked first, so C never sees a resized or short
buffer.

Selection: env ``GENESIS_NN_BACKEND`` in {``auto`` (default), ``python``, ``c``} or
:func:`set_backend`. The C matrix product uses a packed, register-tiled micro-kernel (AVX when
the CPU has it) and splits products of at least ~2M multiply-adds over ``GENESIS_NN_THREADS``
threads (default ``min(2, cpu_count)``, see :func:`set_threads`; 1 disables threading). Both backends are deterministic, and C results
do not depend on the SIMD width or the thread count (each output element is computed by one
thread with a fixed operation order); the two backends may differ from each other in the last
bits because Python's ``sum``/``sumprod`` round differently.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import operator
import os
import random
import time
from array import array

from .. import native

__all__ = ["Kernels", "ops", "kernels", "matmul", "set_backend", "set_threads", "backend_name", "backend_info",
           "kernel_source_digest", "available_backends", "benchmark_matmul"]

KERNEL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kernels.c")
_BACKENDS = ("auto", "python", "c")
_CFLAGS = (*native.BASE_CFLAGS, "-pthread", "-lm")
GELU_C, GELU_K = math.sqrt(2.0 / math.pi), 0.044715

_sum = sum
_mul, _add = operator.mul, operator.add
_dot = getattr(math, "sumprod", None) or (lambda x, y: _sum(map(_mul, x, y)))
_ZERO = array("d", [0.0])

_requested: str | None = None   # None: read GENESIS_NN_BACKEND on first use
_active: str | None = None
_ops = None
_threads: int | None = None
_c: dict = {"tried": False, "ops": None, "compiler": None, "library": None, "failure": None, "simd": None}


class Kernels:
    """Namespace of kernel functions for one backend (attribute names listed in the module doc)."""

    def __init__(self, name: str, **fns):
        self.name = name
        self.__dict__.update(fns)

    def __repr__(self) -> str:
        return f"<genesis.nn kernels {self.name!r}>"


def _zeros(n: int) -> array:
    return _ZERO * n


# ---------------------------------------------------------------- pure Python
def _matmul_python(a, b, m, k, n, ta=False, tb=False, out=None, acc=False):
    rows = [a[i::m] for i in range(m)] if ta else [a[i * k:(i + 1) * k] for i in range(m)]
    cols = [b[j * k:(j + 1) * k] for j in range(n)] if tb else [b[j::n] for j in range(n)]
    dot = _dot
    res = array("d", [dot(r, c) for r in rows for c in cols])
    if out is None:
        return res
    out[:] = array("d", map(_add, out, res)) if acc else res
    return out


def _py_add(a, b):
    return array("d", map(_add, a, b))


def _py_add_(dst, src):
    dst[:] = array("d", map(_add, dst, src))


def _py_add_rows(x, bias, T, D):
    return array("d", map(_add, x, bias * T))


def _py_colsum(g, T, D):
    return array("d", [_sum(g[j::D]) for j in range(D)]) if T else _zeros(D)


def _py_scale(a, s):
    return array("d", [v * s for v in a])


def _py_sumsq(a):
    return _dot(a, a)


def _py_gelu(x):
    c, k, t = GELU_C, GELU_K, math.tanh
    th = array("d", [t(c * (v + k * v * v * v)) for v in x])
    return array("d", [0.5 * v * (1.0 + h) for v, h in zip(x, th)]), th


def _py_gelu_backward(g, x, th):
    c, k = GELU_C, GELU_K
    return array("d", [gv * (0.5 * (1.0 + h) + 0.5 * v * (1.0 - h * h) * c * (1.0 + 3.0 * k * v * v))
                       for gv, v, h in zip(g, x, th)])


def _py_layer_norm(x, gamma, beta, T, D, eps):
    xhat, rstd, out = array("d"), array("d"), array("d")
    for t in range(T):
        row = x[t * D:(t + 1) * D]
        mu = _sum(row) / D
        cen = [v - mu for v in row]
        r = 1.0 / math.sqrt(_dot(cen, cen) / D + eps)
        xh = [c * r for c in cen]
        xhat.extend(xh)
        rstd.append(r)
        out.extend([h * gv + bv for h, gv, bv in zip(xh, gamma, beta)])
    return out, xhat, rstd


def _py_layer_norm_backward(g, xhat, rstd, gamma, T, D, need_dx=True, need_dgamma=True, need_dbeta=True):
    dx = None
    if need_dx:
        dx = array("d")
        for t in range(T):
            xh = xhat[t * D:(t + 1) * D]
            dxh = list(map(_mul, g[t * D:(t + 1) * D], gamma))
            m1, m2, r = _sum(dxh) / D, _dot(dxh, xh) / D, rstd[t]
            dx.extend([r * (d - m1 - h * m2) for d, h in zip(dxh, xh)])
    dgamma = _py_colsum(array("d", map(_mul, g, xhat)), T, D) if need_dgamma else None
    dbeta = _py_colsum(g, T, D) if need_dbeta else None
    return dx, dgamma, dbeta


def _gather(src, rows, stride, off, width):
    out = array("d")
    for t in range(rows):
        s = t * stride + off
        out.extend(src[s:s + width])
    return out


def _scatter(dst, src, rows, stride, off, width):
    for t in range(rows):
        s = t * stride + off
        dst[s:s + width] = src[t * width:(t + 1) * width]


def _py_attention(qkv, T, D, H):
    hd, D3, mm, exp = D // H, 3 * D, _matmul_python, math.exp
    sc = 1.0 / math.sqrt(hd)
    y, P = _zeros(T * D), _zeros(H * T * T)
    for h in range(H):
        Q, K, V = (_gather(qkv, T, D3, off + h * hd, hd) for off in (0, D, 2 * D))
        S, base = mm(Q, K, T, hd, T, False, True), h * T * T
        for t in range(T):
            row = S[t * T:t * T + t + 1]
            mx = max(row)
            e = [exp((s - mx) * sc) for s in row]
            z = _sum(e)
            P[base + t * T:base + t * T + t + 1] = array("d", [v / z for v in e])
        _scatter(y, mm(P[base:base + T * T], V, T, T, hd), T, D, h * hd, hd)
    return y, P


def _py_attention_backward(qkv, P, dy, T, D, H):
    hd, D3, mm = D // H, 3 * D, _matmul_python
    sc = 1.0 / math.sqrt(hd)
    dqkv = _zeros(T * D3)
    for h in range(H):
        Q, K, V = (_gather(qkv, T, D3, off + h * hd, hd) for off in (0, D, 2 * D))
        Ph, dY = P[h * T * T:(h + 1) * T * T], _gather(dy, T, D, h * hd, hd)
        dP, dS = mm(dY, V, T, hd, T, False, True), _zeros(T * T)
        for t in range(T):
            p, dp = Ph[t * T:t * T + t + 1], dP[t * T:t * T + t + 1]
            c = _dot(p, dp)
            dS[t * T:t * T + t + 1] = array("d", [pi * (di - c) * sc for pi, di in zip(p, dp)])
        _scatter(dqkv, mm(dS, K, T, T, hd), T, D3, h * hd, hd)
        _scatter(dqkv, mm(dS, Q, T, T, hd, True), T, D3, D + h * hd, hd)
        _scatter(dqkv, mm(Ph, dY, T, T, hd, True), T, D3, 2 * D + h * hd, hd)
    return dqkv


def _py_cross_entropy(logits, targets, T, V):
    exp, total, probs = math.exp, 0.0, array("d")
    for t, y in enumerate(targets):
        row = logits[t * V:(t + 1) * V]
        mx = max(row)
        e = [exp(v - mx) for v in row]
        z = _sum(e)
        total += mx + math.log(z) - row[y]
        probs.extend([v / z for v in e])
    return total, probs


def _py_cross_entropy_backward(probs, targets, T, V, s):
    d = array("d", [p * s for p in probs])
    for t, y in enumerate(targets):
        d[t * V + y] -= s
    return d


def _py_embedding(w, ids, V, D):
    out = array("d")
    for i in ids:
        out.extend(w[i * D:(i + 1) * D])
    return out


def _py_embedding_backward(g, ids, V, D):
    gw = _zeros(V * D)
    for t, i in enumerate(ids):
        s = i * D
        gw[s:s + D] = array("d", map(_add, gw[s:s + D], g[t * D:(t + 1) * D]))
    return gw


def _py_adamw(w, g, m, v, b1, b2, step_size, inv_bc2, eps, keep):
    c1, c2, sqrt = 1.0 - b1, 1.0 - b2, math.sqrt
    m[:] = array("d", [b1 * mi + c1 * gi for mi, gi in zip(m, g)])
    v[:] = array("d", [b2 * vi + c2 * gi * gi for vi, gi in zip(v, g)])
    w[:] = array("d", [x * keep - step_size * mi / (sqrt(vi * inv_bc2) + eps) for x, mi, vi in zip(w, m, v)])


PYTHON_KERNELS = Kernels(
    "python", matmul=_matmul_python, add=_py_add, add_=_py_add_, add_rows=_py_add_rows, colsum=_py_colsum,
    scale=_py_scale, sumsq=_py_sumsq, gelu=_py_gelu, gelu_backward=_py_gelu_backward, layer_norm=_py_layer_norm,
    layer_norm_backward=_py_layer_norm_backward, attention=_py_attention, attention_backward=_py_attention_backward,
    cross_entropy=_py_cross_entropy, cross_entropy_backward=_py_cross_entropy_backward, embedding=_py_embedding,
    embedding_backward=_py_embedding_backward, adamw=_py_adamw)


# ---------------------------------------------------------------- C
def _ptr(a, n: int) -> int:
    """Address of an ``array('d')`` of exactly ``n`` values (C trusts these sizes)."""
    if type(a) is not array or a.typecode != "d" or len(a) != n:
        raise ValueError(f"expected array('d') of length {n}")
    return a.buffer_info()[0]


_MAX_DIM = 2 ** 31 - 1


def _dims(*ds) -> None:
    """Dimensions handed to C must be ints in [0, 2**31): a pair of negative sizes has a positive product, would
    pass the length checks of ``_ptr`` and become huge ``size_t`` loop bounds in C."""
    for d in ds:
        if type(d) is not int or not 0 <= d <= _MAX_DIM:
            raise ValueError(f"dimension must be an int in [0, {_MAX_DIM}], got {d!r}")


def _iptr(a, n: int) -> int:
    if type(a) is not array or a.typecode != "i" or len(a) != n:
        raise ValueError(f"expected array('i') of length {n}")
    return a.buffer_info()[0]


def _c_kernels(lib) -> Kernels:
    """Declare the C signatures and wrap them with the Python kernel signatures."""
    # PyDLL view of the same handle: calls keep the GIL, so no other thread can resize a buffer mid-call.
    dll = ctypes.PyDLL(lib._name, handle=lib._handle)
    P, L, I, F = ctypes.c_void_p, ctypes.c_long, ctypes.c_int, ctypes.c_double

    def fn(name, argtypes, restype=None):
        try:
            f = getattr(dll, name)
        except AttributeError as exc:
            raise RuntimeError(f"C kernel symbol missing: {name}") from exc
        f.argtypes, f.restype = argtypes, restype
        return f

    c_mm = fn("gn_matmul", [P, P, P] + [I] * 6)
    c_add, c_add_rows = fn("gn_add", [L, P, P, P]), fn("gn_add_rows", [L, L, P, P, P])
    c_colsum, c_scale, c_sumsq = fn("gn_colsum", [L, L, P, P]), fn("gn_scale", [L, P, F, P]), fn("gn_sumsq", [L, P], F)
    c_gelu, c_gelu_bw = fn("gn_gelu_fwd", [L, F, F, P, P, P]), fn("gn_gelu_bwd", [L, F, F, P, P, P, P])
    c_ln, c_ln_bw = fn("gn_layernorm_fwd", [L, L, P, P, P, F, P, P, P]), fn("gn_layernorm_bwd", [L, L] + [P] * 7)
    c_attn, c_attn_bw = fn("gn_attn_fwd", [L, L, L, F, P, P, P]), fn("gn_attn_bwd", [L, L, L, F] + [P] * 5)
    c_xent, c_xent_bw = fn("gn_xent_fwd", [L, L, P, P, P, P], I), fn("gn_xent_bwd", [L, L, P, P, F, P], I)
    c_emb, c_emb_bw = fn("gn_embed_fwd", [L, L, L, P, P, P], I), fn("gn_embed_bwd", [L, L, L, P, P, P], I)
    c_adamw = fn("gn_adamw", [L, P, P, P, P] + [F] * 8)
    _c["set_threads"], _c["set_simd"] = fn("gn_set_threads", [I], I), fn("gn_set_simd", [I], I)

    def matmul(a, b, m, k, n, ta=False, tb=False, out=None, acc=False):
        _dims(m, k, n)
        if out is None:
            out, acc = _zeros(m * n), False
        if m and n:
            c_mm(_ptr(a, m * k), _ptr(b, k * n), _ptr(out, m * n), m, k, n, bool(ta), bool(tb), bool(acc))
        return out

    def add(a, b):
        n = len(a)
        out = _zeros(n)
        c_add(n, _ptr(a, n), _ptr(b, n), out.buffer_info()[0])
        return out

    def add_(dst, src):
        n = len(dst)
        p = _ptr(dst, n)
        c_add(n, p, _ptr(src, n), p)

    def add_rows(x, bias, T, D):
        _dims(T, D)
        out = _zeros(T * D)
        c_add_rows(T, D, _ptr(x, T * D), _ptr(bias, D), out.buffer_info()[0])
        return out

    def colsum(g, T, D):
        _dims(T, D)
        out = _zeros(D)
        c_colsum(T, D, _ptr(g, T * D), out.buffer_info()[0])
        return out

    def scale(a, s):
        n = len(a)
        out = _zeros(n)
        c_scale(n, _ptr(a, n), s, out.buffer_info()[0])
        return out

    def sumsq(a):
        return c_sumsq(len(a), _ptr(a, len(a)))

    def gelu(x):
        n = len(x)
        y, th = _zeros(n), _zeros(n)
        c_gelu(n, GELU_C, GELU_K, _ptr(x, n), y.buffer_info()[0], th.buffer_info()[0])
        return y, th

    def gelu_backward(g, x, th):
        n = len(x)
        dx = _zeros(n)
        c_gelu_bw(n, GELU_C, GELU_K, _ptr(g, n), _ptr(x, n), _ptr(th, n), dx.buffer_info()[0])
        return dx

    def layer_norm(x, gamma, beta, T, D, eps):
        _dims(T, D)
        out, xhat, rstd = _zeros(T * D), _zeros(T * D), _zeros(T)
        c_ln(T, D, _ptr(x, T * D), _ptr(gamma, D), _ptr(beta, D), eps, out.buffer_info()[0],
             xhat.buffer_info()[0], rstd.buffer_info()[0])
        return out, xhat, rstd

    def layer_norm_backward(g, xhat, rstd, gamma, T, D, need_dx=True, need_dgamma=True, need_dbeta=True):
        _dims(T, D)
        dx, dg, db = (_zeros(T * D) if need_dx else None, _zeros(D) if need_dgamma else None,
                      _zeros(D) if need_dbeta else None)
        c_ln_bw(T, D, _ptr(g, T * D), _ptr(xhat, T * D), _ptr(rstd, T), _ptr(gamma, D),
                *(None if b is None else b.buffer_info()[0] for b in (dx, dg, db)))
        return dx, dg, db

    def _check_heads(T, D, H):
        _dims(T, D, H)
        if H <= 0 or D % H:
            raise ValueError("attention: D must be a positive multiple of H")

    def attention(qkv, T, D, H):
        _check_heads(T, D, H)
        y, P = _zeros(T * D), _zeros(H * T * T)
        c_attn(T, D, H, 1.0 / math.sqrt(D // H), _ptr(qkv, 3 * T * D), y.buffer_info()[0], P.buffer_info()[0])
        return y, P

    def attention_backward(qkv, P, dy, T, D, H):
        _check_heads(T, D, H)
        dqkv, scratch = _zeros(3 * T * D), _zeros(T * T)
        c_attn_bw(T, D, H, 1.0 / math.sqrt(D // H), _ptr(qkv, 3 * T * D), _ptr(P, H * T * T), _ptr(dy, T * D),
                  dqkv.buffer_info()[0], scratch.buffer_info()[0])
        return dqkv

    def cross_entropy(logits, targets, T, V):
        _dims(T, V)
        probs, total = _zeros(T * V), _zeros(1)
        if c_xent(T, V, _ptr(logits, T * V), _iptr(targets, T), probs.buffer_info()[0], total.buffer_info()[0]):
            raise ValueError("cross_entropy: target out of range")
        return total[0], probs

    def cross_entropy_backward(probs, targets, T, V, s):
        _dims(T, V)
        d = _zeros(T * V)
        if c_xent_bw(T, V, _ptr(probs, T * V), _iptr(targets, T), s, d.buffer_info()[0]):
            raise ValueError("cross_entropy: target out of range")
        return d

    def embedding(w, ids, V, D):
        T = len(ids)
        _dims(V, D)
        out = _zeros(T * D)
        if c_emb(T, D, V, _ptr(w, V * D), _iptr(ids, T), out.buffer_info()[0]):
            raise ValueError("embedding: id out of range")
        return out

    def embedding_backward(g, ids, V, D):
        T = len(ids)
        _dims(V, D)
        gw = _zeros(V * D)
        if c_emb_bw(T, D, V, _ptr(g, T * D), _iptr(ids, T), gw.buffer_info()[0]):
            raise ValueError("embedding: id out of range")
        return gw

    def adamw(w, g, m, v, b1, b2, step_size, inv_bc2, eps, keep):
        n = len(w)
        c_adamw(n, _ptr(w, n), _ptr(g, n), _ptr(m, n), _ptr(v, n), b1, 1.0 - b1, b2, 1.0 - b2, step_size,
                inv_bc2, eps, keep)

    return Kernels("c", matmul=matmul, add=add, add_=add_, add_rows=add_rows, colsum=colsum, scale=scale,
                   sumsq=sumsq, gelu=gelu, gelu_backward=gelu_backward, layer_norm=layer_norm,
                   layer_norm_backward=layer_norm_backward, attention=attention,
                   attention_backward=attention_backward, cross_entropy=cross_entropy,
                   cross_entropy_backward=cross_entropy_backward, embedding=embedding,
                   embedding_backward=embedding_backward, adamw=adamw)


def _close(x, y, tol: float = 1e-9) -> bool:
    if isinstance(x, float):
        x, y = [x], [y]
    return len(x) == len(y) and all(abs(p - q) <= tol * max(1.0, abs(p)) for p, q in zip(x, y))


def _self_test(c: Kernels) -> None:
    """Compare every C kernel with its Python twin on small random inputs; raise on any mismatch."""
    py, rng = PYTHON_KERNELS, random.Random(20240611)

    def r(n, lo=-2.0, hi=2.0):
        return array("d", [rng.uniform(lo, hi) for _ in range(n)])

    def check(name, got, want):
        if isinstance(want, tuple):
            ok = all(u is None if v is None else _close(u, v) for u, v in zip(got, want))
        else:
            ok = _close(got, want)
        if not ok:
            raise RuntimeError(f"C kernel self-test failed: {name}")

    if list(c.matmul(array("d", [1, 2, 3, 4]), array("d", [5, 6, 7, 8]), 2, 2, 2)) != [19.0, 22.0, 43.0, 50.0]:
        raise RuntimeError("C kernel self-test failed: matmul")
    for m, k, n in ((5, 3, 9), (9, 17, 4), (1, 1, 1)):
        for ta in (False, True):
            for tb in (False, True):
                a, b, o = r(m * k), r(k * n), r(m * n)
                check("matmul", c.matmul(a, b, m, k, n, ta, tb, array("d", o), True),
                      py.matmul(a, b, m, k, n, ta, tb, array("d", o), True))
    # the micro-kernel and the thread count must not change a single bit
    shapes = [(m, k, n, r(m * k), r(k * n)) for m, k, n in ((37, 200, 300), (300, 200, 37))]  # big enough to split
    ref = None
    try:
        for simd, threads in ((1, 1), (2, 1), (3, 1), (0, 2), (0, 3)):
            _c["set_simd"](simd)
            _c["set_threads"](threads)
            got = [list(c.matmul(a, b, m, k, n, ta, tb)) for m, k, n, a, b in shapes
                   for ta, tb in ((False, False), (True, True), (False, True))]
            if ref is None:
                ref = got
            elif got != ref:
                raise RuntimeError(f"C kernel self-test failed: matmul differs with simd={simd} threads={threads}")
    finally:
        _c["set_simd"](0)
        _c["set_threads"](_threads or 1)
    T, D, H, V = 7, 8, 2, 11
    x, g, bias = r(T * D), r(T * D), r(D)
    check("add", c.add(x, g), py.add(x, g))
    x2 = array("d", x)
    c.add_(x2, g)
    check("add_", x2, py.add(x, g))
    check("add_rows", c.add_rows(x, bias, T, D), py.add_rows(x, bias, T, D))
    check("colsum", c.colsum(g, T, D), py.colsum(g, T, D))
    check("scale", c.scale(x, -0.37), py.scale(x, -0.37))
    check("sumsq", c.sumsq(x), py.sumsq(x))
    check("gelu", c.gelu(x), py.gelu(x))
    _, th = py.gelu(x)
    check("gelu_backward", c.gelu_backward(g, x, th), py.gelu_backward(g, x, th))
    gamma = r(D, 0.5, 1.5)
    check("layer_norm", c.layer_norm(x, gamma, bias, T, D, 1e-5), py.layer_norm(x, gamma, bias, T, D, 1e-5))
    _, xhat, rstd = py.layer_norm(x, gamma, bias, T, D, 1e-5)
    for flags in ((True, True, True), (True, False, False), (False, True, False)):
        check("layer_norm_backward", c.layer_norm_backward(g, xhat, rstd, gamma, T, D, *flags),
              py.layer_norm_backward(g, xhat, rstd, gamma, T, D, *flags))
    qkv = r(3 * T * D)
    check("attention", c.attention(qkv, T, D, H), py.attention(qkv, T, D, H))
    _, P = py.attention(qkv, T, D, H)
    check("attention_backward", c.attention_backward(qkv, P, g, T, D, H), py.attention_backward(qkv, P, g, T, D, H))
    logits, tg = r(T * V, -4.0, 4.0), array("i", [rng.randrange(V) for _ in range(T)])
    check("cross_entropy", c.cross_entropy(logits, tg, T, V), py.cross_entropy(logits, tg, T, V))
    _, probs = py.cross_entropy(logits, tg, T, V)
    check("cross_entropy_backward", c.cross_entropy_backward(probs, tg, T, V, 0.25),
          py.cross_entropy_backward(probs, tg, T, V, 0.25))
    w, ids, ge = r(V * D), array("i", [3, 0, 3, 10, 5]), r(5 * D)
    check("embedding", c.embedding(w, ids, V, D), py.embedding(w, ids, V, D))
    check("embedding_backward", c.embedding_backward(ge, ids, V, D), py.embedding_backward(ge, ids, V, D))
    m_, v_ = r(T * D), r(T * D, 0.0, 1.0)
    cw, cm, cv = array("d", x), array("d", m_), array("d", v_)
    c.adamw(cw, g, cm, cv, 0.9, 0.999, 1e-3, 1.25, 1e-8, 0.99)
    pw = array("d", x)
    py.adamw(pw, g, m_, v_, 0.9, 0.999, 1e-3, 1.25, 1e-8, 0.99)
    check("adamw", (cw, cm, cv), (pw, m_, v_))


def kernel_source_digest() -> str | None:
    """SHA-256 hex digest of the shipped C kernel source (None if it is missing)."""
    try:
        with open(KERNEL_SOURCE, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _default_threads() -> int:
    raw = os.environ.get("GENESIS_NN_THREADS", "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return max(1, min(2, os.cpu_count() or 1))


def _setup(lib):
    global _threads
    kern = _c_kernels(lib)
    if _threads is None:
        _threads = _default_threads()
    _c["simd"] = {1: "scalar", 2: "vec2", 3: "avx"}.get(_c["set_simd"](0), "?")
    _self_test(kern)
    _threads = _c["set_threads"](_threads)
    return kern


def _load_c() -> bool:
    """Compile (once, cached on disk, see :mod:`genesis.native`) and load the C kernels; never raises."""
    if _c["tried"]:
        return _c["ops"] is not None
    _c["tried"] = True
    info = native.load(KERNEL_SOURCE, "gn_kernels", _setup, _CFLAGS)
    _c.update(ops=info["handle"], compiler=info["compiler"], library=info["library"], failure=info["failure"])
    return _c["ops"] is not None


def _select(name: str) -> str:
    if name not in _BACKENDS:
        raise ValueError(f"unknown nn backend {name!r}; expected one of {_BACKENDS}")
    if name == "python":
        return "python"
    if _load_c():
        return "c"
    if name == "c":
        raise RuntimeError(f"C backend unavailable: {_c['failure']}")
    return "python"


def ops() -> Kernels:
    """The kernel set of the active backend."""
    global _active, _ops
    if _ops is None:
        req = _requested or (os.environ.get("GENESIS_NN_BACKEND", "auto").strip().lower() or "auto")
        _active = _select(req)
        _ops = _c["ops"] if _active == "c" else PYTHON_KERNELS
    return _ops


def kernels(name: str) -> Kernels:
    """The kernel set of backend ``python`` or ``c`` regardless of the active one (tests, benchmarks)."""
    if name == "python":
        return PYTHON_KERNELS
    if name == "c" and _load_c():
        return _c["ops"]
    raise RuntimeError(f"backend {name!r} unavailable: {_c['failure'] if name == 'c' else 'unknown backend'}")


def set_backend(name: str) -> str:
    """Select ``auto``/``python``/``c`` programmatically; returns the active backend name."""
    global _requested, _active, _ops
    name = str(name).strip().lower()
    active = _select(name)
    _requested, _active = name, active
    _ops = _c["ops"] if active == "c" else PYTHON_KERNELS
    return active


def set_threads(n: int | None = None) -> int:
    """Threads used by large C matrix products (``None``: ``GENESIS_NN_THREADS`` or ``min(2, cpu_count)``).

    Results are bit-identical for any value. Returns the effective count (1 for the Python backend)."""
    global _threads
    if n is not None and (type(n) is not int or n < 1):
        raise ValueError("threads must be an int >= 1")
    _threads = _default_threads() if n is None else n
    if _c["ops"] is not None:
        _threads = _c["set_threads"](_threads)
    return _threads if _c["ops"] is not None else 1


def backend_name() -> str:
    ops()
    return _active


def available_backends() -> list[str]:
    return ["python", "c"] if _load_c() else ["python"]


def backend_info() -> dict:
    ops()
    c_ok = _c["ops"] is not None
    return {"name": _active, "requested": _requested or os.environ.get("GENESIS_NN_BACKEND", "auto"),
            "compiler": _c["compiler"], "library": _c["library"],
            "kernel_source_sha256": kernel_source_digest(), "failure": _c["failure"],
            "simd": _c["simd"] if c_ok else None, "threads": _threads if c_ok else None}


def _as_buffer(x):
    return x if isinstance(x, array) and x.typecode == "d" else array("d", x)


def matmul(a, b, m: int, k: int, n: int, trans_a: bool = False, trans_b: bool = False,
           out=None, accumulate: bool = False):
    """C (m x n) = op(A) @ op(B) on flat row-major ``array('d')`` buffers (see module doc)."""
    _dims(m, k, n)
    a, b = _as_buffer(a), _as_buffer(b)
    if len(a) != m * k or len(b) != k * n:
        raise ValueError(f"matmul size mismatch: len(a)={len(a)}, len(b)={len(b)}, m={m}, k={k}, n={n}")
    if out is not None and (len(out) != m * n or out is a or out is b
                            or not (isinstance(out, array) and out.typecode == "d")):
        raise ValueError("out must be a distinct array('d') of length m*n")
    if accumulate and out is None:
        accumulate = False
    return ops().matmul(a, b, m, k, n, trans_a, trans_b, out, accumulate)


def benchmark_matmul(n: int = 128, repeats: int = 3) -> dict:
    """Best-of-``repeats`` GFLOP/s of an n x n x n product for each available backend."""
    rng = random.Random(0)
    a = array("d", [rng.random() for _ in range(n * n)])
    b = array("d", [rng.random() for _ in range(n * n)])
    out = _zeros(n * n)
    result = {}
    for name in available_backends():
        fn = kernels(name).matmul
        best = math.inf
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            fn(a, b, n, n, n, False, False, out, False)
            best = min(best, time.perf_counter() - t0)
        result[name] = 2.0 * n ** 3 / max(best, 1e-12) / 1e9
    return result
