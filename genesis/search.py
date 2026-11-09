"""Program search: enumerate candidate expressions and score them against observed outputs.

:func:`search_programs` is the inner loop of induction. It enumerates expressions bottom-up exactly like
:func:`genesis.expressions.enumerate_programs` and returns only what induction needs:

* ``matches``: candidates that mispredict at most ``tol`` observations, in enumeration order;
* ``offsets``: bias fits ``e + r`` for small candidates whose residual ``y - e(x)`` equals the same constant
  ``r != 0`` on all but at most ``tol`` observations (the majority residual, so isolated noisy observations do
  not hide an otherwise exact rule);
* ``deep``: when nothing enumerated fits with at most ``deepen_bad`` mismatches, expressions of ``deep_size``
  nodes found *without enumerating them*: bias fits and remainders of the level two nodes below, and a
  meet-in-the-middle step. For every candidate ``A`` and operator, the outputs ``B`` must have so that
  ``A op B = y`` are computed on a window of observations (``y - A``, ``A - y``, ``y + A``, ``y / A``) and looked
  up among the enumerated signatures; each hit is then checked on all observations against ``tol``. Several
  windows are tried, so a noisy observation in one window does not hide the rule. When the level two nodes below
  is not enumerated either, an *affine* meet in the middle finds ``(A op B) + c``: ``B`` is looked up up to a
  constant shift (window differences) and ``c`` is fitted. Iterative deepening tries ``deep_size``,
  ``deep_size + 2``, ... ``deep_max_size`` and stops at the first size that explains all but ``deepen_bad``
  observations. The candidate budget is unchanged: deepening costs hash lookups per candidate and operator, not
  a larger enumeration.

Two interchangeable backends compute the same result. The C kernel (``_search.c``, built from this repository
through :mod:`genesis.native` and self-tested against the Python reference before use) is used when available;
the pure-Python reference is the fallback and the specification. Selection: env ``GENESIS_SEARCH_BACKEND`` in
{``auto`` (default), ``python``, ``c``}. The only observable difference is wall-clock time, so the time budget
(``max_seconds``) may truncate the Python search earlier.
"""

from __future__ import annotations

import ctypes
import itertools
import os
import time
from collections import Counter
from dataclasses import dataclass, field

from . import native
from .expressions import Expr, _trivial, app, const, enumerate_programs, parse_shape, var
from .innate import ARITHMETIC_PRIMITIVES

KERNEL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_search.c")
OPS = ("add", "sub", "mul", "mod")
# Signatures are stored as int64 in a single arena: cap its size so a long history cannot exhaust memory.
ARENA_BYTES = 256 * 2**20
_SAFE = 2**61  # |values| above this go to the Python path (no int64 overflow in residuals)
_C_MAX_LEVELS = 64  # the C kernel keeps fixed per-size level tables; larger max_size runs in Python
CHAIN_MAX = 64  # candidates sharing one window signature that are checked on all observations
_VARIANTS = (("add", True), ("sub", True), ("sub", False), ("mul", True))  # (operator, A on the left?)
_AFFINE = (("add", True), ("sub", True), ("sub", False))  # (A op B) + c, c fitted

_c: dict = {"tried": False, "fn": None, "shapes": None, "failure": None, "library": None, "compiler": None}
_requested: str | None = None


@dataclass
class SearchResult:
    matches: list  # [(Expr, n_bad)]
    offsets: list  # [(Expr, r, n_bad)]: e + r explains all but n_bad observations
    deep: list  # [(Expr, n_bad)] from deepening, deduplicated, in discovery order
    counts: dict  # size -> number of syntactically constructed expressions (for the Occam prior)
    evaluated: int
    truncated: bool
    n_candidates: int
    seconds: float = 0.0
    backend: str = "python"
    leaves: list = field(default_factory=list)
    deep_ran: bool = False  # the deepening phase was triggered (whether or not it found anything)


def _leaves(inputs: list, var_names, constants) -> list:
    leaves = []
    for v in var_names:
        sig = tuple(inp.get(v) for inp in inputs)
        if None not in sig:
            leaves.append((var(v), sig))
    n = len(inputs)
    for c in sorted(set(constants)):
        leaves.append((const(c), (c,) * n))
    return leaves


def _majority_offset(ys, sig, tol: int):
    """Residual shared by all but <= tol observations (Boyer-Moore vote, as in the C kernel), else None."""
    if tol == 0:
        r = ys[0] - sig[0]
    else:
        r, votes = 0, 0
        for y, s in zip(ys, sig):
            d = y - s
            if votes == 0:
                r, votes = d, 1
            else:
                votes += 1 if d == r else -1
    if r == 0:
        return None
    miss = sum(y - s != r for y, s in zip(ys, sig))
    return (r, miss) if miss <= tol else None


def _deep_python(cands, by_size, ys, tol, max_size, ops, deep_size, windows, cap, t0, max_seconds, deep_max=0,
                 deepen_bad=-1):
    """Reference for the deepening phase of the C kernel: returns (raw results, timed out)."""
    from .innate import ARITHMETIC_PRIMITIVES as P
    n = len(ys)
    start, k = {}, 0
    for size in sorted(by_size):
        start[size] = k
        k += len(by_size[size])
    top = max([size for size, level in by_size.items() if level], default=1)
    leaves = list(enumerate(by_size.get(1, [])))
    out = []
    timed_out = False
    for ds in range(deep_size, max(deep_size, deep_max) + 1, 2):
        if len(out) >= cap or timed_out:
            break
        n0 = len(out)
        last = ds - 2
        level = list(enumerate(by_size.get(last, []), start.get(last, 0))) if last <= max_size and n >= 2 else []
        for k, (e, sig) in level:
            if len(out) >= cap:
                break
            fit = _majority_offset(ys, sig, tol)
            if fit is not None:
                out.append((-2, k, -1, fit[1], fit[0]))
        for k, (e, sig) in level if "mod" in ops else []:
            for c, (ce, _) in leaves:
                if len(out) >= cap:
                    break
                if ce.op == "const" and ce.value > 1:
                    bad = sum(v % ce.value != y for v, y in zip(sig, ys))
                    if bad <= tol:
                        out.append(("mod", k, c, bad, 0))
        affine = last > max_size
        for rows in windows:
            if len(out) >= cap or timed_out:
                break
            index: dict = {}
            shifted: dict = {}
            for k, (e, sig) in enumerate(cands):
                index.setdefault(tuple(sig[r] for r in rows), []).append(k)
                if affine:
                    shifted.setdefault(tuple(sig[r] - sig[rows[0]] for r in rows[1:]), []).append(k)
            for a, (ea, sa) in enumerate(cands):
                if len(out) >= cap:
                    break
                if a % 1024 == 0 and time.perf_counter() - t0 > max_seconds:
                    timed_out = True
                    break
                want = ds - 1 - ea.size
                for op, a_left in _VARIANTS if 1 <= want <= top else ():
                    if op not in ops or len(out) >= cap:
                        continue
                    need = []
                    for r in rows:
                        y, x = ys[r], sa[r]
                        if op == "add":
                            need.append(y - x)
                        elif op == "sub":
                            need.append(x - y if a_left else y + x)
                        elif x == 0 or y % x:
                            break
                        else:
                            need.append(y // x)
                    if len(need) != len(rows):
                        continue
                    for b in index.get(tuple(need), [])[:CHAIN_MAX]:
                        if len(out) >= cap:
                            break
                        if ea.size + cands[b][0].size + 1 != ds:
                            continue
                        l, r = (a, b) if a_left else (b, a)
                        el, er = cands[l][0], cands[r][0]
                        if op in ("add", "mul") and (el.size, l) > (er.size, r):
                            continue
                        if _trivial(op, el, er):
                            continue
                        fn = P[op].fn
                        bad = 0
                        for x1, x2, y in zip(cands[l][1], cands[r][1], ys):
                            v = fn(x1, x2)
                            bad += v is None or v != y
                            if bad > tol:
                                break
                        if bad <= tol:
                            out.append((op, l, r, bad, 0))
                awant = ds - 3 - ea.size
                for op, a_left in _AFFINE if affine and 1 <= awant <= top else ():
                    if op not in ops or len(out) >= cap:
                        continue
                    need = [ys[r] - sa[r] if op == "add" else sa[r] - ys[r] if a_left else ys[r] + sa[r]
                            for r in rows]
                    if any(abs(v) > _SAFE for v in need):
                        continue
                    for b in shifted.get(tuple(v - need[0] for v in need[1:]), [])[:CHAIN_MAX]:
                        if len(out) >= cap:
                            break
                        if ea.size + cands[b][0].size + 3 != ds:
                            continue
                        l, r = (a, b) if a_left else (b, a)
                        el, er = cands[l][0], cands[r][0]
                        if op == "add" and (el.size, l) > (er.size, r):
                            continue
                        if _trivial(op, el, er):
                            continue
                        b0 = cands[b][1][rows[0]]
                        c = b0 - need[0] if op == "sub" and a_left else need[0] - b0
                        if c == 0:
                            continue
                        fn, add = P[op].fn, P["add"].fn
                        bad = 0
                        for x1, x2, y in zip(cands[l][1], cands[r][1], ys):
                            v = fn(x1, x2)
                            bad += v is None or add(v, c) != y
                            if bad > tol:
                                break
                        if bad <= tol:
                            out.append((op, l, r, bad, c))
        if any(o[3] <= deepen_bad for o in out[n0:]):
            break
    return out, timed_out


def _finish_deep(raw, node, size_of) -> list:
    """Build expressions for raw deepening results, dropping repeats found in several windows."""
    seen, deep = set(), []
    for kind, l, r, bad, val in raw:
        key = (kind, l, r, val)
        if key in seen:
            continue
        seen.add(key)
        if kind == -2:
            e = app("add", node(l), const(val))
        else:
            e = Expr(kind, (node(l), node(r)), None, size_of(l) + size_of(r) + 1)
            if val:
                e = app("add", e, const(val))
        deep.append((e, bad))
    return deep


def _windows(n: int, win_len: int, n_win: int) -> list:
    """Most recent observations first; consecutive windows overlap by half."""
    if n <= win_len:
        return [list(range(n))]
    step = max(1, win_len // 2)
    out = []
    for k in range(n_win):
        end = n - k * step
        if end - win_len < 0:
            break
        out.append(list(range(end - win_len, end)))
    return out


def _with_deep_count(counts: dict, deep: list, deep_size: int) -> dict:
    """Occam prior for unenumerated sizes: extrapolate the growth of the enumerated levels."""
    if not deep:
        return counts
    counts = dict(counts)
    for size in range(deep_size, max(e.size for e, _ in deep) + 1, 2):
        if size not in counts:
            a, b = counts.get(size - 2, 1), counts.get(size - 4, 1)
            counts[size] = max(a, a * a // max(b, 1))
    return counts


def _search_python(inputs, ys, var_names, constants, tol, max_size, max_candidates, max_seconds, bias_max_size,
                   ops, keep_all_upto, deepen_bad=-1, deep_size=0, windows=(), deep_max=0) -> SearchResult:
    t0 = time.perf_counter()
    en = enumerate_programs(inputs, var_names, constants, max_size, max_candidates, max_seconds, ops, keep_all_upto)
    n = len(ys)
    matches, offsets = [], []
    for e, sig in en.candidates:
        if tol == 0:
            if sig == ys:
                matches.append((e, 0))
        else:
            bad = sum(a != b for a, b in zip(sig, ys))
            if bad <= tol:
                matches.append((e, bad))
        if n >= 2 and e.op != "const" and e.size <= bias_max_size:
            fit = _majority_offset(ys, sig, tol)
            if fit is not None:
                offsets.append((e, fit[0], fit[1]))
    deep, truncated, deep_ran = [], en.truncated, False
    best = min([b for _, b in matches] + [b for _, _, b in offsets], default=n + 1)
    top = max([size for size, level in en.by_size.items() if level], default=1)
    if deepen_bad >= 0 and best > deepen_bad and deep_size > top and deep_size - 1 - top >= 1:
        deep_ran = True
        cap = max(max_candidates, len(en.by_size.get(1, [])) + 1) + 1
        raw, timed_out = _deep_python(en.candidates, en.by_size, ys, tol, max_size, ops, deep_size, windows, cap,
                                      t0, max_seconds, deep_max, deepen_bad)
        truncated = truncated or timed_out
        deep = _finish_deep(raw, lambda k: en.candidates[k][0], lambda k: en.candidates[k][0].size)
    return SearchResult(matches, offsets, deep, _with_deep_count(en.counts, deep, deep_size), en.evaluated,
                        truncated, len(en.candidates), time.perf_counter() - t0, "python", en.by_size.get(1, []),
                        deep_ran)


def _search_c(inputs, ys, var_names, constants, tol, max_size, max_candidates, max_seconds, bias_max_size,
              ops, keep_all_upto, deepen_bad=-1, deep_size=0, windows=(), deep_max=0) -> SearchResult:
    t0 = time.perf_counter()
    n = len(ys)
    leaves = _leaves(inputs, var_names, constants)
    nl = len(leaves)
    cap = max(max_candidates, nl + 1) + 1
    I64, I32 = ctypes.c_int64, ctypes.c_int32
    leaf_sigs = (I64 * (nl * n))(*[v for _, s in leaves for v in s])
    leaf_const = (I32 * nl)(*[e.op == "const" for e, _ in leaves])
    op_codes = (I32 * len(ops))(*[OPS.index(o) for o in ops])
    node_op, node_l, node_r, node_size = (I32 * cap)(), (I32 * cap)(), (I32 * cap)(), (I32 * cap)()
    sigs = (I64 * (cap * n))()
    counts = (I64 * max(2, max_size + 1))()
    stats = (I64 * 4)()
    match_node, match_bad = (I32 * cap)(), (I32 * cap)()
    off_node, off_bad, off_val = (I32 * cap)(), (I32 * cap)(), (I64 * cap)()
    n_match, n_off, n_deep = I64(), I64(), I64()
    win_len = len(windows[0]) if windows else 0
    win_rows = (I32 * max(1, len(windows) * win_len))(*[r for w in windows for r in w])
    deep_kind, deep_l, deep_r, deep_bad, deep_val = (I32 * cap)(), (I32 * cap)(), (I32 * cap)(), (I32 * cap)(), \
        (I64 * cap)()
    rc = _c["fn"](n, (I64 * n)(*ys), nl, leaf_sigs, leaf_const, max_size, keep_all_upto, len(ops), op_codes,
                  max_candidates, float(max_seconds), tol, bias_max_size, cap, node_op, node_l, node_r, node_size,
                  sigs, counts, stats, match_node, match_bad, ctypes.byref(n_match), off_node, off_bad, off_val,
                  ctypes.byref(n_off), deepen_bad, deep_size, max(deep_size, deep_max), len(windows), win_len,
                  win_rows, deep_kind, deep_l,
                  deep_r, deep_bad, deep_val, ctypes.byref(n_deep))
    if rc != 0:
        raise MemoryError("search kernel could not allocate its tables")
    built: dict = {}

    def build(k: int) -> Expr:
        e = built.get(k)
        if e is None:
            if node_op[k] < 0:
                e = leaves[node_l[k]][0]
            else:
                e = Expr(OPS[node_op[k]], (build(node_l[k]), build(node_r[k])), None, node_size[k])
            built[k] = e
        return e

    matches = [(build(match_node[i]), match_bad[i]) for i in range(n_match.value)]
    offsets = [(build(off_node[i]), off_val[i], off_bad[i]) for i in range(n_off.value)]
    raw = [(OPS[deep_kind[i]] if deep_kind[i] >= 0 else -2, deep_l[i], deep_r[i], deep_bad[i], deep_val[i])
           for i in range(n_deep.value)]
    deep = _finish_deep(raw, build, lambda k: node_size[k])
    cnt = {s: counts[s] for s in range(1, max_size + 1, 2) if s == 1 or counts[s] > 0}
    return SearchResult(matches, offsets, deep, _with_deep_count(cnt, deep, deep_size), stats[0], bool(stats[1]),
                        stats[2], time.perf_counter() - t0, "c", leaves, bool(stats[3]))


def _setup(lib):
    sh = lib.gn_shapes
    P32, P64 = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)
    sh.argtypes = [ctypes.c_int, P64, ctypes.c_int, P64, ctypes.c_int, P64, ctypes.c_int, P32, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int64, P64, P32, P64, P64, P64]
    sh.restype = ctypes.c_int
    fn = lib.gn_search
    P32, P64 = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64)
    fn.argtypes = [ctypes.c_int, P64, ctypes.c_int, P64, P32, ctypes.c_int, ctypes.c_int, ctypes.c_int, P32,
                   ctypes.c_int64, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                   P32, P32, P32, P32, P64, P64, P64, P32, P32, P64, P32, P32, P64, P64,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, P32, P32, P32, P32, P32,
                   P64, P64]
    fn.restype = ctypes.c_int
    _c["fn"], _c["shapes"] = fn, sh
    # Self-test against the reference: wrap-around, a noisy observation, bias fits and deepening.
    inputs = [{"x": x, "t": t} for t, x in enumerate([3, 5, 7, 9, 0, 2, 4, 6, 8, 1])]
    ys = tuple((d["x"] + 2) % 10 for d in inputs)
    deep_ys = tuple(d["t"] * (d["t"] + 3) - 2 for d in inputs)
    affine_ys = tuple(2 * d["x"] - 3 * d["t"] + 7 for d in inputs)
    for tol, y in ((0, ys), (2, ys[:4] + (ys[4] + 5,) + ys[5:]), (0, deep_ys), (2, deep_ys[:3] + (7,) + deep_ys[4:]),
                   (2, affine_ys[:6] + (0,) + affine_ys[7:])):
        args = (inputs, y, ["x", "t"], [0, 1, 2, 3, 10], tol, 5, 50000, 60.0, 3, OPS, 3, 0, 7,
                _windows(len(y), 6, 3), 9)
        if _summary(_search_c(*args)) != _summary(_search_python(*args)):
            _c["fn"] = None
            raise RuntimeError("search kernel self-test failed")
    for shape, fit, tol in (("mod(add(V,C),C)", False, 0), ("mul(V,add(V,C))", True, 2), ("sub(V,V)", False, 3)):
        shape = parse_shape(shape)
        args = (shape, inputs, deep_ys[:5] + (0,) + deep_ys[6:], ["x", "t"], [0, 1, 2, 3, 10], tol, fit)
        if _shapes_c(*args) != _shapes_python(*args)[:2]:
            _c["fn"] = None
            raise RuntimeError("shape kernel self-test failed")
    return fn


def _summary(r: SearchResult) -> tuple:
    return ([(str(e), b) for e, b in r.matches], [(str(e), v, b) for e, v, b in r.offsets],
            [(str(e), e.size, b) for e, b in r.deep], r.counts, r.evaluated, r.truncated, r.n_candidates, r.deep_ran)


# ------------------------------------------------------------------ procedural memory: shape instantiation
def _postfix(tree) -> list | None:
    """Shape tree in post-order: an OPS index, -1 (variable slot) or -2 (constant slot); None if unsupported."""
    out: list = []

    def walk(t) -> bool:
        if t in ("V", "C"):
            out.append(-1 if t == "V" else -2)
            return True
        op, args = t
        if op not in OPS or len(args) != 2 or not (walk(args[0]) and walk(args[1])):
            return False
        out.append(OPS.index(op))
        return True

    return out if walk(tree) and len(out) <= 31 else None


def _slots(prog) -> tuple:
    return prog.count(-1), prog.count(-2)


def _shapes_python(tree, inputs, ys, var_names, consts, tol, fit_bias) -> tuple:
    """Reference: exactly genesis.induction's former instance loop (``Expr.evaluate`` semantics; the fitted
    constant is the most common residual, first seen on ties)."""
    prog = _postfix(tree)
    fns = [ARITHMETIC_PRIMITIVES[o].fn for o in OPS]
    nv, nc = _slots(prog)
    hits, evaluated = [], 0
    for vs in itertools.product(var_names, repeat=nv):
        for cs in itertools.product(consts, repeat=nc):
            sig = []
            for d in inputs:
                st, vi, ci = [], 0, 0
                for tok in prog:
                    if tok == -1:
                        st.append(d.get(vs[vi]))
                        vi += 1
                    elif tok == -2:
                        st.append(cs[ci])
                        ci += 1
                    else:
                        b, a = st.pop(), st.pop()
                        st.append(None if a is None or b is None else fns[tok](a, b))
                sig.append(st[0])
            if fit_bias:
                if None in sig:
                    continue
                r = Counter(y - v for y, v in zip(ys, sig)).most_common(1)[0][0]
                add = ARITHMETIC_PRIMITIVES["add"].fn
                bad = sum(add(v, r) != y for v, y in zip(sig, ys))
            else:
                r, bad = None, sum(v != y for v, y in zip(sig, ys))
            evaluated += 1
            if bad <= tol:
                hits.append((vs, cs, r, bad))
    return hits, evaluated, "python"


def _shapes_c(tree, inputs, ys, var_names, consts, tol, fit_bias) -> tuple:
    prog = _postfix(tree)
    nv, nc = _slots(prog)
    n = len(ys)
    total = len(var_names) ** nv * len(consts) ** nc
    if total == 0:
        return [], 0
    I64, I32 = ctypes.c_int64, ctypes.c_int32
    var_sigs = (I64 * max(1, len(var_names) * n))(*[d[v] for v in var_names for d in inputs])
    hit_idx, hit_bad, hit_r = (I64 * total)(), (I32 * total)(), (I64 * total)()
    n_hits, evaluated = I64(), I64()
    rc = _c["shapes"](n, (I64 * n)(*ys), len(var_names), var_sigs, len(consts), (I64 * max(1, len(consts)))(*consts),
                      len(prog), (I32 * len(prog))(*prog), nv, nc, int(fit_bias), tol, total, hit_idx, hit_bad,
                      hit_r, ctypes.byref(n_hits), ctypes.byref(evaluated))
    if rc != 0:
        raise ValueError("shape kernel rejected its arguments")
    hits = []
    for i in range(n_hits.value):
        rem, cs, vs = hit_idx[i], [], []
        for _ in range(nc):
            rem, k = divmod(rem, len(consts))
            cs.append(consts[k])
        for _ in range(nv):
            rem, k = divmod(rem, len(var_names))
            vs.append(var_names[k])
        hits.append((tuple(reversed(vs)), tuple(reversed(cs)), hit_r[i] if fit_bias else None, hit_bad[i]))
    return hits, evaluated.value


def match_shape(tree, inputs: list, ys, var_names, consts, tol: int, fit_bias: bool = False) -> tuple | None:
    """Instances of a procedural-memory shape that explain all but ``tol`` observations.

    Returns ``(hits, evaluated, backend)`` with ``hits = [(variables, constants, fitted_constant, n_bad)]`` in
    ``itertools.product`` order, or None when the shape has operators the kernel does not know (conditionals).
    With ``fit_bias``, ``tree`` is the body X of "add(X, C)" and the constant is fitted, not enumerated."""
    if _postfix(tree) is None:
        return None
    ys, consts, var_names = tuple(ys), list(consts), list(var_names)
    req = _requested_backend()
    if req != "python" and _load_c() and ys and (not fit_bias or 2 * tol < len(ys)):
        values = [abs(v) for v in ys] + [abs(c) for c in consts]
        values += [abs(d[v]) if d.get(v) is not None else _SAFE + 1 for d in inputs for v in var_names]
        if max(values, default=0) <= _SAFE:
            return (*_shapes_c(tree, inputs, ys, var_names, consts, tol, fit_bias), "c")
    return _shapes_python(tree, inputs, ys, var_names, consts, tol, fit_bias)


def _load_c() -> bool:
    if not _c["tried"]:
        _c["tried"] = True
        info = native.load(KERNEL_SOURCE, "gn_search", _setup)
        _c.update(fn=info["handle"], compiler=info["compiler"], library=info["library"], failure=info["failure"])
    return _c["fn"] is not None


def set_backend(name: str | None) -> None:
    """Force ``"python"``/``"c"``/``"auto"`` (None: back to ``GENESIS_SEARCH_BACKEND``)."""
    global _requested
    if name is not None and name not in ("auto", "python", "c"):
        raise ValueError(f"unknown search backend {name!r}")
    _requested = name


def _requested_backend() -> str:
    return _requested or (os.environ.get("GENESIS_SEARCH_BACKEND", "auto").strip().lower() or "auto")


def backend_info() -> dict:
    available = _load_c()
    req = _requested_backend()
    return {"requested": req, "active": "c" if available and req != "python" else "python",
            "compiler": _c["compiler"], "library": _c["library"], "failure": _c["failure"]}


def search_programs(inputs: list, ys, var_names, constants, tol: int, max_size: int, max_candidates: int,
                    max_seconds: float, bias_max_size: int, ops=OPS, keep_all_upto: int = 3, deepen_bad: int = -1,
                    deep_size: int = 0, window: int = 8, n_windows: int = 4, deep_max_size: int = 0) -> SearchResult:
    """See the module docstring. ``deepen_bad < 0`` disables deepening; sizes ``deep_size``, ``deep_size + 2``,
    ... ``deep_max_size`` are tried in turn until one explains all but ``deepen_bad`` observations."""
    ys = tuple(ys)
    n = len(ys)
    if n == 0 or len(inputs) != n:
        raise ValueError("search needs one input per observed output")
    # Same budget for both backends, so results never depend on which one ran.
    max_candidates = max(1, min(max_candidates, ARENA_BYTES // (8 * n)))
    req = _requested_backend()
    if req not in ("auto", "python", "c"):
        raise ValueError(f"GENESIS_SEARCH_BACKEND must be auto, python or c, not {req!r}")
    windows = _windows(n, window, n_windows) if deepen_bad >= 0 else []
    args = (inputs, ys, list(var_names), list(constants), tol, max_size, max_candidates, max_seconds,
            bias_max_size, tuple(ops), keep_all_upto, deepen_bad, deep_size, windows, deep_max_size)
    if req != "python" and _load_c():
        values = [abs(v) for v in ys] + [abs(c) for c in constants]
        values += [abs(inp[v]) for inp in inputs for v in var_names if inp.get(v) is not None]
        if 1 <= max_size < _C_MAX_LEVELS and max(values) <= _SAFE:
            return _search_c(*args)
    elif req == "c":
        raise RuntimeError(f"C search backend unavailable: {_c['failure']}")
    return _search_python(*args)
