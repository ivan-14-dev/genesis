"""Exact sparse linear induction: ``y = a1*f1 + ... + ak*fk + c`` with small integer coefficients.

The enumerative search (genesis.search) pays for every node, so a rule such as ``2*x[t-1] - x[t-2] + 3*t + 1``
(11 nodes) or ``3*x[t-1] - 2*x[t-2] + 5*t - 4`` (13 nodes) is out of its reach, although its *form* is simple:
a linear combination of perceived quantities. This module solves for such rules directly, the way one solves a
small system of equations instead of guessing every expression:

* features are the perceived variables and, optionally, their pairwise products (``t*t``, ``t*x[t-1]``...);
* for every subset of at most ``max_terms`` features (at most one product, which counts as two terms), the
  coefficients and the constant are solved exactly (integer, fraction-free Gauss-Jordan elimination) on every
  choice of k rows among a short window of k + ``spare`` rows; several windows are tried (see
  :func:`dense_windows`), so corrupted observations cannot hide the rule even when no k consecutive rows are clean;
* a solution is kept only with small non-zero integer coefficients and when, over all observations, it explains
  at least ``min_support`` observations beyond those it was solved on, within the mismatch tolerance. A system
  with as many unknowns as rows always has a solution, so these extra confirmations are what separates a rule
  from a fit to noise.

Results are expressions over the innate primitives (add, sub, mul and constants), scored downstream like any other
hypothesis (Occam prior on their size, likelihood on all observations). Pure Python, exact integers, deterministic.
"""

from __future__ import annotations

import ctypes
from itertools import combinations

from . import search
from .expressions import Expr, app, const, var

_C_SAFE = 2**62  # larger |values| are flagged for the kernel, whose systems on them are solved in Python


def solve_exact(m: list) -> tuple | None:
    """Fraction-free Gauss-Jordan elimination of an augmented integer system ``m`` (k rows of k + 1 integers).
    Returns ``(numerators, d)`` with ``x_i = numerators[i] / d`` (``d`` is +-det), or None when singular. Every
    division is exact (Bareiss), so intermediate values stay as small as the determinants."""
    k = len(m)
    m = [list(row) for row in m]
    prev = 1
    for p in range(k):
        if m[p][p] == 0:
            for i in range(p + 1, k):
                if m[i][p]:
                    m[p], m[i] = m[i], m[p]
                    break
            else:
                return None
        rp = m[p]
        pv = rp[p]
        tail = rp[p + 1:]
        for i in range(k):
            if i != p:
                # Columns up to p are settled (zero off the diagonal, the running pivot on it, which ends as d):
                # only the remaining ones are updated.
                ri = m[i]
                f = ri[p]
                ri[p + 1:] = [(pv * a - f * b) // prev for a, b in zip(ri[p + 1:], tail)]
        prev = pv
    return [row[k] for row in m], prev


def dense_windows(n: int, win_len: int, n_win: int) -> list:
    """Up to ``n_win`` windows of ``win_len`` rows, most recent first, at the largest stride (at most half a window)
    that still fits them all: short histories are covered at stride 1, so one corrupted observation spoils as few
    candidate systems as possible; long ones at half a window, reaching further back."""
    if n <= win_len:
        return [list(range(n))]
    step = min(max(1, win_len // 2), max(1, (n - win_len) // max(1, n_win - 1)))
    out = []
    for k in range(n_win):
        end = n - k * step
        if end < win_len:
            break
        out.append(list(range(end - win_len, end)))
    return out


def _term(f: Expr, a: int) -> Expr:
    return f if a == 1 else app("mul", f, const(a))


def build(features: list, coefs: list, c: int) -> Expr:
    """``sum(a * f) + c`` with positive terms first, then subtractions, then the constant."""
    terms = list(zip(features, coefs))
    e = None
    for f, a in terms:
        if a > 0:
            e = _term(f, a) if e is None else app("add", e, _term(f, a))
    for f, a in terms:
        if a < 0:
            e = _term(f, a) if e is None else app("sub", e, _term(f, -a))
    if c:
        e = app("add", e, const(c))
    return e


def features(inputs: list, var_names: list, products: bool) -> list:
    """``[(expr, values, is_product)]``: variables defined on every observation, then their pairwise products.
    Constant columns are dropped (the fitted constant already covers them), and so are repeated columns."""
    base = []
    for v in var_names:
        vals = tuple(d.get(v) for d in inputs)
        if None not in vals:
            base.append((var(v), vals, False))
    out = list(base)
    if products:
        for i in range(len(base)):
            for j in range(i, len(base)):
                (ei, vi, _), (ej, vj, _) = base[i], base[j]
                out.append((app("mul", ei, ej), tuple(a * b for a, b in zip(vi, vj)), True))
    seen, kept = set(), []
    for e, vals, prod in out:
        if len(set(vals)) > 1 and vals not in seen:
            seen.add(vals)
            kept.append((e, vals, prod))
    return kept


def linear_rules(inputs: list, ys, var_names: list, windows, tol: int, max_terms: int = 4,
                 products: bool = True, max_coef: int = 16, max_const: int = 10**6, min_support: int = 3,
                 cap: int = 64, spare: int = 1) -> tuple:
    """Linear rules within ``tol`` mismatches, as ``([(Expr, n_bad)], systems solved)``, in discovery order
    (window rank, then subsets by size and feature order, then rows). ``windows(length)`` gives the windows of
    that many rows, most informative first. A rule with k unknowns (its coefficients and the constant) is solved
    on every k rows of a window of max(4, k + 1) rows: corrupted observations spaced closer than k rows apart
    leave no k consecutive clean rows, but one row to spare per window is enough to step over each of them.
    The C kernel (genesis/_search.c, gn_linear) computes the same result as the Python reference below."""
    ys = tuple(ys)
    feats = features(inputs, var_names, products)
    # A product costs two terms (it is three nodes, not one): at most one, within the same budget.
    subsets = [s for k in range(1, max_terms + 1) for s in combinations(range(len(feats)), k)
               if sum(feats[i][2] for i in s) <= 1 and k + sum(feats[i][2] for i in s) <= max_terms]
    by_len = {}
    for s in subsets:
        length = max(4, len(s) + 1 + spare)
        if length not in by_len:
            by_len[length] = windows(length)
    args = (ys, feats, subsets, by_len, tol, max_terms, max_coef, max_const, min_support, cap, spare)
    if search._requested_backend() != "python" and search._load_c() and search._c.get("linear") is not None:
        got = _linear_c(*args)
        if got is not None:
            return got
    return _linear_python(*args)


def _solve_system(cols: list, rows, ys, max_coef: int, max_const: int) -> tuple | None:
    """``(coefficients, constant)`` of the system on ``rows`` when it has an admissible solution, else None."""
    sol = solve_exact([[col[r] for col in cols] + [1, ys[r]] for r in rows])
    if sol is None:
        return None
    num, d = sol
    if any(x % d for x in num):
        return None
    coefs = [x // d for x in num]
    c = coefs.pop()
    if any(a == 0 or abs(a) > max_coef for a in coefs) or abs(c) > max_const:
        return None
    return coefs, c


def _mismatches(cols: list, coefs: list, c: int, ys, tol: int) -> int:
    """Observations the rule mispredicts, counted until one more than ``tol``."""
    bad = 0
    for r in range(len(ys)):
        bad += sum(a * col[r] for a, col in zip(coefs, cols)) + c != ys[r]
        if bad > tol:
            break
    return bad


def _linear_python(ys, feats, subsets, by_len, tol, max_terms, max_coef, max_const, min_support, cap, spare):
    n = len(ys)
    out, seen, tried, solved = [], set(), set(), 0
    for w in range(max((len(ws) for ws in by_len.values()), default=0)):
        for s in subsets:
            k = len(s) + 1
            ws = by_len[max(4, k + spare)]
            if w >= len(ws) or k > len(ws[w]) or n < k + min_support:
                continue
            for rows in combinations(ws[w], k) if spare else (tuple(ws[w][:k]),):
                if (s, rows) in tried:
                    continue
                tried.add((s, rows))
                cols = [feats[i][1] for i in s]
                sol = _solve_system(cols, rows, ys, max_coef, max_const)
                solved += 1
                if sol is None:
                    continue
                coefs, c = sol
                key = (s, tuple(coefs), c)
                if key in seen:
                    continue
                seen.add(key)
                bad = _mismatches(cols, coefs, c, ys, tol)
                if bad <= tol and n - bad >= k + min_support:
                    out.append((build([feats[i][0] for i in s], coefs, c), bad))
                    if len(out) >= cap:
                        return out, solved
    return out, solved


def _linear_c(ys, feats, subsets, by_len, tol, max_terms, max_coef, max_const, min_support, cap, spare):
    """The reference's loop in C. Each system is solved modulo 31-bit primes (an admissible solution is unique when
    the system is nonsingular modulo a prime, and equals the symmetric residues) and substituted back exactly
    (128-bit integers where the compiler has them, overflow-checked int64 otherwise); singularity is proven when the
    product of the primes it is singular modulo exceeds Hadamard's bound on the determinant. Systems this cannot
    settle (values flagged big, an overflow, a bound beyond the primes) are handed back one at a time and solved
    here, exactly as in the reference, before the kernel resumes; None when the configuration is out of the
    kernel's range."""
    n = len(ys)
    if not subsets or max_terms + 1 > 9 or max_coef > 2**16 or max_const > 2**29:
        return None
    n_win = max((len(ws) for ws in by_len.values()), default=0)
    # Row combinations per (unknowns k, window rank), each tried once per subset: the reference's ``tried`` set.
    off, cnt, flat = [0] * ((max_terms + 2) * n_win), [0] * ((max_terms + 2) * n_win), []
    for k in sorted({len(s) + 1 for s in subsets}):
        ws, seen = by_len[max(4, k + spare)], set()
        for w in range(n_win):
            off[k * n_win + w] = len(flat)
            if w >= len(ws) or k > len(ws[w]) or n < k + min_support:
                continue
            for rows in combinations(ws[w], k) if spare else (tuple(ws[w][:k]),):
                if rows not in seen:
                    seen.add(rows)
                    flat.extend(rows)
                    cnt[k * n_win + w] += 1
    I8, I32, I64 = ctypes.c_int8, ctypes.c_int32, ctypes.c_int64
    values = [v for _, vals, _ in feats for v in vals] + list(ys)
    big = [abs(v) > _C_SAFE for v in values]
    sub_feat = [0] * (len(subsets) * max_terms)
    for i, s in enumerate(subsets):
        sub_feat[i * max_terms: i * max_terms + len(s)] = s
    out_sub, out_bad = (I32 * cap)(), (I32 * cap)()
    out_coef, out_c = (I64 * (cap * max_terms))(), (I64 * cap)()
    n_out, solved, pos = I32(), I64(), (I32 * 3)()
    masked = [0 if b else v for v, b in zip(values, big)]
    nfv = n * len(feats)
    args = (n, (I64 * n)(*masked[nfv:]), len(feats), (I64 * max(1, nfv))(*masked[:nfv]), (I8 * len(big))(*big), len(subsets), (I32 * len(subsets))(*map(len, subsets)),
            (I32 * len(sub_feat))(*sub_feat), max_terms, n_win, (I32 * len(off))(*off), (I32 * len(cnt))(*cnt),
            (I32 * max(1, len(flat)))(*flat), tol, max_coef, max_const, min_support, cap, out_sub, out_coef, out_c,
            out_bad, ctypes.byref(n_out), ctypes.byref(solved), pos)
    while (rc := search._c["linear"](*args)) == 3:
        w, si, ci = pos
        s = subsets[si]
        k = len(s) + 1
        start = off[k * n_win + w] + ci * k
        cols = [feats[i][1] for i in s]
        solved.value += 1
        sol = _solve_system(cols, flat[start:start + k], ys, max_coef, max_const)
        if sol is not None and not any(out_sub[o] == si and out_c[o] == sol[1]
                                       and list(out_coef[o * max_terms: o * max_terms + len(s)]) == sol[0]
                                       for o in range(n_out.value)):
            coefs, c = sol
            bad = _mismatches(cols, coefs, c, ys, tol)
            if bad <= tol and n - bad >= k + min_support:
                o = n_out.value
                out_sub[o], out_c[o], out_bad[o] = si, c, bad
                out_coef[o * max_terms: o * max_terms + len(s)] = coefs
                n_out.value += 1
                if n_out.value >= cap:
                    break
        pos[2] = ci + 1
    else:
        if rc != 0:
            raise ValueError("linear kernel rejected its arguments")
    out = []
    for i in range(n_out.value):
        s = subsets[out_sub[i]]
        coefs = list(out_coef[i * max_terms: i * max_terms + len(s)])
        out.append((build([feats[j][0] for j in s], coefs, out_c[i]), out_bad[i]))
    return out, solved.value
