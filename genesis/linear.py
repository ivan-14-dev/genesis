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

from itertools import combinations

from .expressions import Expr, app, const, var


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
    leave no k consecutive clean rows, but one row to spare per window is enough to step over each of them."""
    ys = tuple(ys)
    n = len(ys)
    feats = features(inputs, var_names, products)
    # A product costs two terms (it is three nodes, not one): at most one, within the same budget.
    subsets = [s for k in range(1, max_terms + 1) for s in combinations(range(len(feats)), k)
               if sum(feats[i][2] for i in s) <= 1 and k + sum(feats[i][2] for i in s) <= max_terms]
    by_len = {}
    for s in subsets:
        length = max(4, len(s) + 1 + spare)
        if length not in by_len:
            by_len[length] = windows(length)
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
                sol = solve_exact([[col[r] for col in cols] + [1, ys[r]] for r in rows])
                solved += 1
                if sol is None:
                    continue
                num, d = sol
                if any(x % d for x in num):
                    continue
                coefs = [x // d for x in num]
                c = coefs.pop()
                if any(a == 0 or abs(a) > max_coef for a in coefs) or abs(c) > max_const:
                    continue
                key = (s, tuple(coefs), c)
                if key in seen:
                    continue
                seen.add(key)
                bad = 0
                for r in range(n):
                    bad += sum(a * col[r] for a, col in zip(coefs, cols)) + c != ys[r]
                    if bad > tol:
                        break
                if bad <= tol and n - bad >= k + min_support:
                    out.append((build([feats[i][0] for i in s], coefs, c), bad))
                    if len(out) >= cap:
                        return out, solved
    return out, solved
