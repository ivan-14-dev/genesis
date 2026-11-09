"""Library learning and dreaming: the "sleep" half of a wake-sleep cycle.

During *wake* the learners induce rules from experience (:mod:`genesis.induction`) with the current library.
During *sleep* (:mod:`genesis.consolidation` calls this module) the agent improves that library:

* **Abstraction (sub-program promotion).** Sub-expressions that recur in several validated rules are promoted to
  *invented primitives*. An invention is a concrete sub-program over perceived variables, e.g.
  ``((x[t-1] * 2) - x[t-2])``. During wake it is one more leaf of the enumeration: one node instead of its size,
  so rules that use it become shorter, get a larger Occam prior and fall within the same search budget. Promotion
  is a minimum-description-length decision: an invention must save more nodes in the validated corpus than it
  costs to define (its size plus :data:`BASIS_COST`, a prior against growing the basis), and it must recur in at
  least :data:`MIN_SUPPORT` different problems. Sub-programs are compared by behaviour on fixed probe inputs, so
  ``(x * 2)`` and ``(x + x)`` count as one.
* **Dreaming.** Fantasy problems are sampled from procedural memory (validated shapes with fresh constants, rolled
  out by the domain's generative hook) and, with replays of real validated problems, train a recognition model
  ``q(shape | features of the observations)``. During wake it is a task-conditional prior: a hypothesis whose shape
  the model finds more plausible than uniform for the observations at hand gets that log-ratio added to its Occam
  prior, and when procedural memory holds more shapes than one search may try, it decides which ones are tried.

Neither phase creates beliefs. A rule found with an invention is stored inlined, in innate primitives only, and
must still be validated by forward predictions; dreams never reach semantic memory. Both phases are recomputed
from the current validated knowledge at every consolidation, so an invention whose supporting rules are refuted
disappears with them.
"""

from __future__ import annotations

import math
import random
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .expressions import Expr, app, fill_shape, parse_shape, shape_slots, var

LEAF_PREFIX = "@"  # invented leaves are variables "@1", "@2", ...: never a perceived variable name
MAX_INVENTIONS = 6
MIN_SUPPORT = 2
MIN_SIZE = 3
BASIS_COST = 2.0
N_PROBES = 12


# ================================================================ abstraction
@dataclass(frozen=True)
class Invention:
    name: str
    expr: Expr
    support: int  # different problems whose validated rule uses it
    uses: int
    gain: float  # description length saved on the validated corpus, in nodes

    def to_json(self) -> dict:
        return {"name": self.name, "expr": self.expr.to_json(), "statement": str(self.expr), "support": self.support,
                "uses": self.uses, "gain": self.gain}

    @staticmethod
    def from_json(d: dict) -> "Invention":
        return Invention(d["name"], Expr.from_json(d["expr"]), d["support"], d["uses"], d["gain"])


def _subtrees(e: Expr) -> Iterable[Expr]:
    yield e
    for a in e.args:
        yield from _subtrees(a)


def _has_op(e: Expr, op: str) -> bool:
    return e.op == op or any(_has_op(a, op) for a in e.args)


def _probes(names: tuple) -> list:
    rng = random.Random(zlib.crc32(repr(names).encode()))
    return [{n: rng.randint(-40, 40) for n in names} for _ in range(N_PROBES)]


class _Classes:
    """Behavioural class of a sub-program: its variables and its exact values on fixed probes."""

    def __init__(self):
        self._probes: dict = {}
        self._memo: dict = {}

    def key(self, e: Expr) -> tuple:
        text = str(e)
        k = self._memo.get(text)
        if k is None:
            names = tuple(sorted(e.variables()))
            probes = self._probes.get(names)
            if probes is None:
                probes = self._probes[names] = _probes(names)
            k = self._memo[text] = (names, tuple(e.evaluate(p, True) for p in probes))
        return k


def _rewrite(e: Expr, classes: _Classes, target: tuple, name: str) -> Expr:
    if e.op in ("var", "const"):
        return e
    if e.size >= MIN_SIZE and not _invented(e) and classes.key(e) == target:
        return var(name)
    return app(e.op, *[_rewrite(a, classes, target, name) for a in e.args])


def _invented(e: Expr) -> bool:
    return any(v.startswith(LEAF_PREFIX) for v in e.variables())


def promote(rules: Iterable, max_inventions: int = MAX_INVENTIONS, min_support: int = MIN_SUPPORT,
            basis_cost: float = BASIS_COST) -> list:
    """Greedy MDL abstraction over ``rules``, an iterable of ``(problem key, validated Expr)``.

    Each round promotes the sub-program class with the largest saving ``uses * (size - 1) - size - basis_cost``
    (ties: larger, then lexicographically smaller), then rewrites the corpus with the new leaf so later rounds only
    count what is left. Only proper sub-programs without conditionals, with a variable and at least
    :data:`MIN_SIZE` nodes are candidates."""
    corpus = sorted(((k, e) for k, e in rules), key=lambda t: (t[0], str(t[1])))
    classes = _Classes()
    out = []
    for _ in range(max_inventions):
        stats: dict = {}
        for k, e in corpus:
            for s in _subtrees(e):
                if s is e or s.size < MIN_SIZE or _has_op(s, "if_eq") or not s.variables() or _invented(s):
                    continue
                ck = classes.key(s)
                if None in ck[1]:
                    continue
                st = stats.get(ck)
                if st is None:
                    stats[ck] = [s, {k}, 1]
                else:
                    st[1].add(k)
                    st[2] += 1
                    if (s.size, str(s)) < (st[0].size, str(st[0])):
                        st[0] = s
        best = None
        for ck, (rep, problems, uses) in stats.items():
            if len(problems) < min_support:
                continue
            gain = uses * (rep.size - 1) - rep.size - basis_cost
            if gain <= 0:
                continue
            rank = (gain, rep.size, [-ord(ch) for ch in str(rep)])
            if best is None or rank > best[0]:
                best = (rank, ck, rep, len(problems), uses, gain)
        if best is None:
            break
        _, ck, rep, support, uses, gain = best
        name = f"{LEAF_PREFIX}{len(out) + 1}"
        out.append(Invention(name, rep, support, uses, gain))
        corpus = [(k, _rewrite(e, classes, ck, name)) for k, e in corpus]
    return out


def inline(e: Expr, defs: dict) -> Expr:
    """Replace invented leaves by their definitions: the result uses innate primitives only."""
    if e.op == "var":
        return defs.get(e.value, e)
    if e.op == "const":
        return e
    return app(e.op, *[inline(a, defs) for a in e.args])


# =================================================================== dreaming
def features(ys: list) -> frozenset:
    """Coarse, noise-tolerant descriptors of an observed output series (most recent last). Each holds when it is
    true of at least three quarters of the observations, so one corrupted observation does not flip it."""
    out = set()
    n = len(ys)
    if n < 4:
        return frozenset(out)

    def most(flags):
        flags = list(flags)
        return bool(flags) and sum(flags) >= 0.75 * len(flags)

    d1 = [b - a for a, b in zip(ys, ys[1:])]
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    for name, seq in (("d0", ys), ("d1", d1), ("d2", d2)):
        if len(seq) >= 3:
            c = Counter(seq).most_common(1)[0][1]
            if c >= 0.75 * len(seq):
                out.add(f"{name}-constant")
    for p in (2, 3):
        if n > p + 2 and most(ys[i] == ys[i - p] for i in range(p, n)):
            out.add(f"period-{p}")
    if most(a != 0 and b % a == 0 and abs(b) >= 2 * abs(a) for a, b in zip(ys, ys[1:])):
        out.add("ratio-growth")
    if most(b != 0 and abs(b) >= 2 * abs(a) for a, b in zip(ys, ys[1:])):
        out.add("fast-growth")
    if len(d1) >= 3 and most(abs(b) > abs(a) for a, b in zip(d1, d1[1:])):
        out.add("accelerating")
    if most(x < 0 for x in ys):
        out.add("negative")
    if len(set(ys)) <= max(2, n // 3):
        out.add("few-values")
    if max(ys) - min(ys) < 16:
        out.add("bounded")
    if len(d1) >= 3 and most(a * b < 0 for a, b in zip(d1, d1[1:])):
        out.add("alternating")
    return frozenset(out)


FEATURES = ("d0-constant", "d1-constant", "d2-constant", "period-2", "period-3", "ratio-growth", "fast-growth",
            "accelerating", "negative", "few-values", "bounded", "alternating")


class Recognizer:
    """Bernoulli naive Bayes ``q(shape | features)`` with Laplace smoothing; deterministic and inspectable."""

    def __init__(self, counts: Optional[dict] = None, totals: Optional[dict] = None):
        self.counts = counts or {}  # shape -> {feature: n}
        self.totals = totals or {}  # shape -> n examples

    def train(self, examples: Iterable) -> None:
        for shape, feats in examples:
            self.totals[shape] = self.totals.get(shape, 0) + 1
            c = self.counts.setdefault(shape, {})
            for f in sorted(feats):
                c[f] = c.get(f, 0) + 1

    def log_score(self, shape: str, feats: frozenset) -> Optional[float]:
        n = self.totals.get(shape)
        if not n:
            return None
        total = sum(self.totals.values())
        c = self.counts.get(shape, {})
        s = math.log(n / total)
        for f in FEATURES:
            p = (c.get(f, 0) + 1) / (n + 2)
            s += math.log(p if f in feats else 1 - p)
        return s

    def rank(self, shapes: list, feats: frozenset) -> list:
        """``shapes`` reordered by recognition score; unknown shapes keep their order after the known ones."""
        scored = [(self.log_score(s, feats), i, s) for i, s in enumerate(shapes)]
        known = sorted((t for t in scored if t[0] is not None), key=lambda t: (-t[0], t[1]))
        return [s for _, _, s in known] + [s for sc, _, s in scored if sc is None]

    def to_json(self) -> dict:
        return {"counts": self.counts, "totals": self.totals}

    @staticmethod
    def from_json(d: Optional[dict]) -> Optional["Recognizer"]:
        return Recognizer(d["counts"], d["totals"]) if d else None


def dream(shapes: list, var_names: list, rollout: Callable, seed: int, per_shape: int = 8,
          consts: Iterable = range(-5, 6), length: int = 16) -> list:
    """Fantasy problems: each shape instantiated with random variables and constants and rolled out by the
    domain's generative hook ``rollout(expr, rng, length) -> ys or None``; degenerate fantasies (undefined,
    constant or too short) are discarded. Returns ``(shape, features)`` training examples."""
    rng = random.Random(seed)
    consts = list(consts)
    out = []
    for shape in shapes:
        tree = parse_shape(shape)
        nv, nc = shape_slots(tree)
        kept = tries = 0
        while kept < per_shape and tries < 4 * per_shape:
            tries += 1
            e = fill_shape(tree, [rng.choice(var_names) for _ in range(nv)], [rng.choice(consts) for _ in range(nc)])
            if e is None:
                continue
            ys = rollout(e, rng, length)
            if not ys or len(ys) < 6 or len(set(ys)) < 2:
                continue
            out.append((shape, features(ys)))
            kept += 1
    return out


def sleep(rules: list, shapes: list, replays: list, dream_domains: dict, seed: int,
          dreamt: Optional[dict] = None) -> dict:
    """One sleep phase. ``rules``: ``(problem key, validated Expr)``; ``shapes``: validated shapes, most successful
    first; ``replays``: ``(shape, ys)`` of real validated problems; ``dream_domains``: ``{name: (var_names,
    rollout)}``. Each shape is dreamt with its own seed, so its dreams never change: ``dreamt`` (the ``"dreamt"``
    entry of the previous library) holds their statistics and only new shapes are dreamt. Returns the
    JSON-serialisable library."""
    inventions = promote(rules)
    rec = Recognizer()
    rec.train((shape, features(ys)) for shape, ys in replays)
    known = dict(dreamt or {})
    out_dreamt = {}
    for i, (name, (names, rollout)) in enumerate(sorted(dream_domains.items())):
        for shape in shapes:
            if "if_eq" in shape:
                continue
            key = f"{name}/{shape}"
            stats = known.get(key)
            if stats is None:
                d = Recognizer()
                d.train(dream([shape], names, rollout, seed * 1009 + i * 7 + zlib.crc32(shape.encode())))
                stats = {"n": d.totals.get(shape, 0), "counts": d.counts.get(shape, {})}
            out_dreamt[key] = stats
            if stats["n"]:
                rec.totals[shape] = rec.totals.get(shape, 0) + stats["n"]
                c = rec.counts.setdefault(shape, {})
                for f, k in sorted(stats["counts"].items()):
                    c[f] = c.get(f, 0) + k
    return {"inventions": [inv.to_json() for inv in inventions], "recognizer": rec.to_json(),
            "dreams": sum(st["n"] for st in out_dreamt.values()), "dreamt": out_dreamt, "replays": len(replays),
            "corpus": len(rules)}


def recognition_summary(rec: Optional[Recognizer]) -> dict:
    if rec is None:
        return {}
    by = defaultdict(list)
    for shape, c in rec.counts.items():
        n = rec.totals.get(shape, 1)
        by[shape] = sorted((f for f, k in c.items() if k >= 0.75 * n))
    return dict(sorted(by.items()))
