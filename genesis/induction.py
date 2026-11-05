"""Learning engine: online Bayesian program induction driven by prediction error.

A :class:`RuleLearner` owns one learning problem: "how does target variable y
depend on the perceived variables?". It keeps a pool of hypotheses (expressions
composed of innate primitives), weights them with an Occam (description length)
prior and a noise-tolerant likelihood, and competes them against a *null*
hypothesis "y is unpredictable from what I perceive".

Key properties
* Surprise-driven: a costly re-induction only happens when the current model
  mispredicts (prediction error), with metacognitive backoff when repeated
  re-inductions keep failing.
* Knowledge is only *validated* after it correctly predicts observations that
  were not used to construct it (forward predictions), never from fit alone.
* Hypotheses that agree on every probe input are merged into one behavioural
  equivalence class, so ``(x + x)`` and ``(x * 2)`` do not split confidence.
* Procedural memory (meta-learning): rule *shapes* that were validated before
  are tried first. If an instantiation of a familiar shape explains all the
  data, the exhaustive search is skipped; otherwise it runs as usual.
"""

from __future__ import annotations

import math
import operator
import random
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import itertools

from .expressions import (Expr, _trivial, app, const, enumerate_programs, fill_shape, if_eq, parse_shape,
                          shape_slots)
from .innate import INNATE_CONSTANTS


@dataclass
class InductionConfig:
    max_size: int = 5
    allow_conditionals: bool = False
    branch_max_size: int = 3
    noise: float = 0.02
    outcome_range: int = 100
    null_prior: float = 0.2
    pool_size: int = 48
    max_candidates: int = 60000
    max_seconds: float = 5.0
    validation_k: int = 4
    validation_posterior: float = 0.9
    max_mismatch_frac: float = 0.2
    unpredictable_min_n: int = 8
    max_backoff: int = 8
    shape_bias: float = 1.0
    n_probes: int = 24
    max_proposed_constants: int = 12
    library_max_shapes: int = 12
    library_max_instances: int = 20000


@dataclass
class Hypothesis:
    expr: Expr
    log_prior: float
    n_ok: int = 0
    n_bad: int = 0
    fwd_ok: int = 0
    fwd_bad: int = 0
    born_n: int = 0
    probe_sig: tuple = ()

    def to_json(self) -> dict:
        return {
            "expr": self.expr.to_json(), "log_prior": self.log_prior, "n_ok": self.n_ok, "n_bad": self.n_bad,
            "fwd_ok": self.fwd_ok, "fwd_bad": self.fwd_bad, "born_n": self.born_n,
        }

    @staticmethod
    def from_json(d: dict) -> "Hypothesis":
        return Hypothesis(Expr.from_json(d["expr"]), d["log_prior"], d["n_ok"], d["n_bad"], d["fwd_ok"],
                          d["fwd_bad"], d["born_n"])


@dataclass
class Prediction:
    value: Optional[int]
    confidence: float
    p_null: float
    distribution: list  # [(value, prob)] for hypothesis-backed values, descending
    hypothesis: Optional[str]

    def prob_of(self, y: int, outcome_range: int) -> float:
        p = dict(self.distribution).get(y, 0.0)
        return p + self.p_null / outcome_range


@dataclass
class LearnResult:
    prediction: Prediction
    correct: bool
    surprise_bits: float
    reinduced: bool
    status_before: str
    status_after: str
    refuted: list = field(default_factory=list)


def _has_trivial(e: Expr) -> bool:
    if e.op in ("var", "const"):
        return False
    if e.op != "if_eq" and _trivial(e.op, e.args[0], e.args[1]):
        return True
    return any(_has_trivial(a) for a in e.args)


def _logsumexp(xs: list) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class RuleLearner:
    def __init__(self, key: str, var_names: list, cfg: InductionConfig,
                 shape_prior: Optional[Callable[[str], float]] = None,
                 probe_fn: Optional[Callable[["RuleLearner", Expr], list]] = None,
                 shape_library: Optional[Callable[[], list]] = None):
        self.key = key
        # shape_library() -> shapes from procedural memory, most successful first.
        self.shape_library = shape_library
        self.library_searches = 0
        self.full_searches = 0
        # probe_fn generates the inputs on which behavioural equivalence is judged.
        # Default: independent random values (appropriate when the agent can intervene).
        self.probe_fn = probe_fn
        self.var_names = list(var_names)
        self.cfg = cfg
        self.shape_prior = shape_prior or (lambda shape: 0.0)
        self.data: list = []
        self.episodes: list = []
        self.provenance: list = []
        self.hyps: list = []
        self.backoff = 1
        self.last_induction_n = -10**9
        self.n_inductions = 0
        self.expressions_evaluated = 0
        self.induction_seconds = 0.0
        self.status_history: list = []
        self.refutations: list = []
        self._prev_top: Optional[Hypothesis] = None
        self._probes: list = []
        self._status = "unknown"

    # ------------------------------------------------------------------ beliefs
    def posterior(self) -> tuple:
        cfg = self.cfg
        n = len(self.data)
        l_ok = math.log(1 - cfg.noise)
        l_bad = math.log(cfg.noise / cfg.outcome_range)
        null = math.log(cfg.null_prior) + n * math.log(1 / cfg.outcome_range)
        scores = [math.log(1 - cfg.null_prior) + h.log_prior + h.n_ok * l_ok + h.n_bad * l_bad for h in self.hyps]
        z = _logsumexp(scores + [null])
        post = sorted(((h, math.exp(s - z)) for h, s in zip(self.hyps, scores)), key=lambda t: -t[1])
        return post, math.exp(null - z)

    def best(self) -> tuple:
        """(top hypothesis, mass of its behavioural equivalence class, p_null)."""
        post, p_null = self.posterior()
        if not post:
            return None, 0.0, p_null
        top = post[0][0]
        mass = sum(p for h, p in post if h.probe_sig == top.probe_sig)
        return top, mass, p_null

    def predict(self, inputs: dict) -> Prediction:
        post, p_null = self.posterior()
        dist: dict = defaultdict(float)
        for h, p in post:
            v = h.expr.evaluate(inputs)
            if v is None:
                p_null += p
            else:
                dist[v] += p
        ranked = sorted(dist.items(), key=lambda t: -t[1])
        if not ranked:
            return Prediction(None, 0.0, p_null, [], None)
        value = ranked[0][0]
        top = next((str(h.expr) for h, _ in post if h.expr.evaluate(inputs) == value), None)
        return Prediction(value, ranked[0][1], p_null, ranked, top)

    def status(self) -> str:
        if not self.data:
            return "unknown"
        top, mass, p_null = self.best()
        if p_null >= 0.9 and len(self.data) >= self.cfg.unpredictable_min_n:
            return "unpredictable"
        if top is not None:
            fwd = top.fwd_ok + top.fwd_bad
            if (mass >= self.cfg.validation_posterior and top.fwd_ok >= self.cfg.validation_k
                    and top.fwd_bad <= int(0.1 * fwd)):
                return "validated"
            if mass >= 0.5:
                return "candidate"
        return "unknown"

    # ----------------------------------------------------------------- learning
    def _tol(self, n: int) -> int:
        return int(self.cfg.max_mismatch_frac * n) if n >= 5 else 0

    def observe(self, inputs: dict, y: int, episode_id: Optional[int] = None,
                provenance: str = "observation") -> LearnResult:
        status_before = self._status
        pred = self.predict(inputs)
        correct = pred.value == y
        p_y = max(pred.prob_of(y, self.cfg.outcome_range), 1e-12)
        surprise = -math.log2(p_y)
        top_before, _, _ = self.best()

        for h in self.hyps:
            if h.expr.evaluate(inputs) == y:
                h.n_ok += 1
                h.fwd_ok += 1
            else:
                h.n_bad += 1
                h.fwd_bad += 1
        self.data.append((dict(inputs), y))
        self.episodes.append(episode_id)
        self.provenance.append(provenance)
        n = len(self.data)
        tol = self._tol(n)
        refuted = []
        survivors = []
        for h in self.hyps:
            if h.n_bad <= tol:
                survivors.append(h)
            elif h is top_before:
                refuted.append(str(h.expr))
                self.refutations.append({"n": n, "hypothesis": str(h.expr), "episode": episode_id,
                                         "forward_ok": h.fwd_ok, "status_before": status_before})
        self.hyps = survivors

        reinduced = False
        top_wrong = top_before is not None and top_before.expr.evaluate(inputs) != y
        if (not correct or not self.hyps or top_wrong) and n - self.last_induction_n >= self.backoff:
            self._induce()
            reinduced = True
        self._status = self.status()
        if self._status != status_before:
            self.status_history.append({"n": n, "status": self._status, "episode": episode_id})
        return LearnResult(pred, correct, surprise, reinduced, status_before, self._status, refuted)

    def _propose_constants(self, inputs: list, ys: list) -> set:
        proposals: Counter = Counter()
        for inp, y in zip(inputs, ys):
            proposals[y] += 1
            for v in self.var_names:
                x = inp.get(v)
                if x is None:
                    continue
                proposals[y - x] += 1
                if x not in (0, 1, -1) and y % x == 0:
                    proposals[y // x] += 1
        consts = set(INNATE_CONSTANTS)
        common = [c for c, _ in proposals.most_common(self.cfg.max_proposed_constants) if abs(c) <= 10**6]
        consts |= set(common)
        # Contrast between salient values (e.g. a regular step and a wrap-around jump).
        consts |= {abs(a - b) for i, a in enumerate(common[:4]) for b in common[i + 1:4] if a != b}
        if ys:
            consts.add(max(ys) + 1)  # perceived range: candidate modulus
        return consts

    def _make_probes(self) -> None:
        if self.probe_fn is not None and self.hyps:
            self._probes = self.probe_fn(self, self.hyps[0].expr)
            return
        rng = random.Random(zlib.crc32(self.key.encode()))
        ranges = {}
        for v in self.var_names:
            vals = [inp[v] for inp, _ in self.data if inp.get(v) is not None] or [0]
            ranges[v] = (min(vals) - 2, max(vals) + 2)
        self._probes = [{v: rng.randint(*ranges[v]) for v in self.var_names} for _ in range(self.cfg.n_probes)]

    def _probe_sig(self, e: Expr) -> tuple:
        return tuple(e.evaluate(p) for p in self._probes)

    def _base_prior(self, e: Expr, counts: dict) -> float:
        return -2 * math.log(e.size) - math.log(counts.get(e.size, 1))

    def _induce(self) -> None:
        cfg = self.cfg
        # Back off when the previous search failed: nothing survived, or its best rule kept failing.
        failed = self._prev_top is None or self._prev_top.fwd_ok < 2
        if self.n_inductions and failed and len(self.data) >= cfg.unpredictable_min_n:
            self.backoff = min(self.backoff * 2, cfg.max_backoff)
        else:
            self.backoff = 1
        t0 = time.perf_counter()
        inputs = [d for d, _ in self.data]
        ys = tuple(y for _, y in self.data)
        n = len(ys)
        tol = self._tol(n)
        consts = self._propose_constants(inputs, ys)
        found = self._library_search(inputs, ys, tol, consts)
        if any(bad == 0 for _, bad, _ in found):
            self.library_searches += 1
        else:
            self.full_searches += 1
            found = self._full_search(inputs, ys, tol, consts)
        self._merge(found, n)
        self._prev_top = self.hyps[0] if self.hyps else None
        self.last_induction_n = n
        self.n_inductions += 1
        self.induction_seconds += time.perf_counter() - t0

    def _approx_prior(self, e: Expr, n_leaves: int) -> float:
        k = (e.size - 1) // 2
        n_size = math.comb(2 * k, k) // (k + 1) * 4 ** k * n_leaves ** (k + 1)
        return -2 * math.log(e.size) - math.log(max(n_size, 1))

    def _library_search(self, inputs: list, ys: tuple, tol: int, consts: set) -> list:
        """Instantiate previously successful shapes; return those that explain the data."""
        if self.shape_library is None:
            return []
        cfg = self.cfg
        consts = sorted(consts)
        n_leaves = len(self.var_names) + len(consts)
        found, seen = [], set()
        for shape in self.shape_library()[: cfg.library_max_shapes]:
            tree = parse_shape(shape)
            if not cfg.allow_conditionals and "if_eq" in shape:
                continue
            nv, nc = shape_slots(tree)
            # "X + C": fit the additive constant from the residual instead of enumerating it.
            fit_bias = isinstance(tree, tuple) and tree[0] == "add" and tree[1][1] == "C"
            n_c_enum = nc - 1 if fit_bias else nc
            if len(self.var_names) ** nv * len(consts) ** n_c_enum > cfg.library_max_instances:
                continue
            for vs in itertools.product(self.var_names, repeat=nv):
                for cs in itertools.product(consts, repeat=n_c_enum):
                    if fit_bias:
                        body = fill_shape(tree[1][0], vs, cs)
                        if body is None:
                            continue
                        sig = [body.evaluate(d) for d in inputs]
                        if None in sig:
                            continue
                        r = Counter(y - v for y, v in zip(ys, sig)).most_common(1)[0][0]
                        e = app("add", body, const(r))
                    else:
                        e = fill_shape(tree, vs, cs)
                    self.expressions_evaluated += 1
                    if e is None or _has_trivial(e) or e.size > cfg.max_size + (2 + 2 * cfg.branch_max_size
                                                                                if e.op == "if_eq" else 0):
                        continue
                    k = str(e)
                    if k in seen:
                        continue
                    seen.add(k)
                    bad = sum(e.evaluate(d) != y for d, y in zip(inputs, ys))
                    if bad <= tol:
                        found.append((e, bad, self._approx_prior(e, n_leaves)))
        return found

    def _full_search(self, inputs: list, ys: tuple, tol: int, consts: set) -> list:
        cfg = self.cfg
        n = len(ys)
        en = enumerate_programs(inputs, self.var_names, consts, cfg.max_size, cfg.max_candidates, cfg.max_seconds)
        self.expressions_evaluated += en.evaluated
        found = []
        ne = operator.ne
        offsets: dict = {}  # insertion-ordered: set order would depend on PYTHONHASHSEED
        for e, sig in en.candidates:
            if tol == 0:
                if sig == ys:
                    found.append((e, 0, self._base_prior(e, en.counts)))
            else:
                bad = sum(map(ne, sig, ys))
                if bad <= tol:
                    found.append((e, bad, self._base_prior(e, en.counts)))
            # Bias fitting: if y - e(x) is the same constant r on every observation, propose e + r.
            if n >= 2 and e.op != "const" and e.size <= cfg.max_size - 2:
                r = ys[0] - sig[0]
                if r != 0 and all(y - s == r for y, s in zip(ys, sig)):
                    offsets.setdefault((e, r), None)
        for e, r in offsets:
            fe = app("add", e, const(r))
            found.append((fe, 0, self._base_prior(fe, en.counts)))
        if cfg.allow_conditionals and not any(b == 0 for _, b, _ in found):
            found += self._conditionals(en, inputs, ys)
        return found

    def _merge(self, found: list, n: int) -> None:
        cfg = self.cfg
        existing = {str(h.expr): h for h in self.hyps}
        fresh = []
        for e, bad, lp in found:
            key = str(e)
            if key in existing:
                continue
            existing[key] = None
            fresh.append(Hypothesis(e, lp + cfg.shape_bias * self.shape_prior(e.shape()), n - bad, bad, 0, 0, n))
        pool = [h for h in self.hyps] + fresh
        self.hyps = pool
        post, _ = self.posterior()
        self.hyps = [h for h, _ in post[: cfg.pool_size]]
        self._make_probes()
        for h in self.hyps:
            h.probe_sig = self._probe_sig(h.expr)

    def _conditionals(self, en, inputs: list, ys: tuple) -> list:
        cfg = self.cfg
        blocks = [(e, sig) for e, sig in en.candidates if e.size <= cfg.branch_max_size]
        conditions = []
        for q in self.var_names:
            for c in sorted({inp[q] for inp in inputs if inp.get(q) is not None}):
                i1 = [i for i, inp in enumerate(inputs) if inp.get(q) == c]
                if 0 < len(i1) < len(inputs):
                    conditions.append((q, c, i1))
        if not conditions:
            return []
        log_nc = math.log(len(conditions))
        out = []
        for q, c, i1 in conditions:
            s1 = set(i1)
            i0 = [i for i in range(len(ys)) if i not in s1]
            y1 = tuple(ys[i] for i in i1)
            y0 = tuple(ys[i] for i in i0)
            m1 = [e for e, sig in blocks if sig[i1[0]] == y1[0] and tuple(sig[i] for i in i1) == y1][:3]
            if not m1:
                continue
            m0 = [e for e, sig in blocks if sig[i0[0]] == y0[0] and tuple(sig[i] for i in i0) == y0][:3]
            for e1 in m1:
                for e0 in m0:
                    if e1 == e0:
                        continue
                    lp = (-log_nc - math.log(4) + self._base_prior(e1, en.counts) + self._base_prior(e0, en.counts))
                    out.append((if_eq(q, c, e1, e0), 0, lp))
        return out

    # -------------------------------------------------------------- persistence
    def to_state(self) -> dict:
        return {
            "key": self.key, "var_names": self.var_names, "data": self.data, "episodes": self.episodes,
            "provenance": self.provenance, "hyps": [h.to_json() for h in self.hyps], "backoff": self.backoff,
            "last_induction_n": self.last_induction_n, "n_inductions": self.n_inductions,
            "expressions_evaluated": self.expressions_evaluated, "induction_seconds": self.induction_seconds,
            "status_history": self.status_history, "refutations": self.refutations, "status": self._status,
            "library_searches": self.library_searches, "full_searches": self.full_searches,
        }

    @classmethod
    def from_state(cls, state: dict, cfg: InductionConfig,
                   shape_prior: Optional[Callable[[str], float]] = None,
                   probe_fn: Optional[Callable[["RuleLearner", Expr], list]] = None,
                   shape_library: Optional[Callable[[], list]] = None) -> "RuleLearner":
        lr = cls(state["key"], state["var_names"], cfg, shape_prior, probe_fn, shape_library)
        lr.library_searches = state.get("library_searches", 0)
        lr.full_searches = state.get("full_searches", 0)
        lr.data = [(dict(d), y) for d, y in state["data"]]
        lr.episodes = list(state["episodes"])
        lr.provenance = list(state["provenance"])
        lr.hyps = [Hypothesis.from_json(h) for h in state["hyps"]]
        lr.backoff = state["backoff"]
        lr.last_induction_n = state["last_induction_n"]
        lr.n_inductions = state["n_inductions"]
        lr.expressions_evaluated = state["expressions_evaluated"]
        lr.induction_seconds = state["induction_seconds"]
        lr.status_history = state["status_history"]
        lr.refutations = state["refutations"]
        lr._status = state["status"]
        lr._make_probes()
        for h in lr.hyps:
            h.probe_sig = lr._probe_sig(h.expr)
        return lr
