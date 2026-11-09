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
  are tried first, in the order a recognition model trained during sleep
  (genesis.library) finds most plausible for the data at hand. If an
  instantiation of a familiar shape explains all the data, the exhaustive
  search is skipped; otherwise it runs as usual.
* Invented primitives (genesis.library): sub-programs promoted during sleep are
  extra leaves of the exhaustive search, so rules built on them are shorter.
  Rules found with them are inlined: beliefs only ever use innate primitives.
"""

from __future__ import annotations

import heapq
import math
import random
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import itertools

from .expressions import (Expr, _trivial, app, const, enumerate_programs, fill_shape, if_eq, parse_shape,
                          shape_slots)
from .library import RESERVED, features as task_features, inline
from .innate import INNATE_CONSTANTS, VALUE_LIMIT
from .linear import dense_windows, linear_rules
from .search import match_shape, search_programs


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
    # Forward-prediction error rate still compatible with "the rule is right, the world is noisy". Errors must
    # also be unstructured (see RuleLearner._structured_errors), so a systematic misfit is never validated.
    validation_max_error: float = 0.25
    # ... or, when the observed rate drifts above it (a burst of corruption), Wald's sequential test between "errs
    # at validation_max_error" and "errs at sprt_null_error" (a rule no better than a coin) must accept the former
    # at level validation_sprt_alpha: a long forward record can carry a few more errors. 0 disables the test.
    validation_sprt_alpha: float = 0.01
    sprt_null_error: float = 0.5
    # A rule survives while it is right at least as often as wrong (far beyond the chance agreement of noise):
    # bursts of corruption well above the world's nominal rate then pause it instead of refuting it.
    max_mismatch_frac: float = 0.5
    # New rules are proposed only with fewer mismatches: a wider search tolerance floods every search on noisy
    # data with half-right candidates (4x the CPU on deeper worlds with 10% noise) for no measured gain.
    search_mismatch_frac: float = 0.4
    # Deepening (genesis.search): when no enumerated rule mispredicts at most deepen_below * n observations,
    # look for rules of deepen_size nodes by meet-in-the-middle, within the same candidate budget.
    deepen_size: int = 7
    deepen_max_size: int = 9  # iterative deepening: deepen_size, deepen_size + 2, ... up to this size
    deepen_below: float = 0.25
    deepen_min_n: int = 6
    # Short windows (4 rows, 8 of them) keep finding a clean window when a few observations are corrupted.
    deepen_window: int = 4
    deepen_windows: int = 8
    deepen_growth: float = 0.5   # after a deep search at n, the next one waits until n * (1 + deepen_growth)
    # When the recognition model learned during sleep judges the series rule-less with at least this probability
    # (after noise_min_n observations), a due deep search (deep, linear and invention stages) is postponed by
    # noise_deepen_growth * n observations: deeper searches are delayed on likely noise, never forbidden.
    noise_recognition: float = 0.9
    noise_min_n: int = 10
    noise_deepen_growth: float = 1.0
    # Exact sparse linear induction (genesis.linear), on the deepening schedule: y = a1*f1 + ... + ak*fk + c over
    # the perceived variables and their pairwise products (at most one per rule), with k <= linear_terms and
    # small integer coefficients, solved exactly on the deepening windows. 0 disables it.
    linear_terms: int = 4
    linear_products: bool = True
    linear_max_coef: int = 16
    linear_min_support: int = 3  # observations a solution must explain beyond those it was solved on
    linear_spare: int = 1  # rows to spare per window: solve on every k of k + spare rows (steps over corruption)
    linear_windows: int = 4
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


def _in_range(inputs: dict, y: int) -> bool:
    return -VALUE_LIMIT <= y <= VALUE_LIMIT and all(
        v is None or -VALUE_LIMIT <= v <= VALUE_LIMIT for v in inputs.values())


def _logsumexp(xs: list) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class RuleLearner:
    def __init__(self, key: str, var_names: list, cfg: InductionConfig,
                 shape_prior: Optional[Callable[[str], float]] = None,
                 probe_fn: Optional[Callable[["RuleLearner", Expr], list]] = None,
                 shape_library: Optional[Callable[[], list]] = None,
                 learned_library: Optional[Callable[[], tuple]] = None):
        self.key = key
        # shape_library() -> shapes from procedural memory, most successful first.
        self.shape_library = shape_library
        # learned_library() -> (invented primitives, recognition model or None), both learned during sleep.
        self.learned_library = learned_library
        self.invention_searches = 0
        self.invention_hits = 0
        self.library_searches = 0
        self.full_searches = 0
        self.deep_searches = 0
        self.next_deep_n = 0
        self.linear_searches = 0
        self.linear_hits = 0
        self.noise_delays = 0
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
        self.last_search: dict = {}
        # Memos of posterior(), best() and predict(), valid while the posterior is unchanged (same object).
        self._post_memo: Optional[tuple] = None
        self._best_memo: Optional[tuple] = None
        self._pred_memo: Optional[tuple] = None

    # ------------------------------------------------------------------ beliefs
    def posterior(self) -> tuple:
        """``(((hypothesis, probability), ...) most probable first, p_null)``, memoised: it is asked for many
        times between two observations. The memo holds the hypotheses themselves, so their ids stay unique."""
        cfg = self.cfg
        n = len(self.data)
        key = (n, cfg.noise, cfg.outcome_range, cfg.null_prior,
               tuple((id(h), h.n_ok, h.n_bad, h.log_prior) for h in self.hyps))
        memo = self._post_memo
        if memo is not None and memo[0] == key:
            return memo[2]
        out = self._posterior(n)
        self._post_memo = (key, tuple(self.hyps), out)
        return out

    def _posterior(self, n: int) -> tuple:
        cfg = self.cfg
        l_ok = math.log(1 - cfg.noise)
        l_bad = math.log(cfg.noise / cfg.outcome_range)
        null = math.log(cfg.null_prior) + n * math.log(1 / cfg.outcome_range)
        scores = [math.log(1 - cfg.null_prior) + h.log_prior + h.n_ok * l_ok + h.n_bad * l_bad for h in self.hyps]
        z = _logsumexp(scores + [null])
        post = tuple(sorted(((h, math.exp(s - z)) for h, s in zip(self.hyps, scores)), key=lambda t: -t[1]))
        return post, math.exp(null - z)

    def best(self) -> tuple:
        """(top hypothesis, mass of its behavioural equivalence class, p_null)."""
        result = self.posterior()
        memo = self._best_memo
        if memo is not None and memo[0] is result:
            return memo[1]
        post, p_null = result
        if not post:
            out = (None, 0.0, p_null)
        else:
            top = post[0][0]
            out = (top, sum(p for h, p in post if h.probe_sig == top.probe_sig), p_null)
        self._best_memo = (result, out)
        return out

    def predict(self, inputs: dict) -> Prediction:
        result = self.posterior()
        try:
            key = tuple(sorted(inputs.items()))
        except TypeError:
            key = None
        memo = self._pred_memo
        if key is not None and memo is not None and memo[0] is result and memo[1] == key:
            return memo[2]
        post, p_null = result
        dist: dict = defaultdict(float)
        values = []
        for h, p in post:
            v = h.expr.evaluate(inputs, True)
            values.append(v)
            if v is None:
                p_null += p
            else:
                dist[v] += p
        ranked = sorted(dist.items(), key=lambda t: -t[1])
        if not ranked:
            pred = Prediction(None, 0.0, p_null, [], None)
        else:
            value = ranked[0][0]
            top = next((str(h.expr) for (h, _), v in zip(post, values) if v == value), None)
            pred = Prediction(value, ranked[0][1], p_null, ranked, top)
        if key is not None:
            self._pred_memo = (result, key, pred)
        return pred

    def status(self) -> str:
        if not self.data:
            return "unknown"
        top, mass, p_null = self.best()
        if p_null >= 0.9 and len(self.data) >= self.cfg.unpredictable_min_n and not self._promising(top):
            return "unpredictable"
        if top is not None:
            fwd = top.fwd_ok + top.fwd_bad
            if (mass >= self.cfg.validation_posterior and top.fwd_ok >= self.cfg.validation_k
                    and (top.fwd_bad <= int(self.cfg.validation_max_error * fwd) or self._sprt_accepts(top))
                    and not self._structured_errors(top)):
                return "validated"
            if mass >= 0.5:
                return "candidate"
        return "unknown"

    def _promising(self, h: Optional[Hypothesis]) -> bool:
        """A surviving rule that predicted the future far better than chance (at least two forward successes, more
        successes than failures) keeps the question open, even while the null still dominates the posterior: a
        long rule pays its Occam prior back only over several more observations, and judging the stream
        unpredictable now would withdraw the attention that would decide it. Noise almost never yields one (each
        forward success is a 1 / outcome_range coincidence)."""
        return h is not None and h.fwd_ok >= 2 and h.fwd_ok > h.fwd_bad

    def _sprt_accepts(self, h: Hypothesis) -> bool:
        """Wald's log-likelihood ratio of the forward record under "errs at validation_max_error" against "errs at
        sprt_null_error" reaches log(1 / validation_sprt_alpha). Its drift is positive only below an error rate of
        about 37% (with the defaults), and the record grows with every forward prediction."""
        alpha, p1, p0 = self.cfg.validation_sprt_alpha, self.cfg.validation_max_error, self.cfg.sprt_null_error
        if alpha <= 0 or not 0 < p1 < p0 < 1:
            return False
        llr = h.fwd_ok * math.log((1 - p1) / (1 - p0)) + h.fwd_bad * math.log(p1 / p0)
        return llr >= math.log(1 / alpha)

    def _structured_errors(self, h: Hypothesis) -> bool:
        """Noise is unstructured: when most of a rule's errors share one residual (a missed wrap-around, an
        off-by-one branch), the rule is incomplete and must not be validated, whatever its error rate."""
        if h.n_bad < 2:
            return False
        residuals: Counter = Counter()
        for d, y in self.data:
            v = h.expr.evaluate(d, True)
            if v != y:
                residuals[None if v is None else y - v] += 1
        if not residuals:
            return False
        return residuals.most_common(1)[0][1] > max(1, sum(residuals.values()) // 2)

    # ----------------------------------------------------------------- learning
    def _tol(self, n: int, search: bool = False) -> int:
        frac = min(self.cfg.search_mismatch_frac, self.cfg.max_mismatch_frac) if search else self.cfg.max_mismatch_frac
        return int(frac * n) if n >= 5 else 0

    def observe(self, inputs: dict, y: int, episode_id: Optional[int] = None,
                provenance: str = "observation") -> LearnResult:
        status_before = self._status
        pred = self.predict(inputs)
        correct = pred.value == y
        p_y = max(pred.prob_of(y, self.cfg.outcome_range), 1e-12)
        surprise = -math.log2(p_y)
        top_before, _, _ = self.best()

        for h in self.hyps:
            if h.expr.evaluate(inputs, True) == y:
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
        top_wrong = top_before is not None and top_before.expr.evaluate(inputs, True) != y
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
        # Per input: contrast between its two most frequent residuals (a regular step and a wrap-around jump).
        for v in self.var_names:
            res = Counter(y - inp[v] for inp, y in zip(inputs, ys) if inp.get(v) is not None).most_common(2)
            if len(res) == 2 and res[1][1] >= 2 and abs(res[0][0] - res[1][0]) <= 10**6:
                consts.add(abs(res[0][0] - res[1][0]))
        if ys:
            # Perceived range: candidate modulus. The few largest values guard against one corrupted maximum.
            consts |= {v + 1 for v in sorted(set(ys))[-3:]}
        return consts | self._propose_moduli(inputs, ys)

    def _propose_moduli(self, inputs: list, ys: list, max_moduli: int = 2) -> set:
        """Moduli that are never observed: y = (a * x + c) % m keeps every y - a * x in one residue class, so the
        differences of y - a * x over a few consecutive observations are multiples of m, and their gcd reveals it
        (with c, the common residue), although m exceeds every observed value. Multipliers are the innate small
        constants; a modulus must exceed the outputs of its window and recur in at least two windows (one corrupted
        observation spoils only the windows containing it), and at most ``max_moduli`` are proposed."""
        n = len(ys)
        if n < 6:
            return set()
        support: Counter = Counter()
        offsets: dict = defaultdict(Counter)
        for v in self.var_names:
            xs = [inp.get(v) for inp in inputs]
            for a in sorted(c for c in INNATE_CONSTANTS if c > 0):
                for w in dense_windows(n, 5, 8):
                    if any(xs[i] is None for i in w):
                        continue
                    r = [ys[i] - a * xs[i] for i in w]
                    g = 0
                    for x in r[1:]:
                        g = math.gcd(g, x - r[0])
                    if g >= 2 and max(ys[i] for i in w) < g <= 10**6:
                        support[g] += 1
                        offsets[g][r[0] % g] += 1
        out = set()
        for g, k in sorted(support.items(), key=lambda t: (-t[1], t[0]))[:max_moduli]:
            if k >= 2:
                out |= {g, offsets[g].most_common(1)[0][0]}
        return out

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
        return tuple(e.evaluate(p, True) for p in self._probes)

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
        n = len(self.data)
        tol = self._tol(n, search=True)
        # The search kernels use bounded arithmetic; beliefs are exact. Observations beyond VALUE_LIMIT (a
        # geometric stream past 10^18) are left out of the search and every proposal is re-scored on them
        # exactly, so a rule found on the in-range data keeps its evidence instead of being refuted by overflow.
        rows, check, partial = self.data, [], True
        big = [i for i, (d, y) in enumerate(self.data) if not _in_range(d, y)]
        if big:
            skip = set(big)
            kept = [r for i, r in enumerate(self.data) if i not in skip]
            if len(kept) >= 5:
                rows, check = kept, [self.data[i] for i in big]
            else:
                check, partial = self.data, False
        inputs = [d for d, _ in rows]
        ys = tuple(y for _, y in rows)
        consts = self._propose_constants(inputs, ys)
        found = self._rescore(self._library_search(inputs, ys, tol, consts), check, partial, tol)
        if any(bad == 0 for _, bad, _ in found):
            self.library_searches += 1
        else:
            self.full_searches += 1
            found = self._rescore(self._full_search(inputs, ys, tol, consts), check, partial, tol)
        self._merge(found, n)
        self._prev_top = self.hyps[0] if self.hyps else None
        self.last_induction_n = n
        self.n_inductions += 1
        self.induction_seconds += time.perf_counter() - t0

    @staticmethod
    def _rescore(found: list, check: list, partial: bool, tol: int) -> list:
        """Exact mismatches on ``check``: added to the search count (``partial``) or replacing it."""
        if not check:
            return found
        out = []
        for e, bad, lp in found:
            miss = sum(e.evaluate(d, True) != y for d, y in check)
            bad = bad + miss if partial else miss
            if bad <= tol:
                out.append((e, bad, lp))
        return out

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
        size_bound = max(cfg.max_size, cfg.deepen_size, cfg.deepen_max_size) if cfg.deepen_size > cfg.max_size \
            else cfg.max_size
        found, seen = [], set()
        shapes = self.shape_library()
        rec = self.learned_library()[1] if self.learned_library is not None else None
        if rec is not None and len(shapes) > cfg.library_max_shapes:
            shapes = rec.rank(shapes, task_features(list(ys)))
        for shape in shapes[: cfg.library_max_shapes]:
            tree = parse_shape(shape)
            if not cfg.allow_conditionals and "if_eq" in shape:
                continue
            nv, nc = shape_slots(tree)
            # "X + C": fit the additive constant from the residual instead of enumerating it.
            fit_bias = isinstance(tree, tuple) and tree[0] == "add" and tree[1][1] == "C"
            n_c_enum = nc - 1 if fit_bias else nc
            if len(self.var_names) ** nv * len(consts) ** n_c_enum > cfg.library_max_instances:
                continue
            body = tree[1][0] if fit_bias else tree
            matched = match_shape(body, inputs, ys, self.var_names, consts, tol, fit_bias)
            if matched is not None:
                # Kernel path (genesis.search): only the instances within tolerance become expressions.
                hits, evaluated, _ = matched
                self.expressions_evaluated += evaluated
                for vs, cs, r, bad in hits:
                    e = fill_shape(body, vs, cs)
                    if fit_bias:
                        e = app("add", e, const(r))
                    if _has_trivial(e) or e.size > size_bound:
                        continue
                    k = str(e)
                    if k not in seen:
                        seen.add(k)
                        found.append((e, bad, self._approx_prior(e, n_leaves)))
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
                    if e is None or _has_trivial(e) or e.size > size_bound + (
                            2 + 2 * cfg.branch_max_size if e.op == "if_eq" else 0):
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
        n = len(self.data)
        deepen = cfg.deepen_size > cfg.max_size and n >= max(cfg.deepen_min_n, self.next_deep_n)
        if deepen and self._likely_noise():
            # The series looks rule-less to the recognition model learned during sleep: the deeper searches (deep,
            # linear, inventions) are postponed, never forbidden.
            deepen = False
            self.noise_delays += 1
            self.next_deep_n = n + max(1, int(n * cfg.noise_deepen_growth))
        deepen_bad = int(cfg.deepen_below * len(ys)) if deepen else -1
        top = self.best()[0] if deepen else None
        if top is not None and top.n_bad and self._structured_errors(top):
            # The best rule is known to be incomplete (its errors share one residual: a missed cycle, a wrap-around):
            # it must not stop the search for a deeper rule that explains more, however few its errors are.
            deepen_bad = min(deepen_bad, top.n_bad - 1)
        if deepen and cfg.linear_terms:
            # Linear rules are solved for, not enumerated, as the first way of looking deeper: only when the
            # enumerated rules leave more than deepen_bad observations unexplained (when the kernel would deepen).
            # When a linear rule explains the data that well, the kernel's deepening phase (by far the most
            # expensive part) only looks for shorter rules (t * (t + 3) is shorter than t * t + 3 * t, and Occam
            # must be able to see it); otherwise the search runs exactly as without the solver.
            found, counts, _ = self._search(inputs, ys, tol, consts, {}, -1)
            deep_ran = min((bad for _, bad, _ in found), default=len(ys) + 1) > deepen_bad
            if deep_ran:
                lin = self._linear(inputs, ys, tol, consts)
                hit = min((e.size for e, bad, _ in lin if bad <= deepen_bad), default=0)
                if hit:
                    self.linear_hits += 1
                    if hit - 2 >= cfg.deepen_size:
                        found, counts, _ = self._search(inputs, ys, tol, consts, {}, deepen_bad,
                                                        min(hit - 2, cfg.deepen_max_size))
                else:
                    found, counts, deep_ran = self._search(inputs, ys, tol, consts, {}, deepen_bad)
                found += lin
        else:
            found, counts, deep_ran = self._search(inputs, ys, tol, consts, {}, deepen_bad)
        # Second stage, only when the innate language found nothing that fits: the same search over a language
        # extended with the invented primitives. Problems the innate language solves cost exactly what they did,
        # and the stage follows the deepening schedule (it is one more way of looking deeper). Its deepening stops
        # at deepen_size: an invention already stands for several nodes, and deeper extended searches cost much
        # more (mostly on noise, which a learner cannot tell from an out-of-reach rule) for no measured gain.
        scheduled = deepen or cfg.deepen_size <= cfg.max_size
        good = int(cfg.deepen_below * len(ys))
        defs = self._usable_inventions(inputs) if scheduled and not any(bad <= good for _, bad, _ in found) else {}
        if defs:
            more, _, ran = self._search(inputs, ys, tol, consts, defs, deepen_bad, cfg.deepen_size)
            self.invention_searches += 1
            self.invention_hits += any(bad <= good for _, bad, _ in more)
            found += more
            deep_ran = deep_ran or ran
        if deep_ran:
            self.deep_searches += 1
            self.next_deep_n = n + max(1, int(n * cfg.deepen_growth))
        if cfg.allow_conditionals and not any(b == 0 for _, b, _ in found):
            blocks = enumerate_programs(inputs, self.var_names, consts, cfg.branch_max_size, cfg.max_candidates,
                                        cfg.max_seconds)
            found += self._conditionals(blocks, counts, inputs, ys)
        return found

    def _linear(self, inputs: list, ys: tuple, tol: int, consts: set) -> list:
        """Exact sparse linear rules (genesis.linear) with the Occam prior of their size."""
        cfg = self.cfg
        n = len(ys)
        rules, solved = linear_rules(inputs, ys, self.var_names,
                                     lambda length: dense_windows(n, max(length, cfg.deepen_window), cfg.linear_windows),
                                     tol, cfg.linear_terms, cfg.linear_products, cfg.linear_max_coef,
                                     min_support=cfg.linear_min_support, spare=cfg.linear_spare)
        self.linear_searches += 1
        self.expressions_evaluated += solved
        n_leaves = len(self.var_names) + len(consts)
        return [(e, bad, self._approx_prior(e, n_leaves)) for e, bad in rules]

    def _search(self, inputs: list, ys: tuple, tol: int, consts: set, defs: dict, deepen_bad: int,
                deep_max_size: int = 0) -> tuple:
        """Candidates within the mismatch tolerance, plus bias fits: when y - e(x) is the same constant r on all but
        <= tol observations, propose e + r (see genesis.search; the C kernel and the Python reference agree).
        ``defs`` are invented leaves: their values are extra inputs, and every result is inlined, while the Occam
        prior is that of the extended language (an invention is one node)."""
        cfg = self.cfg
        if defs:
            inputs = [{**d, **{k: e.evaluate(d) for k, e in defs.items()}} for d in inputs]
        sr = search_programs(inputs, ys, self.var_names + list(defs), sorted(consts), tol, cfg.max_size,
                             cfg.max_candidates, cfg.max_seconds, cfg.max_size - 2, deepen_bad=deepen_bad,
                             deep_size=cfg.deepen_size, window=cfg.deepen_window, n_windows=cfg.deepen_windows,
                             deep_max_size=deep_max_size or cfg.deepen_max_size)
        self.expressions_evaluated += sr.evaluated
        self.last_search = {"backend": sr.backend, "evaluated": sr.evaluated, "truncated": sr.truncated,
                            "seconds": round(sr.seconds, 4), "max_size": cfg.max_size, "deep_found": len(sr.deep),
                            "invented_leaves": len(defs)}
        found = [(e, bad, self._base_prior(e, sr.counts)) for e, bad in sr.matches]
        for e, r, bad in sr.offsets:
            fe = app("add", e, const(r))
            found.append((fe, bad, self._base_prior(fe, sr.counts)))
        found += [(e, bad, self._base_prior(e, sr.counts)) for e, bad in sr.deep]
        if defs:
            found = [(inline(e, defs), bad, lp) for e, bad, lp in found]
        return found, sr.counts, sr.deep_ran

    def _likely_noise(self) -> bool:
        rec = self.learned_library()[1] if self.learned_library is not None else None
        if rec is None or len(self.data) < self.cfg.noise_min_n:
            return False
        p = rec.p_noise(task_features([y for _, y in self.data]))
        return p is not None and p >= self.cfg.noise_recognition

    def _recognition(self) -> Callable[[str], float]:
        """Task-conditional prior from the recognition model: how much more plausible than uniform each familiar
        shape is for the observations at hand (never negative, zero for unfamiliar shapes)."""
        rec = self.learned_library()[1] if self.learned_library is not None else None
        if rec is None or not rec.totals:
            return lambda shape: 0.0
        feats = task_features([y for _, y in self.data])
        scores = {s: rec.log_score(s, feats) for s in rec.totals if s not in RESERVED}
        if not scores:
            return lambda shape: 0.0
        z = _logsumexp(list(scores.values()))
        floor = -math.log(len(scores))
        return lambda shape: max(0.0, scores[shape] - z - floor) if shape in scores else 0.0

    def _usable_inventions(self, inputs: list) -> dict:
        """Invented primitives whose variables are perceived here and whose values stay within the search bound on
        every observation, as ``{leaf name: definition}``."""
        if self.learned_library is None:
            return {}
        names = set(self.var_names)
        out = {}
        for inv in self.learned_library()[0]:
            if inv.expr.variables() <= names and all(inv.expr.evaluate(d) is not None for d in inputs):
                out[inv.name] = inv.expr
        return out

    def _merge(self, found: list, n: int) -> None:
        cfg = self.cfg
        existing = {str(h.expr): h for h in self.hyps}
        fresh = []
        recognition = None
        for e, bad, lp in found:
            key = str(e)
            if key in existing:
                continue
            existing[key] = None
            shape = e.shape()
            if recognition is None:
                recognition = self._recognition()
            fresh.append(Hypothesis(e, lp + cfg.shape_bias * (self.shape_prior(shape) + recognition(shape)),
                                    n - bad, bad, 0, 0, n))
        pool = [h for h in self.hyps] + fresh
        self.hyps = pool
        post, _ = self.posterior()
        self.hyps = [h for h, _ in post[: cfg.pool_size]]
        self._make_probes()
        for h in self.hyps:
            h.probe_sig = self._probe_sig(h.expr)
        self._best_memo = None  # equivalence classes follow the new probes

    def _conditionals(self, en, counts: dict, inputs: list, ys: tuple) -> list:
        """``if q == c then e1 else e0`` from small blocks. Each branch takes its exact blocks when there are
        any (the first three, in enumeration order); otherwise, on a branch with enough observations, the three
        blocks with the fewest mismatches within that branch's tolerance, so corrupted outcomes no longer hide a
        conditional rule. Mismatch sets are bitmasks: one popcount per (block, condition, branch)."""
        cfg = self.cfg
        n = len(ys)
        blocks = []
        for e, sig in en.candidates:
            if e.size <= cfg.branch_max_size:
                miss = 0
                for i, (v, y) in enumerate(zip(sig, ys)):
                    if v != y:
                        miss |= 1 << i
                blocks.append((e, miss))
        conditions = []
        for q in self.var_names:
            for c in sorted({inp[q] for inp in inputs if inp.get(q) is not None}):
                i1 = [i for i, inp in enumerate(inputs) if inp.get(q) == c]
                if 0 < len(i1) < n:
                    conditions.append((q, c, i1))
        if not conditions:
            return []
        log_nc = math.log(len(conditions))
        full = (1 << n) - 1
        out = []
        for q, c, i1 in conditions:
            mask1 = sum(1 << i for i in i1)
            m1 = self._branch_blocks(blocks, mask1, len(i1))
            if not m1:
                continue
            m0 = self._branch_blocks(blocks, full ^ mask1, n - len(i1))
            for e1, b1 in m1:
                for e0, b0 in m0:
                    if e1 == e0:
                        continue
                    lp = (-log_nc - math.log(4) + self._base_prior(e1, counts) + self._base_prior(e0, counts))
                    out.append((if_eq(q, c, e1, e0), b1 + b0, lp))
        return out

    def _branch_blocks(self, blocks: list, mask: int, size: int, k: int = 3) -> list:
        exact = []
        for e, miss in blocks:
            if not miss & mask:
                exact.append((e, 0))
                if len(exact) == k:
                    return exact
        if exact:
            return exact
        tol = max(self._tol(size, search=True), 1 if size >= 3 else 0)
        if not tol:
            return []
        scored = ((bad, i) for i, (_, miss) in enumerate(blocks) if (bad := (miss & mask).bit_count()) <= tol)
        return [(blocks[i][0], bad) for bad, i in heapq.nsmallest(k, scored)]

    # -------------------------------------------------------------- persistence
    def to_state(self) -> dict:
        return {
            "key": self.key, "var_names": self.var_names, "data": self.data, "episodes": self.episodes,
            "provenance": self.provenance, "hyps": [h.to_json() for h in self.hyps], "backoff": self.backoff,
            "last_induction_n": self.last_induction_n, "n_inductions": self.n_inductions,
            "expressions_evaluated": self.expressions_evaluated, "induction_seconds": self.induction_seconds,
            "status_history": self.status_history, "refutations": self.refutations, "status": self._status,
            "library_searches": self.library_searches, "full_searches": self.full_searches,
            "deep_searches": self.deep_searches, "next_deep_n": self.next_deep_n,
            "invention_searches": self.invention_searches, "invention_hits": self.invention_hits,
            "linear_searches": self.linear_searches, "linear_hits": self.linear_hits,
            "noise_delays": self.noise_delays,
        }

    @classmethod
    def from_state(cls, state: dict, cfg: InductionConfig,
                   shape_prior: Optional[Callable[[str], float]] = None,
                   probe_fn: Optional[Callable[["RuleLearner", Expr], list]] = None,
                   shape_library: Optional[Callable[[], list]] = None,
                   learned_library: Optional[Callable[[], tuple]] = None) -> "RuleLearner":
        lr = cls(state["key"], state["var_names"], cfg, shape_prior, probe_fn, shape_library, learned_library)
        lr.invention_searches = state.get("invention_searches", 0)
        lr.invention_hits = state.get("invention_hits", 0)
        lr.library_searches = state.get("library_searches", 0)
        lr.full_searches = state.get("full_searches", 0)
        lr.deep_searches = state.get("deep_searches", 0)
        lr.next_deep_n = state.get("next_deep_n", 0)
        lr.linear_searches = state.get("linear_searches", 0)
        lr.linear_hits = state.get("linear_hits", 0)
        lr.noise_delays = state.get("noise_delays", 0)
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
        lr._best_memo = None
        return lr
