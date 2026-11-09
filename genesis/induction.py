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

import functools
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

# The search keeps every program up to this size, and one program per behaviour on the observations beyond it.
KEEP_ALL_UPTO = 3

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
    # ... or, when a rule that does not hold would agree with the outcomes at most at this rate (a numeric prediction
    # is right by chance about once in outcome_range; 0 disables the test), an anytime-valid test of the forward
    # record (RuleLearner._beats_chance) shows, at level validation_sprt_alpha, that the rule is right more often
    # than chance: being right more often than wrong is not required, so a heavily corrupted stream can be learned.
    validation_chance: float = 0.0
    # Prior weight of "this world corrupts observations at an unknown rate" (uniform prior, integrated out) against
    # "nearly clean" (rate `noise`), in every rule's likelihood (RuleLearner._scores). A rule right on half of the
    # observations then still gains about log(outcome_range / 2) nats per observation over the null, where the
    # fixed rate alone makes each of its errors cost log(outcome_range / noise); at low error rates the posterior is
    # unchanged (within log(1 - weight)). 0 disables it.
    noise_unknown_weight: float = 0.0
    # Severity: every rule within max_size, exact or not and whatever its size, is challenged before its validation
    # (RuleLearner._challenge), and is validated only once the evidence, not its prior alone, separates it from each
    # rival that fits the observations at least as well but predicts another future on its probes
    # (RuleLearner._untested_rival): a simple rule is no longer validated on its simplicity alone, before the
    # input that decides between it and a longer rule. Off: only rules that already err are challenged.
    severe_validation: bool = False
    # Under that "unknown rate" component (and its share of the null), corrupted values follow a law learned from
    # the observations rather than uniform(outcome_range): a Bayes mixture of uniform(outcome_range) and a uniform
    # law on an unknown interval (RuleLearner._value_law). A rule's hit is evidence only as far as the value was
    # unlikely by chance: on a stream of digits, a coincidental hit is a 1-in-10 event, not 1-in-100.
    learned_chance: bool = False
    # A rule survives while it errs on at most this fraction of the observations (by default it is right at least as
    # often as wrong, far beyond the chance agreement of noise): bursts of corruption well above the world's nominal
    # rate then pause it instead of refuting it. Passive streams use 0.75 (genesis.agent.default_configs): right on
    # a quarter of the observations is still 25 times chance, and a stream with half of its observations corrupted
    # keeps its rule through the bursts.
    max_mismatch_frac: float = 0.5
    # New rules are proposed only with at most this many mismatches (capped by max_mismatch_frac). Passive streams
    # use 0.75 (genesis.agent.default_configs: a stream with half of its observations corrupted is still learned,
    # although its first ones can be mostly corrupted); where the learner chooses its experiments, a wider search
    # floods the hypotheses with half-right conditionals that divert the experiments which would settle them.
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
    # When the recognition model learned during sleep, updated with the evidence of the surviving hypotheses
    # (RuleLearner._likely_noise), judges the series rule-less with at least this probability (after noise_min_n
    # observations), a due deep search (deep, linear and invention stages) is postponed by
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
    # Residual counts of the mismatched observations, extended as data arrives (see _structured_errors).
    residuals: Optional[tuple] = field(default=None, compare=False, repr=False)
    # Bitset of the observations the rule predicts exactly (bit i: observation i), extended as data arrives.
    hits: Optional[tuple] = field(default=None, compare=False, repr=False)
    # Markov-model statistics of that hit/miss record (see RuleLearner._pattern_log_ratio).
    pattern: Optional[tuple] = field(default=None, compare=False, repr=False)

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


def _idle_modulus(e: Expr, data: list) -> bool:
    """Whether a ``mod`` in ``e`` never changed its argument on the observations (a % m == a on each). Such a
    rule is another rule plus a clause the data never touched, e.g. x[t-1] + x[t-2] % 27 on a Fibonacci stream
    whose x[t-2] stayed below 27: an ad hoc exception, not a rival explanation of the data. A rule can always
    be given one beyond the observations, so severity does not wait for them; a modulus once exercised stays so."""
    if e.op == "mod":
        a, m = e.args
        idle = True
        for d, _ in data:
            v, mv = a.evaluate(d, True), m.evaluate(d, True)
            if v is not None and mv is not None and mv > 0 and v % mv != v:
                idle = False
                break
        if idle:
            return True
    return any(_idle_modulus(x, data) for x in e.args if isinstance(x, Expr))


def _unseparated(top_bits: int, rival_bits: int, born_n: int) -> bool:
    """Whether the evidence has not separated a rule from a rival (bit i of each mask: observation i predicted
    exactly). Observations the rule was fitted to (before born_n, when it joined the pool) are not tests it could
    have failed, so they cannot separate it alone: a rule found by search can fit a corrupted value by
    coincidence, as -21 + x[t-1] % 38 fitted the one corrupted term of x[t-2] - 4 and every later term as well.
    Unseparated: the rival errs no more often than the rule, or errs once more and the observations made since
    the rule was proposed do not favour the rule. Two more errors are not excused: a second coincidence."""
    top_only = (top_bits & ~rival_bits).bit_count()
    rival_only = (rival_bits & ~top_bits).bit_count()
    if top_only <= rival_only:
        return True
    if top_only > rival_only + 1:
        return False
    novel = ~((1 << born_n) - 1)
    return (top_bits & ~rival_bits & novel).bit_count() <= (rival_bits & ~top_bits & novel).bit_count()


def _logit(p: float) -> float:
    return math.inf if p >= 1 else -math.inf if p <= 0 else math.log(p / (1 - p))


def _logsumexp(xs: list) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _logaddexp(a: float, b: float) -> float:
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _interval_log_prob(y: int, lo: int, hi: int, t: int) -> float:
    """Predictive log probability of ``y`` after ``t >= 2`` integers spanning [lo, hi], when they are drawn
    uniformly from an unknown interval (scale-invariant prior on its width): (t - 1) / (t + 1) of the mass is
    spread over [lo, hi] and the rest decays with the distance d outside it as s^(t-1) / (s + d)^t, where s is the
    span. Summed over all integers it is at most 1."""
    s = hi - lo + 1
    base = math.log(t - 1) - math.log(t + 1)
    if lo <= y <= hi:
        return base - math.log(s)
    d = lo - y if y < lo else y - hi
    return base + (t - 1) * math.log(s) - t * math.log(s + d)


def _log_beta(a: int, b: int) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# Orders of the Markov models that look for a pattern in when a rule is right (RuleLearner._pattern_log_ratio).
PATTERN_ORDERS = 6


@functools.lru_cache(maxsize=4096)
def _chance_log_ratio(ok: int, bad: int, rho: float) -> float:
    """Log likelihood ratio of ``ok`` successes and ``bad`` failures under "right at an unknown rate above rho"
    (uniform prior on (rho, 1)) against "right at rate rho". Under any rate <= rho it is a nonnegative
    supermartingale, so by Ville's inequality it ever reaches log(1 / alpha) with probability at most alpha, however
    long the record grows (an anytime-valid test, Robbins' method of mixtures)."""
    m = ok + bad
    # Mass of Beta(ok + 1, bad + 1) above rho = P(Binomial(m + 1, rho) <= ok).
    l_rho, l_not = math.log(rho), math.log1p(-rho)
    tail = _logsumexp([math.lgamma(m + 2) - math.lgamma(j + 1) - math.lgamma(m + 2 - j) + j * l_rho
                       + (m + 1 - j) * l_not for j in range(ok + 1)])
    return _log_beta(ok + 1, bad + 1) + min(tail, 0.0) - l_not - ok * l_rho - bad * l_not


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
        self.rival_searches = 0
        self.rivals_found = 0
        self.deep_rival_searches = 0
        # Share of expressions_evaluated spent verifying rules (_challenge) rather than discovering them.
        self.rival_expressions_evaluated = 0
        self._challenged: set = set()
        # Rule -> its rivals that only a decisive observation can dismiss (severe_validation).
        self._decisive: dict = {}
        self._untested_memo: Optional[tuple] = None
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
        self._law_memo: Optional[tuple] = None
        self._inc_memo: Optional[tuple] = None
        self._best_memo: Optional[tuple] = None
        self._pred_memo: Optional[tuple] = None

    # ------------------------------------------------------------------ beliefs
    def posterior(self) -> tuple:
        """``(((hypothesis, probability), ...) most probable first, p_null)``, memoised: it is asked for many
        times between two observations. The memo holds the hypotheses themselves, so their ids stay unique."""
        cfg = self.cfg
        n = len(self.data)
        # Hypothesis counts change only in observe(), which then replaces the pool list, as _merge() does: the
        # pool list itself (kept alive by the memo, so its id is not reused) identifies the evidence.
        key = (n, cfg.noise, cfg.outcome_range, cfg.null_prior, cfg.noise_unknown_weight, cfg.learned_chance,
               len(self.hyps))
        memo = self._post_memo
        if memo is not None and memo[1] is self.hyps and memo[0] == key:
            return memo[2]
        out = self._posterior(n)
        self._post_memo = (key, self.hyps, out)
        return out

    def _scores(self, hyps: list) -> list:
        """Unnormalised log posterior of each hypothesis. Its likelihood: each observation is right with probability
        1 - e, else one of outcome_range values, where the error rate e is `noise` or, with prior weight
        noise_unknown_weight, unknown (uniform, integrated out). The unknown rate stands for corruption, which is
        unpredictable: it is not offered to a pooled rule whose errors a rival predicts (_incomplete_ids)."""
        cfg = self.cfg
        l_ok = math.log(1 - cfg.noise)
        l_bad = math.log(cfg.noise / cfg.outcome_range)
        base = math.log(1 - cfg.null_prior)
        w = cfg.noise_unknown_weight
        if w <= 0:
            return [base + h.log_prior + h.n_ok * l_ok + h.n_bad * l_bad for h in hyps]
        l_clean, l_unknown = math.log1p(-w), math.log(w)
        # Log probability of a corrupted value under the unknown rate: the learned law's, or uniform.
        l_value = self._value_law() if cfg.learned_chance else -math.log(cfg.outcome_range)
        incomplete = self._incomplete_ids() if hyps is self.hyps else ()
        out = []
        for h in hyps:
            clean = l_clean + h.n_ok * l_ok + h.n_bad * l_bad
            if id(h) in incomplete:
                out.append(base + h.log_prior + clean)
                continue
            unknown = l_unknown + _log_beta(h.n_bad + 1, h.n_ok + 1) + h.n_bad * l_value
            hi, lo = (clean, unknown) if clean >= unknown else (unknown, clean)
            out.append(base + h.log_prior + hi + math.log1p(math.exp(lo - hi)))
        return out

    def _null_ll(self) -> float:
        """Log likelihood of the observations under the null model (no rule): uniform outcomes, or, with
        learned_chance, uniform with weight 1 - noise_unknown_weight and the learned value law otherwise."""
        cfg = self.cfg
        uniform, w = len(self.data) * math.log(1 / cfg.outcome_range), cfg.noise_unknown_weight
        if not cfg.learned_chance or w <= 0:
            return uniform
        return _logaddexp(math.log1p(-w) + uniform, math.log(w) + len(self.data) * self._value_law())

    def _value_law(self) -> float:
        """Log probability, under the value law learned so far, of each observed value: the chance that a rule
        is right by coincidence. The law is the Bayes mixture, with prior 1/2 each, of uniform(outcome_range) and
        the uniform law on an unknown interval, weighted by how well each predicted the outcomes one after the
        other (_interval_log_prob; uniform until two integers are seen). Each value is then given its current
        probability, about 1/10 on a stream of digits and 1/outcome_range on an unbounded stream: an
        empirical-Bayes estimate, which credits the null with the range it has seen. Chance is at most 1/2: a law
        concentrated on one value is a rule (a constant), not chance. Updated incrementally."""
        memo = self._law_memo
        if memo is None or memo[0] is not self.data or memo[1] > len(self.data):
            memo = (self.data, 0, 0, 0, 0, 0.0)
        data, done, t, lo, hi, interval = memo
        l_uniform = -math.log(self.cfg.outcome_range)
        for _, y in self.data[done:]:
            if not isinstance(y, int) or isinstance(y, bool):
                interval += l_uniform
                continue
            interval += _interval_log_prob(y, lo, hi, t) if t >= 2 else l_uniform
            lo, hi, t = (y, y, 1) if t == 0 else (min(lo, y), max(hi, y), t + 1)
        self._law_memo = (data, len(self.data), t, lo, hi, interval)
        if t < 2:
            return l_uniform
        uniform = len(self.data) * l_uniform
        inside = math.log(t - 1) - math.log(t + 1) - math.log(max(hi - lo + 1, 2))
        return _logaddexp(uniform + l_uniform, interval + inside) - _logaddexp(uniform, interval)

    def _hit_bits(self, h: Hypothesis) -> int:
        """Bit i is set when ``h`` predicts observation i exactly (memoised, extended as data arrives)."""
        memo = h.hits
        if memo is None or memo[0] is not self.data or memo[1] > len(self.data):
            memo = (self.data, 0, 0)
        _, done, bits = memo
        for i in range(done, len(self.data)):
            d, y = self.data[i]
            if h.expr.evaluate(d, True) == y:
                bits |= 1 << i
        h.hits = (self.data, len(self.data), bits)
        return bits

    def _pattern_log_ratio(self, h: Hypothesis) -> float:
        """Log likelihood ratio of the rule's hit/miss record (in observation order) under a Bayes mixture of
        Markov models of orders 1 to PATTERN_ORDERS (Krichevsky-Trofimov estimates per context of previous bits)
        against its best fit as independent draws. Corruption strikes at independent times, so a corrupted rule's
        record is such draws: under any rate, the mixture over that rate is a nonnegative martingale and this ratio
        is below it, so it ever reaches log(1 / alpha) with probability at most alpha (Ville). A rule right on one
        phase of a cycle (every fourth term) reaches it within a few cycles. Updated incrementally."""
        bits, n = self._hit_bits(h), len(self.data)
        memo = h.pattern
        if memo is None or memo[0] is not self.data or memo[1] > n:
            memo = (self.data, 0, [{} for _ in range(PATTERN_ORDERS)], [0.0] * PATTERN_ORDERS)
        _, done, counts, logq = memo
        for i in range(done, n):
            b = (bits >> i) & 1
            for j in range(PATTERN_ORDERS):
                m = min(j + 1, i)
                ctx = ((bits >> (i - m)) & ((1 << m) - 1), m)
                c = counts[j].get(ctx)
                if c is None:
                    c = counts[j][ctx] = [0, 0]
                logq[j] += math.log((c[b] + 0.5) / (c[0] + c[1] + 1))
                c[b] += 1
        h.pattern = (self.data, n, counts, logq)
        k = bits.bit_count()
        best = (k * math.log(k / n) if k else 0.0) + ((n - k) * math.log((n - k) / n) if k < n else 0.0)
        return _logsumexp(logq) - math.log(PATTERN_ORDERS) - best

    def _incomplete_ids(self) -> frozenset:
        """Ids of the pooled rules whose errors are structure they miss, not corruption, by either of two
        anytime-valid tests at level validation_sprt_alpha / 2 each:

        * when it is right is patterned (_pattern_log_ratio), e.g. on one phase of a cycle; or
        * a rival predicts its errors far better than chance (whatever the rule's error rate: an incomplete rule
          right on most inputs but wrong on a region of them is no more corrupted than one wrong on most). Under
          "this rule, corrupted", another rule hits a corrupted value only by coincidence, at the learned chance
          rate (rounded up to 1 / k); the chance test (_chance_log_ratio) on the rival's hits among the rule's
          errors, with the level shared among the rivals, refutes that. Only the errors observed after the rival
          joined the pool count: it was chosen to fit the earlier ones.

        Only with noise_unknown_weight > 0 (the unknown rate is what models corruption). Both tests depend only on
        the rule's hit record (a rival with the same record hits none of its errors), so pooled rules that agree on
        every observation share one test. Memoised for the current pool and data."""
        memo = self._inc_memo
        n = len(self.data)
        if memo is not None and memo[0] is self.hyps and memo[1] == n:
            return memo[2]
        cfg, hyps, out = self.cfg, self.hyps, set()
        alpha = cfg.validation_sprt_alpha / 2
        if cfg.noise_unknown_weight > 0 and alpha > 0:
            threshold = math.log(1 / alpha)
            patterned: dict = {}
            for h in hyps:
                if h.n_bad >= 2:
                    bits = self._hit_bits(h)
                    flag = patterned.get(bits)
                    if flag is None:
                        flag = patterned[bits] = self._pattern_log_ratio(h) >= threshold
                    if flag:
                        out.add(id(h))
            suspects = [h for h in hyps if h.n_bad >= 2 and id(h) not in out]
            if suspects and len(hyps) > 1:
                chance = math.exp(self._value_law()) if cfg.learned_chance else 1 / cfg.outcome_range
                rho = 1 / max(int(1 / chance), 2)
                threshold = math.log((len(hyps) - 1) / alpha)
                # Rivals' hits after they joined the pool, grouped by when they joined.
                rivals: dict = {}
                for g in hyps:
                    rivals.setdefault(g.born_n, set()).add(self._hit_bits(g) >> g.born_n)
                full = (1 << n) - 1
                refuted: dict = {}
                for h in suspects:
                    bits = self._hit_bits(h)
                    flag = refuted.get(bits)
                    if flag is None:
                        flag = refuted[bits] = self._rival_hits_errors(full & ~bits, rivals, rho, threshold)
                    if flag:
                        out.add(id(h))
        ids = frozenset(out)
        self._inc_memo = (hyps, n, ids)
        return ids

    @staticmethod
    def _rival_hits_errors(errors: int, rivals: dict, rho: float, threshold: float) -> bool:
        """Whether some rival's hits among the ``errors`` observed after it joined the pool pass the chance test."""
        for born, records in rivals.items():
            later = errors >> born
            m = None
            for hits in records:
                k = (hits & later).bit_count()
                if k >= 2:
                    if m is None:
                        m = later.bit_count()
                    if _chance_log_ratio(k, m - k, rho) >= threshold:
                        return True
        return False

    def _posterior(self, n: int) -> tuple:
        cfg = self.cfg
        null = math.log(cfg.null_prior) + self._null_ll()
        scores = self._scores(self.hyps)
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
                    and (top.fwd_bad <= int(self.cfg.validation_max_error * fwd) or self._sprt_accepts(top)
                         or self._beats_chance(top))
                    and not self._structured_errors(top) and not self._untested_rival(top)):
                return "validated"
            if mass >= 0.5:
                return "candidate"
        return "unknown"

    def _promising(self, h: Optional[Hypothesis]) -> bool:
        """A surviving rule that predicted the future far better than chance (at least two forward successes, more
        successes than failures) keeps the question open, even while the null still dominates the posterior: a
        long rule pays its Occam prior back only over several more observations, and judging the stream
        unpredictable now would withdraw the attention that would decide it. Noise almost never yields one (each
        forward success is a 1 / outcome_range coincidence). With validation_chance, a record that favours "right
        more often than chance" over chance itself also counts (a heavily corrupted rule errs more often than not),
        unless the rule is incomplete (_incomplete_ids): its errors are then structure it misses, not corruption."""
        if h is None or h.fwd_ok < 2:
            return False
        if h.fwd_ok > h.fwd_bad:
            return True
        rho = self.cfg.validation_chance
        return (0 < rho < 1 and _chance_log_ratio(h.fwd_ok, h.fwd_bad, rho) > 0
                and id(h) not in self._incomplete_ids())

    def _sprt_accepts(self, h: Hypothesis) -> bool:
        """Wald's log-likelihood ratio of the forward record under "errs at validation_max_error" against "errs at
        sprt_null_error" reaches log(1 / validation_sprt_alpha). Its drift is positive only below an error rate of
        about 37% (with the defaults), and the record grows with every forward prediction."""
        alpha, p1, p0 = self.cfg.validation_sprt_alpha, self.cfg.validation_max_error, self.cfg.sprt_null_error
        if alpha <= 0 or not 0 < p1 < p0 < 1:
            return False
        llr = h.fwd_ok * math.log((1 - p1) / (1 - p0)) + h.fwd_bad * math.log(p1 / p0)
        return llr >= math.log(1 / alpha)

    def _beats_chance(self, h: Hypothesis) -> bool:
        """The forward record shows, at level validation_sprt_alpha and whatever its length, that the rule is right
        more often than validation_chance (see _chance_log_ratio). A rule right on half of its forward predictions
        passes after about a dozen of them; one right by chance (on noise, about once in 61 here) almost never does."""
        alpha, rho = self.cfg.validation_sprt_alpha, self.cfg.validation_chance
        if alpha <= 0 or not 0 < rho < 1:
            return False
        return _chance_log_ratio(h.fwd_ok, h.fwd_bad, rho) >= math.log(1 / alpha)

    def _structured_errors(self, h: Hypothesis) -> bool:
        """Noise is unstructured: when most of a rule's errors share one residual (a missed wrap-around, an
        off-by-one branch), or a rival predicts them (_incomplete_ids), the rule is incomplete and must not be
        validated, whatever its error rate."""
        if h.n_bad < 2:
            return False
        if id(h) in self._incomplete_ids():
            return True
        memo = h.residuals
        if memo is None or memo[0] is not self.data or memo[1] > len(self.data):
            memo = (self.data, 0, Counter())
        _, done, residuals = memo
        for d, y in self.data[done:]:
            v = h.expr.evaluate(d, True)
            if v != y:
                residuals[None if v is None else y - v] += 1
        h.residuals = (self.data, len(self.data), residuals)
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

        n_before = len(self.data)
        for h in self.hyps:
            ok = h.expr.evaluate(inputs, True) == y
            if ok:
                h.n_ok += 1
                h.fwd_ok += 1
            else:
                h.n_bad += 1
                h.fwd_bad += 1
            memo = h.hits
            if memo is not None and memo[0] is self.data and memo[1] == n_before:
                h.hits = (self.data, n_before + 1, memo[2] | (1 << n_before) if ok else memo[2])
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
        if self._status == "validated" and self._challenge(self.best()[0]):
            self._status = self.status()
        if self._status != status_before:
            self.status_history.append({"n": n, "status": self._status, "episode": episode_id})
        return LearnResult(pred, correct, surprise, reinduced, status_before, self._status, refuted)

    def _challenge(self, top: Hypothesis) -> bool:
        """Look for rivals of a rule about to be validated: rules within max_size that explain the observations at
        least as well but predict a different future. The search keeps one program per behaviour on the
        observations (the first enumerated), so such a rival can be missing from the pool although it is exactly as
        plausible, e.g. (x - 1) % 10 next to x % 10 - 1 when corruption kept a counter off its wrap-around and the
        decisive input never occurred. Once per rule, the search runs on the observations plus the rule's own probes
        (the inputs it would be used on), so rivals that disagree there stay distinct; they join the pool and the
        posterior weighs them. Returns whether any was found. The probes are labelled with a value no program
        reaches (the kernels are bounded by VALUE_LIMIT), so every candidate misses all of them and the tolerance
        admits exactly the programs that fit the observations as well as the rule.

        Under severe_validation every rule within max_size is challenged, exact or not and whatever its size: a
        rival larger than the rule fits as well but weighs far less in the prior (x[t-3] + 1 against
        (x[t-1] - 2) % 7), so the posterior alone would validate the shorter one before the decisive input. The
        rivals that fit as well and disagree on the probes are kept in _decisive and hold the validation back
        (_untested_rival). A rule that already errs also faces a deeper search, up to deepen_size nodes, for rules
        that explain some of its errors: errors that a longer rule predicts are not corruption. Off, only rules
        larger than KEEP_ALL_UPTO that already err are challenged (an equally simple rival would be in the pool,
        and an exact rule errs on the first decisive input)."""
        cfg = self.cfg
        key = str(top.expr)
        if key in self._challenged or top.expr.size > cfg.max_size:
            return False
        if not cfg.severe_validation and (top.n_bad == 0 or top.expr.size <= KEEP_ALL_UPTO):
            return False
        self._challenged.add(key)
        rows = [(d, y) for d, y in self.data if _in_range(d, y)]
        probes = self.probe_fn(self, top.expr) if self.probe_fn is not None else self._probes
        labelled = []
        for p in probes:
            v = top.expr.evaluate(p, True)
            if v is not None and _in_range(p, v):
                labelled.append((p, v))
        if len(rows) < 5 or not labelled:
            return False
        self.rival_searches += 1
        evaluated_before = self.expressions_evaluated
        top_bad = sum(top.expr.evaluate(d, True) != y for d, y in self.data)
        rows_bad = sum(top.expr.evaluate(d, True) != y for d, y in rows)
        inputs = [d for d, _ in rows]
        ys = [y for _, y in rows]
        consts = self._propose_constants(inputs, tuple(ys))
        # Severe: also rivals with one more mismatch, kept only if it falls on the observations the rule was
        # fitted to (_unseparated).
        extra = 1 if cfg.severe_validation else 0
        sr = search_programs(inputs + [p for p, _ in labelled], ys + [VALUE_LIMIT + 1] * len(labelled),
                             self.var_names, sorted(consts),
                             rows_bad + extra + len(labelled), cfg.max_size,
                             cfg.max_candidates, cfg.max_seconds, 0, keep_all_upto=KEEP_ALL_UPTO)
        self.expressions_evaluated += sr.evaluated
        rivals = []
        top_bits = self._hit_bits(top) if extra else 0
        for e, _ in sr.matches:
            if _has_trivial(e) or all(e.evaluate(p, True) == v for p, v in labelled):
                continue
            exact = sum(e.evaluate(d, True) != y for d, y in self.data)
            if exact > top_bad:
                if exact > top_bad + extra:
                    continue
                bits = 0
                for i, (d, y) in enumerate(self.data):
                    if e.evaluate(d, True) == y:
                        bits |= 1 << i
                if not _unseparated(top_bits, bits, top.born_n):
                    continue
            rivals.append((e, exact, self._base_prior(e, sr.counts)))
        if cfg.severe_validation and rows_bad and cfg.deepen_size > cfg.max_size:
            # Errors that a deeper rule explains are not corruption: rules of up to deepen_size nodes that explain
            # more of the observations than this one (the deepening only runs when nothing smaller does).
            found, _, _ = self._search(inputs, tuple(ys), rows_bad - 1, consts, {}, rows_bad - 1, cfg.deepen_size)
            self.deep_rival_searches += 1
            seen = {str(e) for e, _, _ in rivals}
            for e, _, lp in found:
                if str(e) not in seen and not _has_trivial(e):
                    exact = sum(e.evaluate(d, True) != y for d, y in self.data)
                    if exact < top_bad:
                        seen.add(str(e))
                        rivals.append((e, exact, lp))
        self.rival_expressions_evaluated += self.expressions_evaluated - evaluated_before
        if rivals:
            self.rivals_found += len(rivals)
            self._merge(rivals, len(self.data))
        if cfg.severe_validation:
            # Every pooled rule as good on the evidence and disagreeing on the probes, whether just found or not.
            pending = [str(h.expr) for h in self.hyps if h is not top and self._unseparated(top, h)
                       and any(h.expr.evaluate(p, True) != v for p, v in labelled)
                       and not _idle_modulus(h.expr, self.data)]
            if pending:
                self._decisive[key] = pending
                self._untested_memo = None
                return True
        return bool(rivals)

    def _unseparated(self, top: Hypothesis, rival: Hypothesis) -> bool:
        return _unseparated(self._hit_bits(top), self._hit_bits(rival), top.born_n)

    def _untested_rival(self, top: Hypothesis) -> bool:
        """Severity (severe_validation): a rival found when the rule was challenged is not separated from it yet
        (_unseparated) and still predicts a different value on one of the rule's probes, so only the prior, or
        the observations the rule was fitted to, prefer the rule. The prior ranks rules; it does not test them:
        x[t-3] + 1 fits five terms of a counter modulo 7 as well as (x[t-1] - 2) % 7, and is 2 nodes shorter, but
        only the term after the wrap-around can tell them apart. The rule waits for it: either the rival errs
        there (the rule passed a test it could have failed), or the rule does and is not validated. A rival that
        left the pool, that the rule's predictions have separated from it, or that agrees with the rule on all
        of its current probes no longer stands in the way."""
        names = self._decisive.get(str(top.expr))
        if not names:
            return False
        memo_key = (len(self.data), id(top), id(self.hyps))
        if self._untested_memo is not None and self._untested_memo[0] == memo_key:
            return self._untested_memo[1]
        pool = {str(h.expr): h for h in self.hyps}
        rivals = [pool[k] for k in names if k in pool and self._unseparated(top, pool[k])]
        result = False
        if rivals:
            probes = self.probe_fn(self, top.expr) if self.probe_fn is not None else self._probes
            for p in probes:
                v = top.expr.evaluate(p, True)
                if any(r.expr.evaluate(p, True) != v for r in rivals):
                    result = True
                    break
        self._untested_memo = (memo_key, result)
        return result

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
        # Back off when the previous search failed: nothing survived, or its best rule kept failing. Where rules
        # survive while wrong more often than right (max_mismatch_frac > 1/2), a best rule that is not right more
        # often than wrong has not settled the stream either: a search after each of its errors would rerun on
        # nearly the same data (a reachable rule that is right on one phase of a longer cycle errs on most terms).
        prev = self._prev_top
        failed = prev is None or prev.fwd_ok < 2 or (cfg.max_mismatch_frac > 0.5 and prev.fwd_ok <= prev.fwd_bad)
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
        found = self._rescore(self._library_search(inputs, ys, tol, consts, None if check else cfg.pool_size),
                              check, partial, tol)
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

    def _library_search(self, inputs: list, ys: tuple, tol: int, consts: set, keep: Optional[int] = None) -> list:
        """Instantiate previously successful shapes; return those that explain the data. With ``keep``, at most
        that many new hypotheses per shape (the fewest mismatches first, then in enumeration order): instances of
        one shape share their prior, so the others rank below them and cannot enter a pool of ``keep``; on small
        series a remembered shape has hundreds of instances that agree on every observation. Instances that are
        already hypotheses do not count (the merge skips them)."""
        if self.shape_library is None:
            return []
        existing = {str(h.expr) for h in self.hyps} if keep is not None else set()
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
                order = range(len(hits)) if keep is None else sorted(range(len(hits)), key=lambda i: hits[i][3])
                taken, n_new = [], 0
                for i in order:
                    vs, cs, r, bad = hits[i]
                    e = fill_shape(body, vs, cs)
                    if fit_bias:
                        e = app("add", e, const(r))
                    if _has_trivial(e) or e.size > size_bound:
                        continue
                    k = str(e)
                    if k not in seen:
                        seen.add(k)
                        taken.append((i, (e, bad, self._approx_prior(e, n_leaves))))
                        if keep is not None and k not in existing:
                            n_new += 1
                            if n_new >= keep:
                                break
                found += [item for _, item in sorted(taken, key=lambda t: t[0])]
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
                             cfg.max_candidates, cfg.max_seconds, cfg.max_size - 2, keep_all_upto=KEEP_ALL_UPTO,
                             deepen_bad=deepen_bad, deep_size=cfg.deepen_size, window=cfg.deepen_window,
                             n_windows=cfg.deepen_windows, deep_max_size=deep_max_size or cfg.deepen_max_size)
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
        """Whether a due deep search should wait. The recognition model's odds that the series has no rule (learned
        during sleep from coarse features) are a prior; the learner's own evidence updates them: the likelihood of
        the observations under its surviving hypotheses against the null model (uniform outcomes), as in
        posterior(). A series that a hypothesis already mostly explains is not postponed however noise-like its
        features look (a lag-2 rule under 20% corruption alternates like draws). Only evidence for a rule counts:
        the pool holds what a bounded search found, and that search's failure is what the deeper one is for."""
        cfg = self.cfg
        rec = self.learned_library()[1] if self.learned_library is not None else None
        if rec is None or len(self.data) < cfg.noise_min_n:
            return False
        odds = rec.noise_log_odds(task_features([y for _, y in self.data]))
        if odds is None:
            return False
        if self.hyps:
            null = self._null_ll()
            rules = _logsumexp(self._scores(self.hyps)) - math.log(1 - cfg.null_prior)
            odds += min(0.0, null - rules)
        return odds >= _logit(cfg.noise_recognition)

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
        # The pool_size most probable, by log score (ties: pool order). Ranking the scores rather than their
        # normalised probabilities (posterior()) keeps the order of hypotheses far below the best, whose
        # probabilities all underflow to 0, and makes it independent of the candidates that cannot enter.
        pool = self.hyps + fresh
        scores = self._scores(pool)
        self.hyps = [pool[i] for i in heapq.nsmallest(cfg.pool_size, range(len(pool)), key=lambda i: (-scores[i], i))]
        before = self._probes
        self._make_probes()
        # Signatures of the hypotheses already pooled are kept while the probes are unchanged.
        fresh_ids = {id(h) for h in fresh}
        todo = self.hyps if self._probes != before else [h for h in self.hyps if id(h) in fresh_ids]
        for h in todo:
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
            "noise_delays": self.noise_delays, "rival_searches": self.rival_searches,
            "rivals_found": self.rivals_found, "challenged": sorted(self._challenged),
            "deep_rival_searches": self.deep_rival_searches,
            "rival_expressions_evaluated": self.rival_expressions_evaluated,
            "decisive": self._decisive,
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
        lr.rival_searches = state.get("rival_searches", 0)
        lr.rivals_found = state.get("rivals_found", 0)
        lr.deep_rival_searches = state.get("deep_rival_searches", 0)
        lr.rival_expressions_evaluated = state.get("rival_expressions_evaluated", 0)
        lr._challenged = set(state.get("challenged", ()))
        lr._decisive = {k: list(v) for k, v in state.get("decisive", {}).items()}
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
