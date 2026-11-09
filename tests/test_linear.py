import random
from fractions import Fraction

import pytest

from genesis.adapters import SEQ_VARS, sequence_rollout_probes
from genesis.expressions import var
from genesis.induction import InductionConfig, RuleLearner
from genesis.linear import build, dense_windows, linear_rules, solve_exact


def fraction_solve(m):
    k = len(m)
    m = [[Fraction(v) for v in row] for row in m]
    for p in range(k):
        piv = next((i for i in range(p, k) if m[i][p]), None)
        if piv is None:
            return None
        m[p], m[piv] = m[piv], m[p]
        for i in range(k):
            if i != p and m[i][p]:
                f = m[i][p] / m[p][p]
                m[i] = [a - f * b for a, b in zip(m[i], m[p])]
    return [m[i][k] / m[i][i] for i in range(k)]


def test_solve_exact_matches_rational_elimination():
    rng = random.Random(0)
    for _ in range(2000):
        k = rng.randint(1, 5)
        m = [[rng.randint(-6, 6) for _ in range(k + 1)] for _ in range(k)]
        if rng.random() < 0.2:
            m[-1] = list(m[0])  # singular
        ref = fraction_solve(m)
        got = solve_exact(m)
        if ref is None:
            assert got is None
        else:
            num, d = got
            assert [Fraction(x, d) for x in num] == ref


def test_dense_windows_cover_short_histories_at_stride_one():
    assert dense_windows(3, 4, 8) == [[0, 1, 2]]
    assert [w[0] for w in dense_windows(9, 4, 8)] == [5, 4, 3, 2, 1, 0]
    assert [w[0] for w in dense_windows(40, 4, 8)] == [36, 34, 32, 30, 28, 26, 24, 22]
    assert all(len(w) == 5 for w in dense_windows(12, 5, 8))


def test_build_writes_positive_terms_then_subtractions_then_constant():
    assert str(build([var("x[t-1]"), var("x[t-2]"), var("t")], [3, -2, 1], -4)) == \
        "((((x[t-1] * 3) + t) - (x[t-2] * 2)) + -4)"
    assert build([var("t")], [1], 0) == var("t")
    assert build([var("t")], [2], 7).evaluate({"t": 5}) == 17


def rows_of(terms):
    inputs = [{"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}
              for t in range(3, len(terms))]
    return inputs, tuple(terms[3:])


def series(step, start, n=24):
    terms = list(start)
    while len(terms) < n:
        terms.append(step(terms, len(terms)))
    return terms


@pytest.mark.parametrize("name,step,start", [
    ("third-order-drift", lambda h, t: h[-1] - h[-2] + h[-3] + 3 * t - 2, [1, -4, 2]),
    ("weighted-drift", lambda h, t: 3 * h[-1] - 2 * h[-2] + 5 * t + 4, [2, -1]),
    ("quadratic-forcing", lambda h, t: 2 * h[-1] - h[-2] + t * t - 3, [0, 5]),
])
def test_linear_rules_recover_deep_linear_rules_through_corruption(name, step, start):
    terms = series(step, start)
    inputs, ys = rows_of(terms)
    corrupted = list(ys)
    for r in (2, 9):
        corrupted[r] += 7
    for observed, tol in ((ys, 0), (tuple(corrupted), 2)):
        rules, solved = linear_rules(inputs, observed, SEQ_VARS, lambda length: dense_windows(len(ys), length, 8),
                                     tol)
        assert solved > 0
        exact = [e for e, bad in rules if all(e.evaluate(d, True) == y for d, y in zip(inputs, ys))]
        assert exact, (name, rules)
        assert exact[0].size >= 11
    again, _ = linear_rules(inputs, ys, SEQ_VARS, lambda length: dense_windows(len(ys), length, 8), 0)
    assert [str(e) for e, _ in again] == [str(e) for e, _ in linear_rules(
        inputs, ys, SEQ_VARS, lambda length: dense_windows(len(ys), length, 8), 0)[0]]


def test_linear_rules_find_nothing_in_noise():
    for seed in range(20):
        rng = random.Random(seed)
        terms = [rng.randint(-30, 30) for _ in range(23)]
        inputs, ys = rows_of(terms)
        rules, _ = linear_rules(inputs, ys, SEQ_VARS, lambda length: dense_windows(len(ys), length, 8),
                                int(0.4 * len(ys)))
        assert rules == [], (seed, rules)


def test_learner_validates_a_thirteen_node_rule_and_keeps_its_counters():
    terms = series(lambda h, t: 3 * h[-1] - 2 * h[-2] - 3 * t + 2, [1, 4], 30)
    learner = RuleLearner("s", SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes)
    for t in range(3, len(terms)):
        learner.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])
    assert learner.status() == "validated", learner.best()[0]
    assert learner.best()[0].expr.size == 13 and learner.linear_hits >= 1
    x = {"x[t-1]": terms[-1], "x[t-2]": terms[-2], "x[t-3]": terms[-3], "t": 30}
    assert learner.predict(x).value == 3 * terms[-1] - 2 * terms[-2] - 90 + 2
    clone = RuleLearner.from_state(learner.to_state(), learner.cfg, probe_fn=sequence_rollout_probes)
    assert (clone.linear_searches, clone.linear_hits) == (learner.linear_searches, learner.linear_hits)
    assert clone.predict(x).value == learner.predict(x).value


def test_linear_terms_zero_disables_the_solver():
    terms = series(lambda h, t: 3 * h[-1] - 2 * h[-2] - 3 * t + 2, [1, 4], 30)
    learner = RuleLearner("s", SEQ_VARS, InductionConfig(max_size=5, linear_terms=0),
                          probe_fn=sequence_rollout_probes)
    for t in range(3, len(terms)):
        learner.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])
    assert learner.linear_searches == 0 and learner.status() != "validated"


def test_spare_rows_step_over_corruption_denser_than_the_rule():
    terms = series(lambda h, t: 3 * h[-1] - 2 * h[-2] + 5 * t - 1, [1, -3, -2], 13)
    inputs, ys = rows_of(terms)
    ys = list(ys)
    for r in (1, 5, 8):  # no four consecutive clean rows
        ys[r] += 5
    target = "((((x[t-1] * 3) + (t * 5)) - (x[t-2] * 2)) + -1)"
    found = {spare: [str(e) for e, _ in linear_rules(inputs, ys, SEQ_VARS,
                                                    lambda length: dense_windows(len(ys), length, 4), 4,
                                                    spare=spare)[0]] for spare in (0, 1)}
    assert target not in found[0] and target in found[1]
