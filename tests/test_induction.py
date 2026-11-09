import random

import pytest

from genesis import library
from genesis.adapters import SEQ_VARS, sequence_dream, sequence_rollout_probes
from genesis.expressions import app, const, enumerate_programs, if_eq, var
from genesis.induction import Hypothesis, InductionConfig, RuleLearner


def seq_learner(key="s"):
    return RuleLearner(key, SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes)


def feed_sequence(learner, terms):
    for t in range(3, len(terms)):
        learner.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])


def test_expression_evaluation_and_serialisation():
    e = if_eq("p", 2, app("add", var("x"), const(3)), var("x"))
    assert e.evaluate({"p": 2, "x": 4}) == 7
    assert e.evaluate({"p": 1, "x": 4}) == 4
    assert e.condition_variable() == "p"
    assert type(e).from_json(e.to_json()) == e
    assert app("mod", var("x"), const(0)).evaluate({"x": 5}) is None


def test_belief_evaluation_is_exact_beyond_the_search_bound():
    double = app("mul", var("x"), const(2))
    assert double.evaluate({"x": 10**18}) is None  # bounded innate semantics (search kernels)
    assert double.evaluate({"x": 10**18}, True) == 2 * 10**18
    square = app("mul", var("x"), var("x"))
    assert square.evaluate({"x": 1 << 3000}, True) is None  # exact values are length-capped
    assert app("mod", var("x"), const(0)).evaluate({"x": 5}, True) is None


def test_geometric_rule_survives_past_the_search_bound():
    rng = random.Random(3)
    terms = [3, 6]
    while len(terms) < 90:  # 3 * 2^t passes 10^18 near t = 59
        v = terms[-1] * 2
        if len(terms) % 7 == 0:
            v += rng.randint(1, 15) * rng.choice([-1, 1])  # innovation noise: later terms continue from it
        terms.append(v)
    learner = seq_learner()
    feed_sequence(learner, terms)
    best, _, _ = learner.best()
    assert learner.status() == "validated"
    assert str(best.expr) in ("(x[t-1] * 2)", "(x[t-1] + x[t-1])")
    nxt = {"x[t-1]": terms[-1], "x[t-2]": terms[-2], "x[t-3]": terms[-3], "t": len(terms)}
    assert learner.predict(nxt).value == 2 * terms[-1] > 10**27


def test_enumeration_is_deduplicated_and_grows_with_size():
    rng = random.Random(0)
    probes = [{"x": rng.randint(-9, 9), "y": rng.randint(-9, 9)} for _ in range(10)]
    en = enumerate_programs(probes, ["x", "y"], [0, 1, 2], 4, 100000, 5.0)
    exprs = [str(e) for e, _ in en.candidates]
    assert len(exprs) == len(set(exprs))
    assert "(x + y)" in exprs or "(y + x)" in exprs
    assert en.counts[1] < en.counts[3]


def test_discovers_arithmetic_rule_from_observation():
    learner = seq_learner()
    feed_sequence(learner, [5 + 7 * t for t in range(14)])
    best, mass, _ = learner.best()
    assert learner.status() == "validated"
    assert best.expr.evaluate({"x[t-1]": 100, "x[t-2]": 93, "x[t-3]": 86, "t": 14}) == 107


def test_discovers_fibonacci_like_recurrence():
    terms = [2, 5]
    while len(terms) < 16:
        terms.append(terms[-1] + terms[-2])
    learner = seq_learner()
    feed_sequence(learner, terms)
    assert learner.status() == "validated"
    assert learner.predict({"x[t-1]": 10, "x[t-2]": 6, "x[t-3]": 4, "t": 30}).value == 16


def test_discovers_modular_counter():
    learner = seq_learner()
    feed_sequence(learner, [t % 5 for t in range(24)])
    assert learner.status() == "validated"
    assert [learner.predict({"x[t-1]": t % 5, "x[t-2]": (t - 1) % 5, "x[t-3]": (t - 2) % 5, "t": t + 1}).value
            for t in range(30, 35)] == [(t + 1) % 5 for t in range(30, 35)]


def test_noise_is_judged_unpredictable_and_never_validated():
    rng = random.Random(3)
    learner = seq_learner()
    statuses = []
    terms = [rng.randint(0, 99) for _ in range(30)]
    for t in range(3, len(terms)):
        learner.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])
        statuses.append(learner.status())
    assert "validated" not in statuses
    assert learner.status() == "unpredictable"


def test_no_knowledge_before_experience():
    learner = seq_learner()
    p = learner.predict({"x[t-1]": 1, "x[t-2]": 2, "x[t-3]": 3, "t": 4})
    assert learner.status() == "unknown"
    assert p.value is None or p.confidence < 0.5


def test_discovers_conditional_rule():
    rng = random.Random(1)
    learner = RuleLearner("c", ["p0", "p1"], InductionConfig(max_size=3, allow_conditionals=True, outcome_range=101))
    for _ in range(40):
        s = {"p0": rng.randint(0, 5), "p1": rng.randint(0, 5)}
        y = 4 if s["p1"] == 2 else s["p0"]
        learner.observe(s, y)
    assert learner.status() == "validated"
    assert learner.best()[0].expr.condition_variable() == "p1"


def test_discovers_conditional_rule_through_corrupted_outcomes():
    rng = random.Random(2)
    cfg = InductionConfig(max_size=3, allow_conditionals=True, outcome_range=101, deepen_size=0)
    learner = RuleLearner("c", ["p0", "p1", "p2"], cfg)
    corrupted = 0
    for i in range(60):
        s = {p: rng.randint(0, 5) for p in ("p0", "p1", "p2")}
        y = s["p2"] if s["p0"] == 3 else s["p1"]  # "if p0 == 3: copy p2", else unchanged
        if i == 1 or rng.random() < 0.1:  # an early corrupted outcome: no exact conditional ever exists
            y += rng.choice([-3, -2, -1, 1, 2, 3])
            corrupted += 1
        learner.observe(s, y)
    best = learner.best()[0].expr
    assert corrupted >= 3
    assert learner.status() == "validated"
    assert best.condition_variable() == "p0" and best.args[1].value == 3
    assert [best.evaluate({"p0": p0, "p1": 1, "p2": 4}) for p0 in (2, 3)] == [1, 4]


def test_learner_state_round_trip():
    learner = seq_learner()
    feed_sequence(learner, [3 * t for t in range(12)])
    clone = RuleLearner.from_state(learner.to_state(), learner.cfg, probe_fn=sequence_rollout_probes)
    x = {"x[t-1]": 30, "x[t-2]": 27, "x[t-3]": 24, "t": 11}
    assert clone.predict(x).value == learner.predict(x).value == 33
    assert clone.status() == learner.status()


def test_learns_through_corrupted_observations():
    rng = random.Random(4)
    terms = []
    for t in range(40):
        terms.append(4 * t - 9 + (rng.choice([-7, 3, 11]) if t > 2 and rng.random() < 0.2 else 0))
    learner = seq_learner()
    feed_sequence(learner, terms)
    assert learner.status() == "validated"
    assert str(learner.best()[0].expr) in ("(x[t-1] + 4)", "((4 * t) + -9)", "((t * 4) + -9)")
    assert learner.best()[0].expr.evaluate({"x[t-1]": 151, "x[t-2]": 147, "x[t-3]": 143, "t": 40}) == 151


def test_structured_errors_block_validation_of_incomplete_rules():
    learner = seq_learner()
    feed_sequence(learner, [t % 5 for t in range(30)])
    assert learner.status() == "validated"
    incomplete = Hypothesis(app("add", var("x[t-1]"), const(1)), 0.0, n_ok=21, n_bad=6, fwd_ok=21, fwd_bad=6)
    assert learner._structured_errors(incomplete)
    assert not learner._structured_errors(learner.best()[0])


@pytest.mark.parametrize("family,make", [
    ("quadratic-linear", lambda t, h: t * (t + 3) - 4),
    ("second-order", lambda t, h: 2 * h[-1] - h[-2] + 3),
    ("affine-modular", lambda t, h: (2 * h[-1] + 5) % 11),
])
def test_deepening_learns_size_seven_rules(family, make):
    terms = [1, 4, 2]
    for t in range(3, 40):
        terms.append(make(t, terms))
    learner = seq_learner()
    feed_sequence(learner, terms)
    assert learner.status() == "validated", (family, learner.best()[0])
    assert learner.best()[0].expr.size == 7 and learner.deep_searches >= 1
    h = terms[-3:]
    x = {"x[t-1]": h[-1], "x[t-2]": h[-2], "x[t-3]": h[-3], "t": 40}
    assert learner.predict(x).value == make(40, terms)
    clone = RuleLearner.from_state(learner.to_state(), learner.cfg, probe_fn=sequence_rollout_probes)
    assert clone.next_deep_n == learner.next_deep_n and clone.predict(x).value == make(40, terms)


def test_sequential_test_tolerates_a_burst_above_the_error_threshold():
    learner = seq_learner()
    assert learner._sprt_accepts(Hypothesis(var("t"), 0.0, fwd_ok=50, fwd_bad=19))  # 27.5% over a long record
    assert not learner._sprt_accepts(Hypothesis(var("t"), 0.0, fwd_ok=20, fwd_bad=12))  # 37.5%: no evidence
    assert not learner._sprt_accepts(Hypothesis(var("t"), 0.0, fwd_ok=9, fwd_bad=0))  # short: fast criterion's job
    off = RuleLearner("s", SEQ_VARS, InductionConfig(max_size=5, validation_sprt_alpha=0))
    assert not off._sprt_accepts(Hypothesis(var("t"), 0.0, fwd_ok=500, fwd_bad=170))


def test_unobserved_modulus_is_proposed_from_residue_classes():
    terms = [8]
    while len(terms) < 30:
        terms.append((3 * terms[-1] + 2) % 11)  # cycles through 8, 4, 3, 0, 2: 11 is never observed
    learner = seq_learner()
    inputs = [{"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t} for t in range(3, 30)]
    assert 11 in learner._propose_moduli(inputs, terms[3:])
    rng = random.Random(0)
    noise = [rng.randint(-30, 30) for _ in range(30)]
    assert learner._propose_moduli([{"x[t-1]": noise[t - 1], "x[t-2]": noise[t - 2], "x[t-3]": noise[t - 3],
                                     "t": t} for t in range(3, 30)], noise[3:]) == set()
    feed_sequence(learner, terms)
    assert learner.status() == "validated" and "% 11" in str(learner.best()[0].expr)


def test_promising_long_rule_keeps_the_question_open_through_dense_corruption():
    terms, corrupt = [1, -3, -2], {4: 2, 8: -2, 11: -7, 14: 12}  # no 4 consecutive clean observations early on
    for t in range(3, 34):
        terms.append(3 * terms[-1] - 2 * terms[-2] + 5 * t - 1 + corrupt.get(t, 0))
    learner = seq_learner()
    history = []
    for t in range(3, len(terms)):
        learner.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])
        history.append(learner.status())
    assert "unpredictable" in history[:10] and history[-1] == "validated"
    assert str(learner.best()[0].expr) == "((((x[t-1] * 3) + (t * 5)) - (x[t-2] * 2)) + -1)"


def test_recognized_noise_postpones_deeper_searches_but_not_on_rules():
    lib = library.sleep([], ["add(V,C)", "mul(V,C)"], [], {"sequence": (SEQ_VARS, sequence_dream)}, seed=3)
    rec = library.Recognizer.from_json(lib["recognizer"])

    def run(terms, learned):
        learner = RuleLearner("s", SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes,
                              learned_library=learned)
        feed_sequence(learner, terms)
        return learner

    rng = random.Random(4)
    noise = [rng.randint(0, 60) for _ in range(40)]
    plain, judged = run(noise, None), run(noise, lambda: ([], rec))
    assert judged.noise_delays >= 1 and judged.deep_searches < plain.deep_searches
    assert judged.expressions_evaluated < plain.expressions_evaluated
    assert judged.status() == plain.status() == "unpredictable"
    clone = RuleLearner.from_state(judged.to_state(), judged.cfg, probe_fn=sequence_rollout_probes)
    assert clone.noise_delays == judged.noise_delays and clone.next_deep_n == judged.next_deep_n
    terms = [1, 4, 0]
    for t in range(3, 40):
        terms.append(2 * terms[-1] - terms[-2] + 3 * t - 7)
    learned, alone = run(terms, lambda: ([], rec)), run(terms, None)
    assert learned.noise_delays == 0 and learned.status() == alone.status() == "validated"
    assert learned.deep_searches == alone.deep_searches
