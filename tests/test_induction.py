import random

from genesis.adapters import SEQ_VARS, sequence_rollout_probes
from genesis.expressions import app, const, enumerate_programs, if_eq, var
from genesis.induction import InductionConfig, RuleLearner


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


def test_learner_state_round_trip():
    learner = seq_learner()
    feed_sequence(learner, [3 * t for t in range(12)])
    clone = RuleLearner.from_state(learner.to_state(), learner.cfg, probe_fn=sequence_rollout_probes)
    x = {"x[t-1]": 30, "x[t-2]": 27, "x[t-3]": 24, "t": 11}
    assert clone.predict(x).value == learner.predict(x).value == 33
    assert clone.status() == learner.status()
