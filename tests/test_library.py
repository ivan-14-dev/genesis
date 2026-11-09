from genesis import library
from genesis.adapters import SEQ_VARS, sequence_dream, sequence_rollout_probes
from genesis.agent import Genesis
from genesis.environments import sandboxed
from genesis.environments.sequence_world import SequenceWorld
from genesis.expressions import app, const, var
from genesis.induction import InductionConfig, RuleLearner

X1, X2, T = var("x[t-1]"), var("x[t-2]"), var("t")
TWO_X1_MINUS_X2 = app("sub", app("mul", X1, const(2)), X2)  # 5 nodes
SAME_BEHAVIOUR = app("sub", app("add", X1, X1), X2)


def test_promotion_merges_behaviourally_equal_subprograms_across_problems():
    rules = [("a", app("add", TWO_X1_MINUS_X2, T)), ("b", app("add", SAME_BEHAVIOUR, const(3))),
             ("c", app("mul", X1, const(3)))]
    inv = library.promote(rules)
    assert len(inv) == 1
    assert inv[0].name == "@1" and inv[0].support == 2 and inv[0].uses == 2
    assert inv[0].gain == 2 * (5 - 1) - 5 - library.BASIS_COST
    assert str(inv[0].expr) in (str(TWO_X1_MINUS_X2), str(SAME_BEHAVIOUR))
    assert [i.to_json() for i in library.promote(list(reversed(rules)))] == [i.to_json() for i in inv]
    assert library.Invention.from_json(inv[0].to_json()) == inv[0]


def test_promotion_needs_support_from_several_problems_and_a_saving():
    one_problem = [("a", app("add", TWO_X1_MINUS_X2, T)), ("a", app("add", TWO_X1_MINUS_X2, const(1)))]
    assert library.promote(one_problem) == []
    too_small = [("a", app("add", app("mul", X1, const(2)), T)), ("b", app("sub", app("mul", X1, const(2)), T))]
    assert library.promote(too_small) == []  # 2 uses of a 3-node sub-program do not pay for its definition


def test_inline_restores_innate_primitives():
    defs = {"@1": TWO_X1_MINUS_X2}
    e = app("add", var("@1"), T)
    flat = library.inline(e, defs)
    assert flat == app("add", TWO_X1_MINUS_X2, T)
    env = {"x[t-1]": 7, "x[t-2]": -2, "t": 5}
    assert flat.evaluate(env) == e.evaluate({**env, "@1": TWO_X1_MINUS_X2.evaluate(env)})


def _learn(terms, inventions):
    cfg = InductionConfig(max_size=3, deepen_size=0)
    lr = RuleLearner("s", SEQ_VARS, cfg, probe_fn=sequence_rollout_probes,
                     learned_library=(lambda: (inventions, None)) if inventions is not None else None)
    for t in range(3, len(terms)):
        lr.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])
    return lr


def test_invention_brings_a_rule_within_the_search_budget():
    terms = [1, 4, 2]
    while len(terms) < 24:  # x[t] = 2 x[t-1] - x[t-2] + t: 7 nodes, beyond a 3-node budget
        t = len(terms)
        terms.append(2 * terms[-1] - terms[-2] + t)
    without = _learn(terms, None)
    assert without.status() != "validated"
    inv = library.Invention("@1", TWO_X1_MINUS_X2, 2, 2, 1.0)
    lr = _learn(terms, [inv])
    assert lr.status() == "validated"
    best = lr.best()[0].expr
    assert not any(v.startswith("@") for v in best.variables())  # stored inlined, in innate primitives
    assert all(best.evaluate({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t})
               == terms[t] for t in range(3, len(terms)))
    assert lr.invention_hits >= 1
    state = lr.to_state()
    assert RuleLearner.from_state(state, lr.cfg).invention_hits == lr.invention_hits


def test_inventions_over_unperceived_variables_are_ignored():
    inv = library.Invention("@1", app("add", app("mul", var("u"), const(2)), var("v")), 2, 2, 1.0)
    lr = _learn([1, 2, 3, 4, 5, 6, 7, 8], [inv])
    assert lr._usable_inventions([{"x[t-1]": 1, "x[t-2]": 0, "x[t-3]": 0, "t": 3}]) == {}


def test_features_and_recognizer_identify_the_family_of_a_series():
    arithmetic = [3 + 4 * i for i in range(12)]
    assert "d1-constant" in library.features(arithmetic)
    geometric = [3 * 2 ** i for i in range(12)]
    assert {"ratio-growth", "fast-growth"} <= library.features(geometric)
    noisy = list(arithmetic)
    noisy[5] = 999
    assert "d1-constant" in library.features(noisy)  # one corrupted observation does not flip a feature
    shapes = ["add(V,C)", "mul(V,C)"]
    examples = library.dream(shapes, SEQ_VARS, sequence_dream, seed=4, per_shape=32)
    assert examples == library.dream(shapes, SEQ_VARS, sequence_dream, seed=4, per_shape=32)
    assert len(examples) == 64
    rec = library.Recognizer()
    rec.train(examples)
    assert rec.rank(shapes, library.features(geometric))[0] == "mul(V,C)"  # only products grow geometrically
    replays = library.Recognizer()
    replays.train([("add(V,C)", library.features(arithmetic)), ("mul(V,C)", library.features(geometric))])
    assert replays.rank(shapes, library.features(noisy))[0] == "add(V,C)"
    again = library.Recognizer.from_json(rec.to_json())
    assert again.rank(list(reversed(shapes)), library.features(geometric))[0] == "mul(V,C)"
    assert rec.rank(["unknown", *shapes], library.features(geometric))[-1] == "unknown"


def test_sleep_stores_the_library_and_skips_when_nothing_changed(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"), seed=1)
    world = SequenceWorld(1)
    agent.explore_sequences(sandboxed(world, 200), 150)
    lib = agent.ltm.get_meta("library")
    assert lib is not None and lib["replays"] > 0 and lib["dreams"] > 0 and lib["recognizer"]["totals"]
    assert agent.consolidate()["library"] == {"unchanged": True}
    # Dreams and inventions never become beliefs: every stored rule uses perceived variables only.
    for k in agent.ltm.knowledge(kind="rule"):
        assert "@" not in k.statement
    agent.close()


def test_library_learning_can_be_disabled(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"), seed=1, library_learning=False)
    agent.explore_sequences(sandboxed(SequenceWorld(1), 120), 100)
    assert agent.ltm.get_meta("library") is None
    assert agent.report()["invented_primitives"] == []
    agent.close()
