import random
import zlib

import pytest

from genesis import search
from genesis.environments.sequence_world import DEEP_FAMILIES, SequenceWorld
from genesis.expressions import parse_shape


def _random_case(rng):
    n = rng.randint(2, 14)
    names = ["a", "b", "t"][: rng.randint(1, 3)]
    inputs = [{"a": rng.randint(-9, 9), "b": rng.randint(0, 12), "t": t} for t in range(n)]
    k1, k2 = rng.randint(-4, 4), rng.randint(2, 9)
    f = rng.choice([lambda d: d["t"] * (d["t"] + k1) + k2, lambda d: (2 * d["a"] + k1) % k2,
                    lambda d: d["a"] + 2 * d["t"] - k1, lambda d: rng.randint(-50, 50),
                    lambda d: 2 * d["a"] - 3 * d["t"] + k1, lambda d: d["t"] * (2 * d["t"] + k1) - k2])
    ys = [f(d) for d in inputs]
    for _ in range(rng.randint(0, 2)):
        ys[rng.randrange(n)] += rng.randint(1, 9)
    consts = sorted({0, 1, 2, 3} | {rng.randint(-6, 12) for _ in range(rng.randint(0, 4))})
    tol = rng.randint(0, int(0.4 * n))
    return (inputs, tuple(ys), names, consts, tol, rng.choice([3, 5, 5]), rng.choice([300, 5000, 60000]), 100.0, 3,
            search.OPS, 3, rng.choice([-1, 0, tol, n]), 7, search._windows(n, rng.choice([2, 4, 6, 8]),
                                                                           rng.randint(1, 4)),
            rng.choice([0, 7, 9, 11]))


def test_c_kernel_matches_python_reference():
    if search.backend_info()["active"] != "c":
        pytest.skip("no C compiler: the Python reference is the only backend")
    rng = random.Random(11)
    deep_hits = affine_hits = 0
    for _ in range(160):
        args = _random_case(rng)
        c, py = search._search_c(*args), search._search_python(*args)
        assert search._summary(c) == search._summary(py)
        deep_hits += bool(py.deep)
        affine_hits += any(e.size > args[5] + 2 and e.op == "add" and e.args[1].op == "const" for e, _ in py.deep)
    assert deep_hits > 0 and affine_hits > 0


def test_bias_fit_tolerates_corrupted_observations():
    inputs = [{"x": x} for x in range(12)]
    ys = [3 * x + 7 for x in range(12)]
    ys[4] += 5
    ys[9] -= 2
    for backend in ["python"] + (["c"] if search.backend_info()["active"] == "c" else []):
        search.set_backend(backend)
        try:
            r = search.search_programs(inputs, ys, ["x"], [0, 1, 2, 3], 3, 3, 60000, 5.0, 3)
        finally:
            search.set_backend(None)
        assert ("(x * 3)", 7, 2) in [(str(e), v, b) for e, v, b in r.offsets]


def test_deepening_finds_rules_beyond_the_enumerated_size():
    inputs = [{"t": t} for t in range(10)]
    ys = [t * (t + 3) - 2 for t in range(10)]
    ys[6] += 4
    shallow = search.search_programs(inputs, ys, ["t"], [0, 1, 2, 3], 2, 5, 60000, 5.0, 3)
    assert not any(b <= 2 for _, b in shallow.matches) and not shallow.deep and not shallow.deep_ran
    deep = search.search_programs(inputs, ys, ["t"], [0, 1, 2, 3], 2, 5, 60000, 5.0, 3, deepen_bad=2, deep_size=7)
    assert deep.deep_ran
    assert all(e.size == 7 and b <= 2 for e, b in deep.deep)
    t = 20
    assert any(e.evaluate({"t": t}) == t * (t + 3) - 2 for e, _ in deep.deep)
    assert deep.counts[7] > deep.counts[5]


def test_iterative_deepening_reaches_nine_and_eleven_nodes():
    inputs = [{"a": a, "b": b, "t": t} for t, (a, b) in enumerate(zip([3, -1, 4, 1, -5, 9, 2, -6, 5, 3, 5, 8],
                                                                      [2, 7, 1, -8, 2, 8, -1, 8, 2, 8, 4, 5]))]
    cases = [  # (rule, smallest size, constants): none is reachable by enumeration (5) or size-7 deepening
        (lambda d: 2 * d["a"] - 3 * d["b"] + 7, 9, [0, 1, 2, 3]),
        (lambda d: 2 * d["a"] + 3 * d["t"] * d["t"] - 4, 11, [0, 1, 2, 3]),
    ]
    for rule, size, consts in cases:
        ys = [rule(d) for d in inputs]
        ys[1] += 6  # one corrupted observation (outside the most recent window)
        for backend in ["python"] + (["c"] if search.backend_info()["active"] == "c" else []):
            search.set_backend(backend)
            try:
                seven = search.search_programs(inputs, ys, ["a", "b", "t"], consts, 2, 5, 60000, 20.0, 3,
                                               deepen_bad=2, deep_size=7)
                deep = search.search_programs(inputs, ys, ["a", "b", "t"], consts, 2, 5, 60000, 20.0, 3,
                                              deepen_bad=2, deep_size=7, deep_max_size=11)
            finally:
                search.set_backend(None)
            assert not any(b <= 2 for _, b in seven.deep)
            probe = {"a": 40, "b": -13, "t": 30}
            hits = [e for e, b in deep.deep if b <= 2 and e.evaluate(probe) == rule(probe)]
            assert hits and min(e.size for e in hits) == size
            assert deep.counts[size] > deep.counts[size - 2]


def test_search_rejects_inconsistent_inputs(monkeypatch):
    with pytest.raises(ValueError):
        search.search_programs([{"x": 1}], [1, 2], ["x"], [0], 0, 3, 100, 1.0, 1)
    monkeypatch.setenv("GENESIS_SEARCH_BACKEND", "gpu")
    with pytest.raises(ValueError):
        search.search_programs([{"x": 1}], [1], ["x"], [0], 0, 3, 100, 1.0, 1)


def test_out_of_range_sizes_use_the_python_reference():
    inputs = [{"x": x} for x in range(6)]
    ys = [2 * x + 1 for x in range(6)]
    for max_size in (0, 65):
        ref = search._search_python(inputs, tuple(ys), ["x"], [0, 1, 2, 3], 0, max_size, 60000, 5.0, 3,
                                    search.OPS, 3, -1, 7, [])
        got = search.search_programs(inputs, ys, ["x"], [0, 1, 2, 3], 0, max_size, 60000, 5.0, 3)
        assert search._summary(got) == search._summary(ref)


def test_sequence_world_defaults_are_unchanged():
    out = []
    for seed in (0, 7, 123):
        w = SequenceWorld(seed)
        for sid in w.stream_ids():
            terms = [w.next_term(sid) for _ in range(25)]
            out.append((seed, sid, w._family(sid), zlib.crc32(repr(terms).encode())))
    assert zlib.crc32(repr(out).encode()) == 3985857382


def test_sequence_world_noise_is_deterministic_and_spares_noise_streams():
    def run():
        w = SequenceWorld(5, families=DEEP_FAMILIES, noise=0.2)
        return w, {sid: [w.next_term(sid) for _ in range(40)] for sid in w.stream_ids()}

    (w1, h1), (_, h2) = run(), run()
    assert h1 == h2
    assert sum(w1.corrupted.values()) > 0
    assert all(w1.corrupted[sid] == 0 for sid in w1.stream_ids() if w1._family(sid) == "noise")
    assert {w1._family(sid) for sid in w1.stream_ids()} - {"noise"} <= {g(random.Random(0))["family"]
                                                                         for g in DEEP_FAMILIES}


def test_shape_kernel_matches_python_reference():
    if search.backend_info()["active"] != "c":
        pytest.skip("no C compiler: the Python reference is the only backend")
    rng = random.Random(5)
    shapes = ["add(V,C)", "V", "mod(add(V,C),C)", "mul(V,add(V,C))", "sub(add(V,V),V)", "mod(add(mul(C,V),C),C)",
              "mul(V,V)", "sub(V,mul(V,C))"]
    n_hits = 0
    for _ in range(120):
        n = rng.randint(1, 20)
        inputs = [{"a": rng.randint(-9, 9), "t": t} for t in range(n)]
        k = rng.randint(-3, 3)
        ys = [rng.choice([d["t"] * (d["t"] + k) + 2, (2 * d["a"] + k) % 7, 2 * d["a"] - d["t"], rng.randint(-9, 9)])
              for d in inputs]
        consts = sorted({0, 1, 2, 3, 7} | {rng.randint(-5, 9) for _ in range(rng.randint(0, 3))})
        tol = rng.randint(0, (n - 1) // 2)
        fit = rng.random() < 0.5
        tree = parse_shape(rng.choice(shapes))
        c = search._shapes_c(tree, inputs, ys, ["a", "t"], consts, tol, fit)
        py = search._shapes_python(tree, inputs, ys, ["a", "t"], consts, tol, fit)
        assert c == py[:2]
        n_hits += len(py[0])
    assert n_hits > 0
    assert search.match_shape(parse_shape("if_eq(V,C,V,V)"), [{"a": 1}], [1], ["a"], [0], 0) is None


def test_bias_fit_at_half_tolerance_stays_on_the_kernel(monkeypatch):
    """A fitted constant must be the residual of a strict majority: a tolerance of n / 2 is capped below it, on both
    backends, instead of sending every bias-fitted shape to the slow Python path."""
    inputs = [{"a": a, "t": t} for t, a in enumerate([3, 1, 4, 1, 5, 9, 2, 6])]
    ys = [a + 2 if t % 2 else a + 5 for t, a in ((d["t"], d["a"]) for d in inputs)]
    tree = parse_shape("V")
    got = search.match_shape(tree, inputs, ys, ["a", "t"], [0, 1, 2], 4, fit_bias=True)
    assert got[0] == []  # +2 and +5 tie: no strict majority, no fitted constant
    ys[0] = inputs[0]["a"] + 2
    got = search.match_shape(tree, inputs, ys, ["a", "t"], [0, 1, 2], 4, fit_bias=True)
    assert got[0] == [(("a",), (), 2, 3)]
    if search.backend_info()["active"] == "c":
        assert got[2] == "c"
    monkeypatch.setenv("GENESIS_SEARCH_BACKEND", "python")
    assert search.match_shape(tree, inputs, ys, ["a", "t"], [0, 1, 2], 4, fit_bias=True)[:2] == got[:2]


def test_remembered_deep_shapes_transfer_to_new_worlds():
    from genesis.experiments import run_procedural_transfer

    world = {"families": DEEP_FAMILIES}
    on = run_procedural_transfer(1, n_worlds=3, procedural_prior=True, world_options=world)
    off = run_procedural_transfer(1, n_worlds=3, procedural_prior=False, world_options=world)
    assert on["accuracy_per_world"] == off["accuracy_per_world"] == [1.0, 1.0, 1.0]
    assert on["late_worlds_mean"] < off["late_worlds_mean"]
    assert on["late_worlds_search_cost_per_rule_stream"] * 3 < off["late_worlds_search_cost_per_rule_stream"]
