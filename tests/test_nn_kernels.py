"""C kernels of genesis.nn: equivalence with their pure-Python twins, bit-exact determinism and fallback."""

import os
import random
import subprocess
import sys
from array import array

import pytest

from genesis.nn import backend, serialize
from genesis.nn import tensor as T
from genesis.nn.models import TinyGPT
from genesis.nn.train import train_lm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = backend.PYTHON_KERNELS


@pytest.fixture
def ck():
    if "c" not in backend.available_backends():
        pytest.skip(f"C backend unavailable: {backend.backend_info()['failure']}")
    return backend.kernels("c")


@pytest.fixture
def c_backend(ck):
    prev = backend.backend_info()["requested"]
    backend.set_backend("c")
    yield ck
    backend.set_backend(prev)


def rand(rng, n, lo=-2.0, hi=2.0):
    return array("d", [rng.uniform(lo, hi) for _ in range(n)])


def assert_close(got, want, tol=1e-9):
    if isinstance(want, tuple):
        assert len(got) == len(want)
        for u, v in zip(got, want):
            assert_close(u, v, tol)
    elif want is None or isinstance(want, float):
        assert (got is None) if want is None else abs(got - want) <= tol * max(1.0, abs(want))
    else:
        assert type(got) is array and got.typecode == "d" and len(got) == len(want)
        for p, q in zip(got, want):
            assert abs(p - q) <= tol * max(1.0, abs(q)), (p, q)


def test_matmul_matches_python(ck):
    rng = random.Random(0)
    # edge tiles, k > KC (256), m > MC (64), n > NC (512)
    for m, k, n in ((1, 1, 1), (4, 8, 8), (5, 3, 9), (70, 300, 20), (9, 2, 530)):
        for ta in (False, True):
            for tb in (False, True):
                a, b, o = rand(rng, m * k), rand(rng, k * n), rand(rng, m * n)
                assert_close(ck.matmul(a, b, m, k, n, ta, tb), PY.matmul(a, b, m, k, n, ta, tb))
                got, want = array("d", o), array("d", o)
                assert ck.matmul(a, b, m, k, n, ta, tb, got, True) is got
                assert_close(got, PY.matmul(a, b, m, k, n, ta, tb, want, True))
    assert list(ck.matmul(array("d"), array("d"), 0, 3, 0)) == []
    assert list(ck.matmul(array("d"), array("d"), 2, 0, 2)) == [0.0] * 4


def test_elementwise_kernels_match_python(ck):
    rng = random.Random(1)
    for n, (rows, cols) in ((1, (1, 1)), (7, (0, 5)), (1000, (25, 40))):
        a, b = rand(rng, n), rand(rng, n)
        assert_close(ck.add(a, b), PY.add(a, b))
        d = array("d", a)
        ck.add_(d, b)
        assert_close(d, PY.add(a, b))
        assert_close(ck.scale(a, -0.37), PY.scale(a, -0.37))
        assert_close(ck.sumsq(a), PY.sumsq(a))
        assert_close(ck.gelu(a), PY.gelu(a))
        _, th = PY.gelu(a)
        assert_close(ck.gelu_backward(b, a, th), PY.gelu_backward(b, a, th))
        x, bias = rand(rng, rows * cols), rand(rng, cols)
        assert_close(ck.add_rows(x, bias, rows, cols), PY.add_rows(x, bias, rows, cols))
        assert_close(ck.colsum(x, rows, cols), PY.colsum(x, rows, cols))
    big = array("d", [30.0, -30.0, 0.0, 1e-300, -1e-12])  # tanh saturation and tiny inputs
    assert_close(ck.gelu(big), PY.gelu(big))


def test_layer_norm_kernels_match_python(ck):
    rng = random.Random(2)
    for rows, cols in ((1, 1), (3, 5), (17, 64)):
        x, g, gamma, beta = rand(rng, rows * cols), rand(rng, rows * cols), rand(rng, cols, 0.5, 1.5), rand(rng, cols)
        assert_close(ck.layer_norm(x, gamma, beta, rows, cols, 1e-5), PY.layer_norm(x, gamma, beta, rows, cols, 1e-5))
        _, xhat, rstd = PY.layer_norm(x, gamma, beta, rows, cols, 1e-5)
        for flags in ((True, True, True), (True, False, True), (False, True, False), (False, False, False)):
            assert_close(ck.layer_norm_backward(g, xhat, rstd, gamma, rows, cols, *flags),
                         PY.layer_norm_backward(g, xhat, rstd, gamma, rows, cols, *flags))


def test_attention_kernels_match_python(ck):
    rng = random.Random(3)
    for t, d, h in ((1, 4, 1), (5, 6, 3), (33, 8, 2), (70, 16, 4)):  # crosses the 32-row causal blocks
        qkv, dy = rand(rng, 3 * t * d), rand(rng, t * d)
        y, P = ck.attention(qkv, t, d, h)
        assert_close((y, P), PY.attention(qkv, t, d, h))
        for i in range(h * t):
            row = P[i * t:(i + 1) * t]
            assert all(v == 0.0 for v in row[i % t + 1:]) and abs(sum(row) - 1.0) < 1e-12
        assert_close(ck.attention_backward(qkv, P, dy, t, d, h), PY.attention_backward(qkv, P, dy, t, d, h))
    with pytest.raises(ValueError):
        ck.attention(rand(rng, 3 * 2 * 6), 2, 6, 4)


def test_loss_embedding_and_optimizer_kernels_match_python(ck):
    rng = random.Random(4)
    for t, v in ((1, 1), (6, 11), (40, 300)):
        logits, tg = rand(rng, t * v, -30.0, 30.0), array("i", [rng.randrange(v) for _ in range(t)])
        assert_close(ck.cross_entropy(logits, tg, t, v), PY.cross_entropy(logits, tg, t, v))
        _, probs = PY.cross_entropy(logits, tg, t, v)
        assert_close(ck.cross_entropy_backward(probs, tg, t, v, 0.125), PY.cross_entropy_backward(probs, tg, t, v, 0.125))
        ids = array("i", [rng.randrange(v) for _ in range(t)] + [0, 0])  # repeated ids accumulate
        w, g = rand(rng, v * 5), rand(rng, len(ids) * 5)
        assert_close(ck.embedding(w, ids, v, 5), PY.embedding(w, ids, v, 5))
        assert_close(ck.embedding_backward(g, ids, v, 5), PY.embedding_backward(g, ids, v, 5))
    n = 333
    w, g, m, v2 = rand(rng, n), rand(rng, n), rand(rng, n), rand(rng, n, 0.0, 1.0)
    for keep in (1.0, 0.97):
        cw, cm, cv, pw, pm, pv = (array("d", z) for z in (w, m, v2, w, m, v2))
        ck.adamw(cw, g, cm, cv, 0.9, 0.95, 3e-3, 1.7, 1e-8, keep)
        PY.adamw(pw, g, pm, pv, 0.9, 0.95, 3e-3, 1.7, 1e-8, keep)
        assert_close((cw, cm, cv), (pw, pm, pv))


def test_c_kernels_validate_buffers(ck):
    a = array("d", [1.0] * 6)
    with pytest.raises(ValueError):
        ck.add(a, array("d", [1.0] * 5))
    with pytest.raises(ValueError):
        ck.add(a, [1.0] * 6)
    with pytest.raises(ValueError):
        ck.add(a, array("f", [1.0] * 6))
    with pytest.raises(ValueError):
        ck.matmul(a, a, 2, 3, 3)
    with pytest.raises(ValueError):
        ck.cross_entropy(a, array("i", [0, 3]), 2, 3)
    with pytest.raises(ValueError):
        ck.cross_entropy_backward(a, array("i", [-1, 0]), 2, 3, 1.0)
    with pytest.raises(ValueError):
        ck.embedding(a, array("i", [2]), 2, 3)
    with pytest.raises(ValueError):
        ck.embedding_backward(a[:3], array("i", [-1]), 2, 3)
    one = array("d", [1.0])
    for call in (lambda: ck.matmul(one, one, -1, -1, -1), lambda: ck.colsum(one, -1, -1),
                 lambda: ck.add_rows(one, one, -1, -1), lambda: ck.layer_norm(one, one, one, -1, -1, 1e-5),
                 lambda: ck.attention(array("d", [1.0] * 3), -1, -1, 1),
                 lambda: ck.embedding(one, array("i", [0]), -1, -1),
                 lambda: backend.matmul(one, one, -1, -1, -1)):
        with pytest.raises(ValueError):  # negative pairs have positive products: must not reach C as size_t
            call()


def test_matmul_bits_independent_of_simd_and_threads(ck):
    rng = random.Random(5)
    shapes = ((37, 70, 133), (37, 200, 300), (300, 210, 37), (130, 128, 129))  # all but the first are split
    data = [(rand(rng, m * k), rand(rng, k * n), rand(rng, m * n)) for m, k, n in shapes]

    def run():
        res = []
        for (m, k, n), (a, b, o) in zip(shapes, data):
            for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
                res.append(ck.matmul(a, b, m, k, n, ta, tb).tobytes())
                res.append(ck.matmul(a, b, m, k, n, ta, tb, array("d", o), True).tobytes())
        return res

    prev = backend.backend_info()["threads"]
    try:
        backend._c["set_simd"](1)
        backend.set_threads(1)
        ref = run()
        for simd in (2, 3, 0):
            for threads in (1, 2, 3):
                backend._c["set_simd"](simd)
                backend.set_threads(threads)
                assert run() == ref, (simd, threads)
    finally:
        backend._c["set_simd"](0)
        backend.set_threads(prev)
    assert backend.backend_info()["simd"] in ("scalar", "vec2", "avx")


def _gradcheck(fn, inputs, eps=1e-6, tol=1e-5):
    for t in inputs:
        t.grad = None
    fn().backward()
    for t in inputs:
        for i in range(0, t.size, max(1, t.size // 7)):
            old = t.data[i]
            with T.no_grad():
                t.data[i] = old + eps
                fp = fn().item()
                t.data[i] = old - eps
                fm = fn().item()
            t.data[i] = old
            num = (fp - fm) / (2 * eps)
            assert abs(num - t.grad[i]) <= tol * max(1.0, abs(num)), (t.shape, i, num, t.grad[i])


def test_fused_op_gradients_on_c_backend(c_backend):
    rng = random.Random(6)

    def p(shape, std=1.0):
        return T.randn(shape, rng, std, requires_grad=True)

    w = T.Tensor(rand(rng, 35 * 6, -1.0, 1.0), (35, 6))

    def project(out):
        return T.sum(T.mul(out, T.Tensor(w.data[:out.size], out.shape)))

    a, bias, x, gamma, beta = p((5, 6)), p((6,)), p((5, 6)), p((6,)), p((6,))
    _gradcheck(lambda: project(T.gelu(T.add(a, bias))), [a, bias])
    _gradcheck(lambda: project(T.layer_norm(x, gamma, beta)), [x, gamma, beta])
    _gradcheck(lambda: project(T.mean_rows(a)), [a])
    xa, ws = p((35, 6)), [p((6, 18), 0.5), p((18,), 0.5), p((6, 6), 0.5), p((6,), 0.5)]
    _gradcheck(lambda: project(T.causal_self_attention(xa, *ws, n_heads=3)), [xa, *ws])
    logits, emb = p((4, 9), 2.0), p((9, 6))
    _gradcheck(lambda: T.cross_entropy(logits, [0, 8, 2, 2]), [logits])
    _gradcheck(lambda: project(T.embedding(emb, [1, 3, 1, 0, 8])), [emb])
    m = TinyGPT(vocab_size=6, block_size=5, dim=6, n_layers=2, n_heads=2, seed=3)
    _gradcheck(lambda: m.loss([1, 4, 0, 1, 5], [4, 0, 1, 5, 2]), m.parameters())


def _train(seed=0, steps=3):
    m = TinyGPT(vocab_size=30, block_size=40, dim=16, n_layers=2, n_heads=2, seed=seed)
    hist = train_lm(m, [(i * 7 + i // 5) % 30 for i in range(300)], steps=steps, batch_size=2, lr=3e-3, seed=1)
    return serialize.digest(serialize.dumps(m)), hist["train_loss"]


def test_c_training_is_bit_reproducible_and_matches_python(c_backend):
    prev = backend.backend_info()["threads"]
    try:
        backend.set_threads(1)
        one = _train()
        backend.set_threads(2)
        assert _train() == _train() == one  # same bits for any run and thread count
    finally:
        backend.set_threads(prev)
    assert _train(seed=1)[0] != one[0]
    backend.set_backend("python")
    digest, losses = _train()
    for x, y in zip(losses, one[1]):
        assert abs(x - y) <= 1e-9 * abs(y)


def test_old_python_produced_weights_load_and_run_on_c(c_backend):
    backend.set_backend("python")
    m = TinyGPT(vocab_size=9, block_size=8, dim=8, n_layers=1, n_heads=2, seed=2)
    train_lm(m, [i % 9 for i in range(60)], steps=2, batch_size=2, seed=0)
    data, ref = serialize.dumps(m), m.next_logits([1, 2, 3])
    backend.set_backend("c")
    clone = serialize.loads(data)
    assert serialize.dumps(clone) == data
    assert_close(array("d", clone.next_logits([1, 2, 3])), array("d", ref))


_FALLBACK_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from genesis.nn import TinyGPT, backend, serialize, train_lm
info = backend.backend_info()
m = TinyGPT(vocab_size=7, block_size=6, dim=8, n_layers=1, n_heads=2, seed=3)
train_lm(m, [i * 3 % 7 for i in range(40)], steps=2, batch_size=2, lr=1e-2, seed=1)
print(info["name"], info["failure"] is None, info["simd"], info["threads"], serialize.digest(serialize.dumps(m)))
"""


def test_python_fallback_when_c_cannot_be_built(tmp_path):
    script = _FALLBACK_SCRIPT.format(root=ROOT)
    no_cc = tmp_path / "bin"
    no_cc.mkdir()
    cache = tmp_path / "cache"
    outs = []
    for extra in ({"GENESIS_NN_BACKEND": "python"}, {"PATH": str(no_cc), "GENESIS_CACHE_DIR": str(cache)}):
        env = {k: v for k, v in os.environ.items() if k != "GENESIS_NN_BACKEND"}
        env.update(extra)
        res = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT, capture_output=True, text=True,
                             timeout=120)
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.split())
    assert outs[0][:4] == ["python", "True", "None", "None"]
    assert outs[1][:4] == ["python", "False", "None", "None"]  # auto: compile failure reported, not raised
    assert outs[0][4] == outs[1][4]
