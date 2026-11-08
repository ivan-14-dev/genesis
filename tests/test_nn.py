import json
import math
import os
import random
import struct
import subprocess
import sys
from array import array

import pytest

from genesis.nn import backend, serialize
from genesis.nn import tensor as T
from genesis.nn.models import MLPClassifier, TinyGPT
from genesis.nn.train import evaluate_classifier, evaluate_lm, train_classifier, train_lm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def py_backend():
    prev = backend.backend_info()["requested"]
    backend.set_backend("python")
    yield
    backend.set_backend(prev)


def param(shape, rng, std=1.0, away_from_zero=False):
    t = T.randn(shape, rng, std, requires_grad=True)
    if away_from_zero:  # keep relu inputs clear of the kink
        t.data[:] = array("d", [v + (0.2 if v >= 0 else -0.2) for v in t.data])
    return t


def project(out, seed=123):
    """Random linear functional so every output element gets a distinct gradient."""
    r = random.Random(seed)
    w = T.Tensor(array("d", [r.uniform(-1.0, 1.0) for _ in range(out.size)]), out.shape)
    return T.sum(T.mul(out, w))


def gradcheck(fn, inputs, eps=1e-6, tol=1e-5, indices=None):
    for t in inputs:
        t.grad = None
    loss = fn()
    assert loss.shape == () and loss.requires_grad
    loss.backward()
    for t in inputs:
        assert t.grad is not None and len(t.grad) == t.size
        for i in (range(t.size) if indices is None else indices(t)):
            old = t.data[i]
            with T.no_grad():
                t.data[i] = old + eps
                fp = fn().item()
                t.data[i] = old - eps
                fm = fn().item()
            t.data[i] = old
            num = (fp - fm) / (2 * eps)
            assert abs(num - t.grad[i]) <= tol * max(1.0, abs(num)), (t.shape, i, num, t.grad[i])


def test_elementwise_and_reduction_gradients(py_backend):
    rng = random.Random(0)
    a, b, bias = param((3, 4), rng), param((3, 4), rng), param((4,), rng)
    gradcheck(lambda: project(T.add(a, b)), [a, b])
    gradcheck(lambda: project(T.add(a, bias)), [a, bias])
    gradcheck(lambda: project(T.mul(a, b)), [a, b])
    gradcheck(lambda: project(T.scale(a, -1.7)), [a])
    gradcheck(lambda: project(T.tanh(a)), [a])
    gradcheck(lambda: project(T.gelu(T.scale(a, 2.0))), [a])
    r = param((3, 4), rng, away_from_zero=True)
    gradcheck(lambda: project(T.relu(r)), [r])
    gradcheck(lambda: project(T.mean_rows(a)), [a])
    gradcheck(lambda: T.mean(T.mul(a, b)), [a, b])
    gradcheck(lambda: T.sum(T.tanh(a)), [a])


def test_matmul_and_embedding_gradients(py_backend):
    rng = random.Random(1)
    a, b, bt = param((3, 4), rng), param((4, 5), rng), param((5, 4), rng)
    gradcheck(lambda: project(T.matmul(a, b)), [a, b])
    gradcheck(lambda: project(T.matmul(a, bt, trans_b=True)), [a, bt])
    w = param((5, 3), rng)
    gradcheck(lambda: project(T.embedding(w, [1, 3, 1, 0, 1])), [w])
    with pytest.raises(ValueError):
        T.embedding(w, [5])
    with pytest.raises(ValueError):
        T.embedding(w, [-1])


def test_layer_norm_attention_cross_entropy_gradients(py_backend):
    rng = random.Random(2)
    x, g, b = param((3, 5), rng), param((5,), rng), param((5,), rng)
    gradcheck(lambda: project(T.layer_norm(x, g, b)), [x, g, b])

    D = 4
    xa = param((4, D), rng)
    ws = [param((D, 3 * D), rng, 0.5), param((3 * D,), rng, 0.5), param((D, D), rng, 0.5), param((D,), rng, 0.5)]
    gradcheck(lambda: project(T.causal_self_attention(xa, *ws, n_heads=2)), [xa, *ws])

    with T.no_grad():  # causal mask: changing the last position leaves earlier outputs untouched
        before = T.causal_self_attention(xa, *ws, n_heads=2).tolist()
        saved = array("d", xa.data)
        xa.data[3 * D:] = array("d", [5.0] * D)
        after = T.causal_self_attention(xa, *ws, n_heads=2).tolist()
        xa.data[:] = saved
    assert after[:3] == before[:3] and after[3] != before[3]

    logits = param((4, 6), rng, 2.0)
    gradcheck(lambda: T.cross_entropy(logits, [0, 5, 2, 2]), [logits])
    with T.no_grad():
        ce = T.cross_entropy(logits, [0, 5, 2, 2]).item()
    rows = logits.tolist()
    ref = sum(math.log(sum(math.exp(v) for v in r)) - r[y] for r, y in zip(rows, [0, 5, 2, 2])) / 4
    assert abs(ce - ref) < 1e-12


def test_tinygpt_end_to_end_gradients(py_backend):
    m = TinyGPT(vocab_size=5, block_size=4, dim=4, n_layers=1, n_heads=2, mlp_mult=2, seed=3)
    rng = random.Random(4)
    for _, p in m.named_parameters():  # larger random weights make every path matter
        p.data[:] = array("d", [rng.gauss(0.0, 0.5) for _ in range(p.size)])
    ids, targets = [1, 4, 0, 1], [4, 0, 1, 3]
    gradcheck(lambda: m.loss(ids, targets), m.parameters(), indices=lambda t: range(0, t.size, max(1, t.size // 3)))


def test_tensor_reuse_accumulates_gradients(py_backend):
    x = T.tensor_from_list([[0.5, -1.0], [2.0, 3.0]], requires_grad=True)
    T.sum(T.add(T.mul(x, x), T.scale(x, 3.0))).backward()
    assert list(x.grad) == [2 * v + 3 for v in x.data]
    T.sum(T.add(x, x)).backward()  # leaf grads accumulate across backward calls
    assert list(x.grad) == [2 * v + 3 + 2 for v in x.data]
    with T.no_grad():
        y = T.mul(x, x)
    assert not y.requires_grad


def test_backend_equivalence():
    if "c" not in backend.available_backends():
        pytest.skip(f"C backend unavailable: {backend.backend_info()['failure']}")
    prev = backend.backend_info()["requested"]
    rng = random.Random(7)
    try:
        for m, k, n in ((7, 13, 9), (3, 70, 261)):
            for ta in (False, True):
                for tb in (False, True):
                    a = array("d", [rng.uniform(-1, 1) for _ in range(m * k)])
                    b = array("d", [rng.uniform(-1, 1) for _ in range(k * n)])
                    out0 = array("d", [rng.uniform(-1, 1) for _ in range(m * n)])
                    res = {}
                    for name in ("python", "c"):
                        backend.set_backend(name)
                        acc = array("d", out0)
                        backend.matmul(a, b, m, k, n, ta, tb, out=acc, accumulate=True)
                        res[name] = (backend.matmul(a, b, m, k, n, trans_a=ta, trans_b=tb), acc)
                    for x, y in zip(res["python"], res["c"]):
                        assert max(abs(p - q) for p, q in zip(x, y)) < 1e-9
        assert backend.backend_info()["kernel_source_sha256"] == backend.kernel_source_digest()
    finally:
        backend.set_backend(prev)
    with pytest.raises(ValueError):
        backend.set_backend("gpu")


def _toy_examples():
    rng = random.Random(1)
    return [([rng.randrange(4 * (i % 3), 4 * (i % 3) + 4) for _ in range(rng.randint(1, 5))], i % 3)
            for i in range(60)]


def test_mlp_classifier_learns_separable_task():
    ex = _toy_examples()
    m = MLPClassifier(12, 3, dim=8, hidden=16, seed=0)
    hist = train_classifier(m, ex, epochs=10, lr=3e-2, seed=0)
    assert hist["accuracy"][-1] == 1.0 and evaluate_classifier(m, ex) == 1.0
    p = m.predict_proba([0, 1])
    assert abs(sum(p) - 1.0) < 1e-12 and m.predict([0, 1]) == 0


def test_tinygpt_learns_cycle_and_generates():
    m = TinyGPT(vocab_size=6, block_size=8, dim=16, n_layers=1, n_heads=2, seed=0)
    hist = train_lm(m, [0, 1, 2, 3, 4, 5] * 20, steps=80, batch_size=4, lr=1e-2, seed=0)
    assert sum(hist["train_loss"][-5:]) / 5 < 0.3 < hist["train_loss"][0]
    assert evaluate_lm(m, [0, 1, 2, 3, 4, 5] * 5)["nll"] < 0.3
    assert m.generate([0, 1, 2], 9, random.Random(0), temperature=0) == [3, 4, 5, 0, 1, 2, 3, 4, 5]
    # long prompts are cropped to block_size; sampling is reproducible for a given rng
    assert m.generate([0, 1, 2, 3, 4, 5] * 3, 3, random.Random(0), temperature=0) == [0, 1, 2]
    s1 = m.generate([0], 12, random.Random(9), temperature=1.0, top_k=2)
    assert s1 == m.generate([0], 12, random.Random(9), temperature=1.0, top_k=2) and len(s1) == 12
    assert m.generate([0, 1], 9, random.Random(0), temperature=0, stop_ids=(5,)) == [2, 3, 4]


def _trained_bytes(model_seed, train_seed):
    m = TinyGPT(vocab_size=7, block_size=6, dim=8, n_layers=1, n_heads=2, seed=model_seed)
    train_lm(m, [i * 3 % 7 for i in range(50)], steps=8, batch_size=2, lr=1e-2, seed=train_seed)
    return serialize.dumps(m)


def test_training_is_deterministic():
    a = _trained_bytes(0, 0)
    assert a == _trained_bytes(0, 0)
    assert a != _trained_bytes(1, 0) and a != _trained_bytes(0, 1)


def _rewrite(data, fn):
    hlen = struct.unpack_from("<I", data, len(serialize.MAGIC))[0]
    start = len(serialize.MAGIC) + 4
    header = json.loads(data[start:start + hlen])
    fn(header)
    hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return serialize.MAGIC + struct.pack("<I", len(hb)) + hb + data[start + hlen:]


def test_serialize_round_trip_and_rejects_malformed():
    gpt = TinyGPT(vocab_size=7, block_size=6, dim=8, n_layers=2, n_heads=2, seed=5)
    data = serialize.dumps(gpt)
    clone = serialize.loads(data)
    assert serialize.dumps(clone) == data and serialize.digest(data) == serialize.digest(serialize.dumps(clone))
    assert clone.next_logits([1, 2, 3]) == gpt.next_logits([1, 2, 3])
    assert [n for n, _ in clone.named_parameters()] == [n for n, _ in gpt.named_parameters()]
    mlp = MLPClassifier(12, 3, dim=4, hidden=5, seed=2)
    mclone = serialize.loads(serialize.dumps(mlp))
    assert mclone.predict_proba([1, 5, 9]) == mlp.predict_proba([1, 5, 9])
    assert _rewrite(data, lambda h: None) == data  # header is canonical JSON

    def set_shape(h):
        h["params"][0][1] = [7, 9]

    def huge(h):
        h["config"]["vocab_size"] = 10 ** 9
        h["params"][0][1][0] = 10 ** 9

    bad = [b"XXXXX\n" + data[6:], data[:-8], data + b"\0", data[:8], 42,
           _rewrite(data, set_shape), _rewrite(data, huge),
           _rewrite(data, lambda h: h.update({"class": "os.system"})),
           _rewrite(data, lambda h: h.update({"format_version": 2})),
           _rewrite(data, lambda h: h["config"].update({"dim": 6})),
           _rewrite(data, lambda h: h["config"].update({"evil": 1})),
           _rewrite(data, lambda h: h.update({"extra": True})),
           serialize.MAGIC + struct.pack("<I", 1 << 30) + data[10:],
           serialize.MAGIC + struct.pack("<I", 5) + b"{oops" + data[10:],
           data[:-8] + struct.pack("<d", float("nan"))]
    for blob in bad:
        with pytest.raises(ValueError):
            serialize.loads(blob)


_HASHSEED_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
from genesis.nn import TinyGPT, serialize, train_lm, backend
assert backend.backend_name() == "python"
m = TinyGPT(vocab_size=7, block_size=6, dim=8, n_layers=1, n_heads=2, seed=3)
train_lm(m, [i * 3 % 7 for i in range(40)], steps=4, batch_size=2, lr=1e-2, seed=1)
print(serialize.digest(serialize.dumps(m)))
"""


def test_determinism_independent_of_hash_seed(py_backend):
    script = _HASHSEED_SCRIPT.format(root=ROOT)
    procs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs, GENESIS_NN_BACKEND="python")
        procs.append(subprocess.Popen([sys.executable, "-c", script], env=env, cwd=ROOT,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        outs.append(out.strip())
    m = TinyGPT(vocab_size=7, block_size=6, dim=8, n_layers=1, n_heads=2, seed=3)
    train_lm(m, [i * 3 % 7 for i in range(40)], steps=4, batch_size=2, lr=1e-2, seed=1)
    assert outs[0] == outs[1] == serialize.digest(serialize.dumps(m))
