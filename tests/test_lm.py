"""Tests for genesis.lm: byte-level BPE tokenizer and n-gram language models.

All training text is generated here from a tiny made-up grammar (no external data).
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from genesis.lm import BPETokenizer, NGramLM, UnigramLM, pretokenize

REPO_ROOT = Path(__file__).resolve().parents[1]

MASC = ["brivon", "zorblat", "kitovar", "grelot", "vintak", "pommaré", "fàsil", "trémoul"]
FEM = ["tazelle", "mélorine", "çaprine", "ündelle", "suloé", "brindâme", "quavette", "lirène"]
ADJ = [("vert", "verte"), ("rapide", "rapide"), ("ombré", "ombrée"), ("calme", "calme"), ("zinté", "zintée")]
VERB_FR = ["mange", "voit", "suit", "chérit", "dévore", "porte"]
ADV_FR = ["souvent", "toujours", "doucement", "jamais"]
NOUN_EN = ["florp", "grindle", "wuzzle", "snorkap", "blevin", "quiddet"]
VERB_EN = ["grabs", "lifts", "follows", "sees", "eats"]
ADJ_EN = ["brisk", "shiny", "fierce", "quiet"]


def _np(rng: random.Random, start: bool = False) -> str:
    fem = rng.random() < 0.5
    noun = rng.choice(FEM if fem else MASC)
    det = rng.choice(["la", "une"] if fem else ["le", "un"])
    if start:
        det = det.capitalize()
    if rng.random() < 0.4:
        return f"{det} {noun} {rng.choice(ADJ)[1 if fem else 0]}"
    return f"{det} {noun}"


def make_sentences(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            s = f"{_np(rng, True)} {rng.choice(VERB_FR)} {_np(rng)}."
        elif r < 0.6:
            s = f"{_np(rng, True)} {rng.choice(VERB_FR)} {rng.choice(ADV_FR)} !"
        elif r < 0.7:
            s = f"Il y a {rng.randint(2, 999)} {rng.choice(MASC)}s près de {_np(rng)}."
        else:
            s = (f"The {rng.choice(ADJ_EN)} {rng.choice(NOUN_EN)} {rng.choice(VERB_EN)} "
                 f"a {rng.choice(NOUN_EN)}" + rng.choice([".", ".", "?", " 🙂"]))
        out.append(s)
    return out


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    return make_sentences(1500, seed=1)


@pytest.fixture(scope="module")
def tok(corpus) -> BPETokenizer:
    return BPETokenizer.train(corpus, vocab_size=420)


# ---------------------------------------------------------------------------------------------- BPE

ROUND_TRIP = [
    "",
    "Le brivon vert mange la tazelle.",
    "Élève, garçon, où est l'œuvre ? À bientôt — déjà vu !",
    "emoji 🙂🚀👩‍💻 and flags 🇫🇷",
    "中文字符和日本語のテキスト、한국어",
    "tabs\tand\nnew\r\nlines   with  spaces \n\n",
    "digits 1234567 3.14159 -42 0x1F",
    "snake_case __dunder__ _x_ a__b",
    "<|eos|> <|bos|><|pad|> literal specials",
    " leading and trailing ",
    "\u0301combining e\u0301 and ½ ² Ⅻ",
    "zero\u200bwidth\u00a0nbsp\x00nul",
]


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_bpe_round_trip(tok, text):
    assert "".join(pretokenize(text)) == text
    ids = tok.encode(text)
    assert all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text
    assert tok.decode(tok.encode(text, add_bos=True, add_eos=True)) == text


def test_bpe_round_trip_random_unicode(tok):
    rng = random.Random(5)
    for _ in range(200):
        chars = [chr(rng.choice([rng.randint(0, 0x7F), rng.randint(0x80, 0xD7FF), rng.randint(0xE000, 0x10FFFF)]))
                 for _ in range(rng.randint(0, 30))]
        text = "".join(chars)
        assert tok.decode(tok.encode(text)) == text


def test_bpe_untrained_is_byte_level():
    t = BPETokenizer()
    assert t.vocab_size == 259
    assert t.encode("é") == list("é".encode("utf-8"))


def test_literal_special_text_never_yields_special_ids(tok):
    specials = {tok.bos_id, tok.eos_id, tok.pad_id}
    trained = BPETokenizer.train(["<|eos|> <|bos|> <|pad|>"] * 50 + ["a <|eos|>"] * 50, vocab_size=300)
    for t in (tok, trained):
        ids = t.encode("hello <|eos|><|bos|> <|pad|>")
        assert not specials & set(ids)
        assert t.decode(ids) == "hello <|eos|><|bos|> <|pad|>"
    ids = tok.encode("x", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "x"
    assert tok.decode(ids, skip_special=False) == "<|bos|>x<|eos|>"


def test_bpe_training_is_deterministic(corpus):
    a = BPETokenizer.train(corpus, vocab_size=380)
    b = BPETokenizer.train(list(corpus), vocab_size=380)
    assert a.digest() == b.digest()
    assert a.merges == b.merges


def _merge_all(sym, a, b, new):
    out, i = [], 0
    while i < len(sym):
        if i + 1 < len(sym) and sym[i] == a and sym[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out


def test_bpe_matches_naive_reference():
    """Incremental training == recounting every pair at each step; encoding == replaying merges in order."""
    rng = random.Random(3)
    texts = ["".join(rng.choice("aab ba_é1 \n") for _ in range(rng.randint(0, 40))) for _ in range(150)]
    texts += make_sentences(100, seed=4)
    tok = BPETokenizer.train(texts, vocab_size=330, min_frequency=3)

    words = []
    for t in texts:
        words += [list(p.encode("utf-8")) for p in pretokenize(t)]
    tb = [bytes([i]) for i in range(256)] + [b""] * 3
    merges = []
    while len(tb) < 330:
        counts: dict = {}
        for w in words:
            for p in zip(w, w[1:]):
                counts[p] = counts.get(p, 0) + 1
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], tb[p[0]], tb[p[1]], p))
        if counts[best] < 3:
            break
        merges.append(best)
        tb.append(tb[best[0]] + tb[best[1]])
        words = [_merge_all(w, best[0], best[1], len(tb) - 1) for w in words]
    assert tok.merges == merges

    for t in texts[:60] + ROUND_TRIP:
        expected = []
        for piece in pretokenize(t):
            sym = list(piece.encode("utf-8"))
            for r, (a, b) in enumerate(merges):
                sym = _merge_all(sym, a, b, 259 + r)
            expected += sym
        assert tok.encode(t) == expected


TRAIN_SNIPPET = textwrap.dedent("""
    import random
    from genesis.lm import BPETokenizer
    rng = random.Random(7)
    words = ["brivon", "tazelle", "mélor", "çapri", "zorblat", "kito", "ünd", "92", "fà", "a_b", "🙂"]
    texts = [" ".join(rng.choice(words) for _ in range(12)) + rng.choice([".", "!", "?"]) for _ in range(300)]
    tok = BPETokenizer.train(texts, vocab_size=330)
""")


def test_bpe_digest_independent_of_hash_seed():
    ns: dict = {}
    exec(TRAIN_SNIPPET, ns)
    expected = ns["tok"].digest()
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", TRAIN_SNIPPET + "print(tok.digest())"],
                             cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == expected


def test_vocab_size_and_frequent_word(corpus):
    t = BPETokenizer.train(corpus + [" zorblat"] * 500, vocab_size=400)
    assert t.vocab_size == 400
    assert len(t.encode(" zorblat")) == 1
    assert t.token_str(t.encode(" zorblat")[0]) == " zorblat"
    small = BPETokenizer.train(["ab ab ab"], vocab_size=1000)
    assert 259 < small.vocab_size < 1000  # stopped by min_frequency
    with pytest.raises(ValueError):
        BPETokenizer.train(corpus, vocab_size=258)
    with pytest.raises(ValueError):
        BPETokenizer.train(corpus, vocab_size=300, special_tokens=("<|x|>", "<|x|>"))


def test_bpe_compresses(tok, corpus):
    text = "\n".join(corpus[:200])
    assert len(text.encode("utf-8")) / len(tok.encode(text)) > 2.5


def test_bpe_serialization_round_trip(tok, corpus):
    d = json.loads(json.dumps(tok.to_dict()))
    t2 = BPETokenizer.from_dict(d)
    assert t2.digest() == tok.digest()
    assert t2.vocab_size == tok.vocab_size
    for text in corpus[:50] + ROUND_TRIP:
        assert t2.encode(text) == tok.encode(text)
    assert [t2.token_bytes(i) for i in range(t2.vocab_size)] == [tok.token_bytes(i) for i in range(tok.vocab_size)]


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(version=99),
    lambda d: d.update(format="other"),
    lambda d: d.pop("merges"),
    lambda d: d.update(merges="nope"),
    lambda d: d["merges"].append([1]),
    lambda d: d["merges"].append([1, 10_000]),
    lambda d: d["merges"].append([256, 97]),        # special ids cannot be merged
    lambda d: d["merges"].append(list(d["merges"][0])),  # duplicate
    lambda d: d["merges"].append([True, 1]),
    lambda d: d.update(specials=[1, 2]),
    lambda d: d.update(specials=["<|a|>", "<|a|>"]),
    lambda d: d.update(pattern="."),
])
def test_bpe_from_dict_rejects_malformed(tok, mutate):
    d = json.loads(json.dumps(tok.to_dict()))
    mutate(d)
    with pytest.raises(ValueError):
        BPETokenizer.from_dict(d)
    with pytest.raises(ValueError):
        BPETokenizer.from_dict(["not", "a", "dict"])


def test_bpe_invalid_ids(tok):
    for bad in (-1, tok.vocab_size, "3", 1.0, None):
        with pytest.raises(ValueError):
            tok.decode([bad])
    with pytest.raises(ValueError):
        tok.token_bytes(tok.vocab_size)


# ---------------------------------------------------------------------------------------------- n-gram

@pytest.fixture(scope="module")
def encoded(tok, corpus):
    train = [tok.encode(s) for s in corpus]
    held_out = [tok.encode(s) for s in make_sentences(300, seed=2)]
    return train, held_out


@pytest.fixture(scope="module")
def kn(tok, encoded):
    return NGramLM(order=3).fit(encoded[0], tok.vocab_size, tok.bos_id, tok.eos_id)


def test_distribution_sums_to_one(kn, tok, encoded):
    seen = encoded[0][0][:2]
    unseen = [tok.encode("🚀")[0], 7]
    for ctx in ([], seen, encoded[0][3][:5], unseen, [tok.eos_id, tok.eos_id]):
        dist = kn.distribution(ctx)
        assert len(dist) == tok.vocab_size
        assert abs(math.fsum(dist) - 1.0) < 1e-9
        assert min(dist) > 0
        for t in (0, 255, tok.pad_id, tok.vocab_size - 1):
            assert kn.prob(t, ctx) == dist[t]
            assert kn.logprob(t, ctx) == pytest.approx(math.log(dist[t]))
    for order in (1, 2, 4):
        m = NGramLM(order).fit(encoded[0][:200], tok.vocab_size, tok.bos_id, tok.eos_id)
        assert abs(math.fsum(m.distribution(seen)) - 1.0) < 1e-9


def test_kneser_ney_beats_unigram(kn, tok, encoded):
    train, held_out = encoded
    uni = UnigramLM().fit(train, tok.vocab_size, tok.bos_id, tok.eos_id)
    assert abs(math.fsum(uni.distribution()) - 1.0) < 1e-9
    pp_kn, pp_uni = kn.perplexity(held_out), uni.perplexity(held_out)
    assert pp_kn < 0.5 * pp_uni
    bigram = NGramLM(2).fit(train, tok.vocab_size, tok.bos_id, tok.eos_id)
    assert pp_kn < bigram.perplexity(held_out) < pp_uni
    for k, (d1, d2, d3) in enumerate(kn.discounts):
        assert 0 < d1 < 1 and 0 < d2 < 2 and 0 < d3 < 3, k


def test_perplexity_includes_eos():
    m = NGramLM(2).fit([[2, 3], [2, 4]], 6, 0, 1)
    seq = [2, 3]
    lp = math.log(m.prob(2, [])) + math.log(m.prob(3, [2])) + math.log(m.prob(1, [2, 3]))
    assert m.perplexity([seq]) == pytest.approx(math.exp(-lp / 3))


def test_kneser_ney_matches_reference_formula():
    """Compare with a direct transcription of Chen & Goodman's interpolated modified KN."""
    rng = random.Random(8)
    V, BOS, EOS, N = 9, 0, 1, 3
    seqs = [[rng.choice([2, 3, 3, 4, 5, 5, 5, 6, 7, 8]) for _ in range(rng.randint(0, 12))] for _ in range(60)]
    model = NGramLM(N).fit(seqs, V, BOS, EOS)

    padded = [[BOS] * (N - 1) + s + [EOS] for s in seqs]
    adj = {}
    for n in range(1, N + 1):
        raw, left = {}, {}
        for s in padded:
            for i in range(N - 1, len(s)):
                g = tuple(s[i - n + 1:i + 1])
                raw[g] = raw.get(g, 0) + 1
                left.setdefault(g, set()).add(s[i - n] if n < N else None)
        adj[n] = {g: (raw[g] if n == N or g[0] == BOS else len(left[g])) for g in raw}

    def discounts(n):
        nk = [sum(1 for c in adj[n].values() if c == k) for k in range(5)]
        formulas = [lambda y: 1 - 2 * y * nk[2] / nk[1], lambda y: 2 - 3 * y * nk[3] / nk[2],
                    lambda y: 3 - 4 * y * nk[4] / nk[3]]
        out = []
        for k, f in enumerate(formulas, 1):
            try:
                d = f(nk[1] / (nk[1] + 2 * nk[2]))
            except ZeroDivisionError:
                d = 0.0
            out.append(d if 0 < d < k else 0.5 * k)
        return out

    def p_ref(w, h):
        n = len(h) + 1
        lower = p_ref(w, h[1:]) if h else 1 / V
        entries = {g[-1]: c for g, c in adj[n].items() if g[:-1] == h}
        if not entries:
            return lower
        d = discounts(n)
        disc = lambda c: d[min(c, 3) - 1]
        total = sum(entries.values())
        gamma = sum(disc(c) for c in entries.values()) / total
        c = entries.get(w, 0)
        return (c - disc(c) if c else 0) / total + gamma * lower

    for h in [(BOS, BOS), (BOS, 3), (3, 5), (5, 5), (8, 2), (6, 6), (1, 1)]:
        for w in range(V):
            assert model.prob(w, list(h)) == pytest.approx(p_ref(w, h), rel=1e-12)


def test_top_prediction_on_toy_sequence():
    cycle = [5, 6, 7, 8, 9]
    seqs = [cycle * 4 for _ in range(3)]
    m = NGramLM(3).fit(seqs, 12, 0, 1)

    def top(ctx):
        d = m.distribution(ctx)
        return max(range(len(d)), key=lambda i: (d[i], -i))

    assert top([]) == 5
    for i in range(len(cycle)):
        ctx = [cycle[i], cycle[(i + 1) % 5]]
        assert top(ctx) == cycle[(i + 2) % 5]
    m2 = NGramLM(3).fit([[5, 6, 7]] * 4, 12, 0, 1)
    assert max(range(12), key=lambda i: (m2.prob(i, [6, 7]), -i)) == 1  # EOS after 6 7


def test_generate_deterministic_and_greedy(kn, tok):
    a = kn.generate([], 30, random.Random(42), temperature=0.8, top_k=20, stop_ids=[tok.eos_id])
    b = kn.generate([], 30, random.Random(42), temperature=0.8, top_k=20, stop_ids=[tok.eos_id])
    assert a == b and a
    assert tok.bos_id not in a
    assert kn.generate([], 5, random.Random(1), temperature=1.0) == kn.generate([], 5, random.Random(1))

    pattern = [10, 11, 12, 13, 14, 15]
    m = NGramLM(3).fit([pattern] * 5 + [[10, 11, 20]], 30, 0, 1)
    out = m.generate([], 20, random.Random(0), temperature=0, stop_ids=[1])
    assert out == pattern + [1]
    assert m.generate([12, 13], 3, random.Random(0), temperature=0) == [14, 15, 1]
    assert m.generate([], 0, random.Random(0)) == []
    with pytest.raises(ValueError):
        m.generate([], 3, random.Random(0), temperature=-1)
    with pytest.raises(ValueError):
        m.generate([99], 3, random.Random(0))


def test_ngram_serialization_round_trip(kn, encoded):
    d = json.loads(json.dumps(kn.to_dict()))
    assert all(isinstance(k, str) for k in d["counts"])
    m2 = NGramLM.from_dict(d)
    assert m2.digest() == kn.digest()
    for ctx in ([], encoded[1][0][:2], encoded[1][4][:7], [3, 4]):
        assert m2.distribution(ctx) == kn.distribution(ctx)
    assert m2.perplexity(encoded[1][:20]) == kn.perplexity(encoded[1][:20])
    assert m2.discounts == kn.discounts

    uni = UnigramLM().fit(encoded[0], kn.vocab_size, kn.bos_id, kn.eos_id)
    u2 = UnigramLM.from_dict(json.loads(json.dumps(uni.to_dict())))
    assert u2.digest() == uni.digest() and u2.distribution() == uni.distribution()


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(version=2),
    lambda d: d.update(model="unigram-add-one"),
    lambda d: d.update(order=0),
    lambda d: d.update(bos_id=d["eos_id"]),
    lambda d: d.update(counts={}),
    lambda d: d["counts"].update({"1": [[2, 1]]}),          # wrong context length
    lambda d: d["counts"].update({"1,x": [[2, 1]]}),
    lambda d: d["counts"].update({"1,99999": [[2, 1]]}),
    lambda d: d["counts"].update({"1,2": [[2, 0]]}),
    lambda d: d["counts"].update({"1,2": [[99999, 1]]}),
    lambda d: d["counts"].update({"1,2": [[2, 1], [2, 3]]}),
    lambda d: d["counts"].update({"1,2": [[d["bos_id"], 1]]}),
])
def test_ngram_from_dict_rejects_malformed(kn, mutate):
    d = json.loads(json.dumps(kn.to_dict()))
    mutate(d)
    with pytest.raises(ValueError):
        NGramLM.from_dict(d)


def test_ngram_input_validation():
    m = NGramLM(3)
    with pytest.raises(RuntimeError):
        m.prob(1, [])
    with pytest.raises(ValueError):
        m.fit([[2, 3, 7]], 6, 0, 1)
    with pytest.raises(ValueError):
        m.fit([[2, 0, 3]], 6, 0, 1)  # BOS inside a sequence
    with pytest.raises(ValueError):
        m.fit([], 6, 0, 1)
    with pytest.raises(ValueError):
        NGramLM(0)
    m.fit([[2, 3]], 6, 0, 1)
    with pytest.raises(ValueError):
        m.prob(6, [])
    with pytest.raises(ValueError):
        m.distribution([-1])
