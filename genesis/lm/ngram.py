"""N-gram language models over token ids, estimated from local data only.

:class:`NGramLM` is an interpolated **modified Kneser–Ney** model (Chen & Goodman 1998, "An Empirical Study of
Smoothing Techniques for Language Modeling", Harvard TR-10-98)::

    P_n(w | h) = (a(h, w) - D_n(a(h, w))) / S(h) + gamma(h) * P_{n-1}(w | h')
    gamma(h)   = (D_n1 * N1(h.) + D_n2 * N2(h.) + D_n3+ * N3+(h.)) / S(h)

where ``h'`` drops the oldest token of ``h``, ``S(h) = sum_w a(h, w)`` and ``N_k(h.)`` counts the words with
``a(h, w) == k`` (``>= 3`` for ``N3+``). The highest order uses raw counts ``a = c``; lower orders use
continuation counts ``a = N1+(. h w)`` (number of distinct left extensions), except n-grams starting with BOS,
which cannot be extended to the left and keep raw counts (as in KenLM). Discounts are estimated per order
from the count-of-counts ``n_k``: ``Y = n1 / (n1 + 2 n2)``, ``D1 = 1 - 2Y n2/n1``, ``D2 = 2 - 3Y n3/n2``,
``D3+ = 3 - 4Y n4/n3``; a value that is undefined or outside ``(0, k)`` falls back to ``0.5 * k``. Unseen
contexts back off to the lower order, and the recursion ends with the uniform distribution over the
vocabulary, so every id in ``[0, vocab_size)`` has nonzero probability.

Sequences are left-padded with ``order - 1`` BOS ids and EOS is appended as a prediction target. A context
shorter than ``order - 1`` is therefore read as the start of a sequence and padded with BOS the same way.

:class:`UnigramLM` (add-one smoothing) is a baseline with the same interface.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Iterable, Sequence

FORMAT = "genesis-ngram"
VERSION = 1


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_vocab(vocab_size, bos_id, eos_id) -> tuple[int, int, int]:
    if not _is_int(vocab_size) or vocab_size < 2:
        raise ValueError("vocab_size must be an int >= 2")
    for name, x in (("bos_id", bos_id), ("eos_id", eos_id)):
        if not _is_int(x) or not 0 <= x < vocab_size:
            raise ValueError(f"{name} must be an int in [0, vocab_size)")
    if bos_id == eos_id:
        raise ValueError("bos_id and eos_id must differ")
    return vocab_size, bos_id, eos_id


def _check_ids(ids, vocab_size: int, what: str) -> list[int]:
    if isinstance(ids, (str, bytes)):
        raise ValueError(f"{what} must be a sequence of token ids")
    out = list(ids)
    for t in out:
        if not _is_int(t) or not 0 <= t < vocab_size:
            raise ValueError(f"{what} contains an invalid token id: {t!r}")
    return out


def _check_sequences(sequences, vocab_size: int, bos_id: int) -> list[list[int]]:
    if isinstance(sequences, (str, bytes)):
        raise ValueError("sequences must be a list of token-id lists")
    seqs = [_check_ids(s, vocab_size, "sequence") for s in sequences]
    if not seqs:
        raise ValueError("need at least one sequence")
    for s in seqs:
        if bos_id in s:
            raise ValueError("sequences must not contain bos_id (it is added as padding)")
    return seqs


def _canonical_digest(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _parse_pairs(pairs, vocab_size: int, bos_id: int, where: str) -> dict[int, int]:
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{where}: expected a non-empty list of [token, count] pairs")
    out: dict[int, int] = {}
    for item in pairs:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{where}: malformed [token, count] pair {item!r}")
        tok, c = item
        if not _is_int(tok) or not 0 <= tok < vocab_size or tok == bos_id:
            raise ValueError(f"{where}: invalid token {tok!r}")
        if not _is_int(c) or c < 1:
            raise ValueError(f"{where}: invalid count {c!r}")
        if tok in out:
            raise ValueError(f"{where}: duplicate token {tok}")
        out[tok] = c
    return out


def _check_header(d, model: str) -> None:
    if not isinstance(d, dict):
        raise ValueError("model data must be a dict")
    if d.get("format") != FORMAT or d.get("version") != VERSION or d.get("model") != model:
        raise ValueError(f"unsupported model data (expected {FORMAT} v{VERSION}, model {model!r})")


class _BaseLM:
    """Shared scoring/sampling code; subclasses implement ``_key``, ``_prob`` and ``_dist``."""

    _pad = 0

    def __init__(self) -> None:
        self.vocab_size = 0
        self.bos_id = -1
        self.eos_id = -1

    def _check_fitted(self) -> None:
        if not self.vocab_size:
            raise RuntimeError("model is not fitted")

    def _key(self, context: Sequence[int]) -> tuple[int, ...]:
        raise NotImplementedError

    def _prob(self, token: int, key: tuple[int, ...]) -> float:
        raise NotImplementedError

    def _dist(self, key: tuple[int, ...]) -> list[float]:
        raise NotImplementedError

    def _check_token(self, token) -> int:
        if not _is_int(token) or not 0 <= token < self.vocab_size:
            raise ValueError(f"invalid token id: {token!r}")
        return token

    def prob(self, token: int, context: Sequence[int] = ()) -> float:
        self._check_fitted()
        self._check_token(token)
        return self._prob(token, self._key(_check_ids(context, self.vocab_size, "context")))

    def logprob(self, token: int, context: Sequence[int] = ()) -> float:
        return math.log(self.prob(token, context))

    def distribution(self, context: Sequence[int] = ()) -> list[float]:
        """Probabilities of every id in ``[0, vocab_size)`` after ``context`` (sums to 1)."""
        self._check_fitted()
        return self._dist(self._key(_check_ids(context, self.vocab_size, "context")))

    def perplexity(self, sequences: Iterable[Sequence[int]]) -> float:
        """exp of the mean negative log-probability per predicted token, EOS predictions included."""
        self._check_fitted()
        seqs = _check_sequences(sequences, self.vocab_size, self.bos_id)
        k = self._pad
        total, n = 0.0, 0
        for seq in seqs:
            s = [self.bos_id] * k + seq + [self.eos_id]
            for i in range(k, len(s)):
                total += math.log(self._prob(s[i], tuple(s[i - k:i])))
                n += 1
        return math.exp(-total / n)

    def generate(self, context_ids: Sequence[int], max_new: int, rng: random.Random,
                 temperature: float = 1.0, top_k: int | None = None,
                 stop_ids: Iterable[int] = ()) -> list[int]:
        """Sample up to ``max_new`` ids after ``context_ids``; returns only the new ids.

        ``temperature == 0`` is greedy (ties -> smallest id). BOS is never produced. Generation stops after
        emitting an id in ``stop_ids`` (that id is included in the result).
        """
        self._check_fitted()
        history = _check_ids(context_ids, self.vocab_size, "context_ids")
        if not _is_int(max_new) or max_new < 0:
            raise ValueError("max_new must be an int >= 0")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite number >= 0")
        if top_k is not None and (not _is_int(top_k) or top_k < 1):
            raise ValueError("top_k must be None or an int >= 1")
        stops = set(_check_ids(stop_ids, self.vocab_size, "stop_ids"))
        out: list[int] = []
        for _ in range(max_new):
            p = self._dist(self._key(history))
            cands = [i for i in range(self.vocab_size) if i != self.bos_id]
            if top_k is not None and top_k < len(cands):
                cands = sorted(sorted(cands, key=lambda i: (-p[i], i))[:top_k])
            if temperature == 0:
                tok = min(cands, key=lambda i: (-p[i], i))
            else:
                if temperature == 1:
                    weights = [p[i] for i in cands]
                else:
                    logs = [math.log(p[i]) if p[i] > 0 else -math.inf for i in cands]
                    top = max(logs)
                    weights = [math.exp((lp - top) / temperature) for lp in logs]
                r = rng.random() * math.fsum(weights)
                tok = cands[-1]
                acc = 0.0
                for i, w in zip(cands, weights):
                    acc += w
                    if r < acc:
                        tok = i
                        break
            out.append(tok)
            history.append(tok)
            if tok in stops:
                break
        return out

    def to_dict(self) -> dict:
        raise NotImplementedError

    def digest(self) -> str:
        """sha256 of the canonical JSON of :meth:`to_dict`."""
        return _canonical_digest(self.to_dict())


class NGramLM(_BaseLM):
    """Interpolated modified Kneser–Ney n-gram model (see module docstring)."""

    def __init__(self, order: int = 3):
        super().__init__()
        if not _is_int(order) or order < 1:
            raise ValueError("order must be an int >= 1")
        self.order = order
        self._pad = order - 1
        # Index n-1 holds order n: {context (n-1 ids): {token: count}}, raw at the top, adjusted below.
        self._counts: list[dict[tuple[int, ...], dict[int, int]]] = []
        self._stats: list[dict[tuple[int, ...], tuple[int, float]]] = []  # context -> (S(h), gamma(h) * S(h))
        self._discounts: list[tuple[float, float, float]] = []

    def fit(self, sequences: Iterable[Sequence[int]], vocab_size: int, bos_id: int, eos_id: int) -> "NGramLM":
        vocab_size, bos_id, eos_id = _check_vocab(vocab_size, bos_id, eos_id)
        seqs = _check_sequences(sequences, vocab_size, bos_id)
        k = self._pad
        top: dict[tuple[int, ...], dict[int, int]] = {}
        for seq in seqs:
            s = [bos_id] * k + seq + [eos_id]
            for i in range(k, len(s)):
                h = tuple(s[i - k:i])
                d = top.get(h)
                if d is None:
                    top[h] = d = {}
                w = s[i]
                d[w] = d.get(w, 0) + 1
        self._build(top, vocab_size, bos_id, eos_id)
        return self

    def _build(self, top, vocab_size: int, bos_id: int, eos_id: int) -> None:
        n_orders = self.order
        counts: list = [None] * n_orders
        counts[n_orders - 1] = top
        for n in range(n_orders - 1, 0, -1):
            lower: dict[tuple[int, ...], dict[int, int]] = {}
            for h, dist in counts[n].items():
                hl = h[1:]
                bos_start = bool(hl) and hl[0] == bos_id
                d = lower.get(hl)
                if d is None:
                    lower[hl] = d = {}
                for w, a in dist.items():
                    d[w] = d.get(w, 0) + (a if bos_start else 1)
            counts[n - 1] = lower
        discounts, stats = [], []
        for table in counts:
            disc = self._estimate_discounts(table)
            d1, d2, d3 = disc
            st: dict[tuple[int, ...], tuple[int, float]] = {}
            for h, dist in table.items():
                total = n1 = n2 = n3 = 0
                for c in dist.values():
                    total += c
                    if c == 1:
                        n1 += 1
                    elif c == 2:
                        n2 += 1
                    else:
                        n3 += 1
                st[h] = (total, d1 * n1 + d2 * n2 + d3 * n3)
            discounts.append(disc)
            stats.append(st)
        self.vocab_size, self.bos_id, self.eos_id = vocab_size, bos_id, eos_id
        self._counts, self._stats, self._discounts = counts, stats, discounts

    @staticmethod
    def _estimate_discounts(table) -> tuple[float, float, float]:
        nk = [0] * 5
        for dist in table.values():
            for c in dist.values():
                if c <= 4:
                    nk[c] += 1
        _, n1, n2, n3, n4 = nk
        y = n1 / (n1 + 2 * n2) if n1 + 2 * n2 else 0.0
        raw = (1 - 2 * y * n2 / n1 if n1 else None,
               2 - 3 * y * n3 / n2 if n2 else None,
               3 - 4 * y * n4 / n3 if n3 else None)
        return tuple(d if d is not None and 0 < d < k else 0.5 * k for k, d in enumerate(raw, start=1))

    def _key(self, context: Sequence[int]) -> tuple[int, ...]:
        k = self._pad
        if k == 0:
            return ()
        ctx = tuple(context[-k:])
        if len(ctx) < k:
            ctx = (self.bos_id,) * (k - len(ctx)) + ctx
        return ctx

    def _prob(self, token: int, key: tuple[int, ...]) -> float:
        p = 1.0 / self.vocab_size
        k = self._pad
        for n in range(1, self.order + 1):
            h = key[k - n + 1:]
            st = self._stats[n - 1].get(h)
            if st is None:
                break  # unseen at order n => unseen at every higher order
            total, mass = st
            c = self._counts[n - 1][h].get(token, 0)
            d1, d2, d3 = self._discounts[n - 1]
            num = c - (d1 if c == 1 else d2 if c == 2 else d3) if c else 0.0
            p = (num + mass * p) / total
        return p

    def _dist(self, key: tuple[int, ...]) -> list[float]:
        p = [1.0 / self.vocab_size] * self.vocab_size
        k = self._pad
        for n in range(1, self.order + 1):
            h = key[k - n + 1:]
            st = self._stats[n - 1].get(h)
            if st is None:
                break
            total, mass = st
            d1, d2, d3 = self._discounts[n - 1]
            q = [mass * x for x in p]
            for w, c in self._counts[n - 1][h].items():
                q[w] = (c - (d1 if c == 1 else d2 if c == 2 else d3)) + q[w]
            p = [x / total for x in q]
        return p

    @property
    def discounts(self) -> list[tuple[float, float, float]]:
        """(D1, D2, D3+) for orders 1..order."""
        return list(self._discounts)

    @property
    def num_ngrams(self) -> list[int]:
        """Number of distinct n-grams for orders 1..order."""
        return [sum(len(d) for d in table.values()) for table in self._counts]

    def to_dict(self) -> dict:
        self._check_fitted()
        top = self._counts[-1]
        return {
            "format": FORMAT,
            "version": VERSION,
            "model": "kneser-ney",
            "order": self.order,
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "counts": {",".join(map(str, h)): [[w, c] for w, c in sorted(top[h].items())] for h in sorted(top)},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NGramLM":
        _check_header(d, "kneser-ney")
        order = d.get("order")
        if not _is_int(order) or order < 1:
            raise ValueError("invalid order")
        vocab_size, bos_id, eos_id = _check_vocab(d.get("vocab_size"), d.get("bos_id"), d.get("eos_id"))
        raw = d.get("counts")
        if not isinstance(raw, dict) or not raw:
            raise ValueError("'counts' must be a non-empty dict")
        top: dict[tuple[int, ...], dict[int, int]] = {}
        for key, pairs in raw.items():
            parts = key.split(",") if isinstance(key, str) and key else []
            if not isinstance(key, str) or len(parts) != order - 1 \
                    or not all(p.isascii() and p.isdigit() for p in parts):
                raise ValueError(f"invalid context key {key!r} for order {order}")
            h = tuple(int(p) for p in parts)
            if any(t >= vocab_size for t in h) or h in top:
                raise ValueError(f"invalid or duplicate context key {key!r}")
            top[h] = _parse_pairs(pairs, vocab_size, bos_id, f"context {key!r}")
        model = cls(order)
        model._build(top, vocab_size, bos_id, eos_id)
        return model

    def __repr__(self) -> str:
        return f"NGramLM(order={self.order}, vocab_size={self.vocab_size}, ngrams={self.num_ngrams if self.vocab_size else []})"


class UnigramLM(_BaseLM):
    """Add-one smoothed unigram baseline: ``P(w) = (c(w) + 1) / (N + V)``, context ignored."""

    def __init__(self) -> None:
        super().__init__()
        self._counts: list[int] = []
        self._total = 0

    def fit(self, sequences: Iterable[Sequence[int]], vocab_size: int, bos_id: int, eos_id: int) -> "UnigramLM":
        vocab_size, bos_id, eos_id = _check_vocab(vocab_size, bos_id, eos_id)
        seqs = _check_sequences(sequences, vocab_size, bos_id)
        counts = [0] * vocab_size
        for seq in seqs:
            for t in seq:
                counts[t] += 1
            counts[eos_id] += 1
        self._set(counts, vocab_size, bos_id, eos_id)
        return self

    def _set(self, counts: list[int], vocab_size: int, bos_id: int, eos_id: int) -> None:
        self.vocab_size, self.bos_id, self.eos_id = vocab_size, bos_id, eos_id
        self._counts = counts
        self._total = sum(counts)

    def _key(self, context: Sequence[int]) -> tuple[int, ...]:
        return ()

    def _prob(self, token: int, key: tuple[int, ...]) -> float:
        return (self._counts[token] + 1) / (self._total + self.vocab_size)

    def _dist(self, key: tuple[int, ...]) -> list[float]:
        z = self._total + self.vocab_size
        return [(c + 1) / z for c in self._counts]

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "format": FORMAT,
            "version": VERSION,
            "model": "unigram-add-one",
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "counts": [[w, c] for w, c in enumerate(self._counts) if c],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnigramLM":
        _check_header(d, "unigram-add-one")
        vocab_size, bos_id, eos_id = _check_vocab(d.get("vocab_size"), d.get("bos_id"), d.get("eos_id"))
        pairs = _parse_pairs(d.get("counts"), vocab_size, bos_id, "counts")
        counts = [0] * vocab_size
        for w, c in pairs.items():
            counts[w] = c
        model = cls()
        model._set(counts, vocab_size, bos_id, eos_id)
        return model

    def __repr__(self) -> str:
        return f"UnigramLM(vocab_size={self.vocab_size}, tokens={self._total})"
