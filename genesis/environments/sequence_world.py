"""Stage-1 world: several integer streams, each produced by a hidden generator.

Generators are sampled at runtime from a seed, so no answer can be stored in
the agent's code. Some streams are pure noise (no rule exists) to test whether
the agent can say "I found no pattern" instead of hallucinating one.

Optional harder settings (defaults leave the original world unchanged):

* ``noise``: probability that a rule stream emits a corrupted term. The corruption is an *innovation*: the
  generator continues from the corrupted value, so the rule stays exact on every other term.
* ``families``: generator families to sample from, e.g. :data:`DEEP_FAMILIES`, whose shortest rules have
  seven nodes (beyond the default size-5 hypothesis space).
"""

from __future__ import annotations

import random
import zlib


def _arithmetic(rng):
    c = rng.choice([v for v in range(-9, 10) if v != 0])
    return {"family": "arithmetic", "start": [rng.randint(-20, 20)], "step": lambda h, t: h[-1] + c}


def _geometric(rng):
    k = rng.choice([2, 3, -2])
    return {"family": "geometric", "start": [rng.randint(1, 4)], "step": lambda h, t: h[-1] * k}


def _fibonacci_like(rng):
    return {"family": "additive-recurrence", "start": [rng.randint(0, 5), rng.randint(1, 5)],
            "step": lambda h, t: h[-1] + h[-2]}


def _periodic(rng):
    p = rng.choice([2, 3])
    vals = rng.sample(range(0, 30), p)
    return {"family": f"periodic-{p}", "start": vals, "step": lambda h, t: h[-p]}


def _quadratic(rng):
    c = rng.randint(-10, 10)
    return {"family": "quadratic-index", "start": [c], "step": lambda h, t: t * t + c}


def _counter_mod(rng):
    m = rng.randint(5, 12)
    c = rng.randint(1, m - 1)
    return {"family": "modular-counter", "start": [rng.randint(0, m - 1)], "step": lambda h, t: (h[-1] + c) % m}


def _alternating(rng):
    c = rng.choice([v for v in range(-6, 7) if v != 0])
    return {"family": "lag2-arithmetic", "start": [rng.randint(0, 20), rng.randint(0, 20)],
            "step": lambda h, t: h[-2] + c}


def _noise(rng):
    lo = rng.randint(0, 10)
    return {"family": "noise", "start": [rng.randint(lo, lo + 60)],
            "step": lambda h, t, r=random.Random(rng.random()): r.randint(lo, lo + 60)}


RULE_FAMILIES = [_arithmetic, _geometric, _fibonacci_like, _periodic, _quadratic, _counter_mod, _alternating]


def _lcg(rng):
    m = rng.randint(7, 13)
    a, c = rng.choice([2, 3]), rng.randint(1, m - 1)
    return {"family": "affine-modular", "start": [rng.randint(0, m - 1)], "step": lambda h, t: (a * h[-1] + c) % m}


def _quadratic_linear(rng):
    b, c = rng.choice([v for v in range(-5, 6) if v]), rng.choice([v for v in range(-10, 11) if v])
    return {"family": "quadratic-linear", "start": [c], "step": lambda h, t: t * (t + b) + c}


def _second_order(rng):
    c = rng.choice([v for v in range(-5, 6) if v])
    return {"family": "second-order", "start": [rng.randint(-10, 10), rng.randint(-10, 10)],
            "step": lambda h, t: 2 * h[-1] - h[-2] + c}


def _index_recurrence(rng):
    k, c = rng.choice([2, 3]), rng.choice([v for v in range(-5, 6) if v])
    return {"family": "index-recurrence", "start": [rng.randint(-10, 10)], "step": lambda h, t: h[-1] + k * t + c}


DEEP_FAMILIES = [_lcg, _quadratic_linear, _second_order, _index_recurrence]


class SequenceWorld:
    AGENT_METHODS = ("stream_ids", "next_term")

    def __init__(self, seed: int, n_rule_streams: int = 8, n_noise_streams: int = 2,
                 families=None, noise: float = 0.0):
        self.env_id = f"sequence-world-{seed}"
        rng = random.Random(seed)
        gens = [rng.choice(families or RULE_FAMILIES)(rng) for _ in range(n_rule_streams)]
        gens += [_noise(rng) for _ in range(n_noise_streams)]
        rng.shuffle(gens)
        self._gens = {f"S{i}": g for i, g in enumerate(gens)}
        self._hist = {sid: [] for sid in self._gens}
        self.noise = noise
        self._corrupt = {sid: random.Random(zlib.crc32(f"{seed}/{sid}/noise".encode())) for sid in self._gens}
        self.corrupted = {sid: 0 for sid in self._gens}

    def _term(self, sid: str, history: list) -> int:
        g = self._gens[sid]
        t = len(history)
        return g["start"][t] if t < len(g["start"]) else g["step"](history, t)

    # ----------------------------------------------------------- agent-visible
    def stream_ids(self) -> list:
        return list(self._gens)

    def next_term(self, sid: str) -> int:
        h = self._hist[sid]
        v = self._term(sid, h)
        r = self._corrupt[sid]
        if self.noise and self._gens[sid]["family"] != "noise" and r.random() < self.noise:
            v += r.choice([d for d in range(-15, 16) if d])
            self.corrupted[sid] += 1
        h.append(v)
        return v

    # ------------------------------------------------------------ harness-only
    def _future(self, sid: str, k: int) -> list:
        g = self._gens[sid]
        if g["family"] == "noise":
            # Noise has no future to predict; any guess is chance level.
            r = random.Random(zlib.crc32(f"{sid}:{len(self._hist[sid])}".encode()))
            return [r.randint(-10**6, 10**6) for _ in range(k)]
        # Rule streams are scored on their noiseless continuation: did the agent identify the generator?
        h = list(self._hist[sid])
        out = []
        for _ in range(k):
            h.append(self._term(sid, h))
            out.append(h[-1])
        return out

    def _family(self, sid: str) -> str:
        return self._gens[sid]["family"]
