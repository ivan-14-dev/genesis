"""Perception engine: raw observations -> named integer features, novelty, differences.

The engine is modality-agnostic at its interface (:class:`Percept`). Images
are first read by :mod:`genesis.vision` (learned discrete codes, readout
calibrated by the agent's own motor commands, abstention), which produces the
same feature dictionaries; audio is not implemented.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class Percept:
    modality: str
    features: dict
    novelty: float
    signature: tuple


class Perception:
    def __init__(self):
        self.visits: Counter = Counter()

    def perceive(self, modality: str, features: dict, context: str = "") -> Percept:
        sig = (modality, context, tuple(sorted(features.items())))
        return Percept(modality, dict(features), self.novelty(sig), sig)

    def novelty(self, signature: tuple) -> float:
        return 1.0 / math.sqrt(1.0 + self.visits[signature])

    def register(self, percept: Percept) -> None:
        self.visits[percept.signature] += 1

    @staticmethod
    def sequence_features(window: list, t: int) -> dict:
        """window = most recent terms, oldest first; returns lag features x[t-k]."""
        feats = {f"x[t-{k}]": window[-k] for k in range(1, min(3, len(window)) + 1)}
        feats["t"] = t
        return feats

    @staticmethod
    def differences(before: dict, after: dict) -> dict:
        return {k: after[k] - before[k] for k in before if k in after and after[k] != before[k]}

    @staticmethod
    def similarity(a: dict, b: dict) -> float:
        keys = set(a) | set(b)
        return sum(a.get(k) == b.get(k) for k in keys) / max(len(keys), 1)
