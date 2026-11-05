"""Attention engine: salience of experiences and selection among options."""

from __future__ import annotations

import random


class Attention:
    def __init__(self, w_novelty: float = 0.4, w_surprise: float = 0.4, w_reward: float = 0.2):
        self.w = (w_novelty, w_surprise, w_reward)

    def salience(self, novelty: float, surprise_bits: float, reward: float = 0.0) -> float:
        wn, ws, wr = self.w
        return wn * novelty + ws * min(surprise_bits / 7.0, 1.0) + wr * min(abs(reward), 1.0)

    @staticmethod
    def select(options: list, utilities: list, rng: random.Random) -> int:
        """Index of the highest-utility option; ties broken randomly."""
        best = max(utilities)
        top = [i for i, u in enumerate(utilities) if u >= best - 1e-9]
        return rng.choice(top)
