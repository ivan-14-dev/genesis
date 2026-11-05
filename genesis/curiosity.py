"""Curiosity engine: intrinsic motivation from expected information gain.

The value of an experiment is how much it is expected to reduce uncertainty
over the agent's *own* competing hypotheses (Bayesian experimental design),
plus a decaying count-based novelty bonus for problems with little data
(knowledge gaps). It is not random behaviour: an experiment on which all
surviving hypotheses agree has (near) zero value, including pure-noise sources
that the agent has already judged unpredictable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .induction import Prediction


def expected_information_gain(pred: Prediction, outcome_range: int) -> float:
    """Mutual information (nats) between the outcome and the hypothesis identity.

    Each hypothesis predicts a point value; the null hypothesis spreads its mass
    uniformly over ``outcome_range`` values. EIG = H(mixture) - E_h[H(outcome|h)].
    """
    w_null = max(pred.p_null, 0.0)
    q = w_null / outcome_range
    h_mix = 0.0
    for _, p in pred.distribution:
        pv = p + q
        if pv > 0:
            h_mix -= pv * math.log(pv)
    rest = outcome_range - len(pred.distribution)
    if q > 0 and rest > 0:
        h_mix -= rest * q * math.log(q)
    h_cond = w_null * math.log(outcome_range)
    return max(h_mix - h_cond, 0.0)


@dataclass
class Question:
    key: str
    text: str
    alternatives: list


class Curiosity:
    def __init__(self, novelty_weight: float = 1.0, review_weight: float = 0.3):
        self.novelty_weight = novelty_weight
        self.review_weight = review_weight

    def utility(self, predictions: dict, data_counts: dict, outcome_range: dict, review_due: set,
                warmup_bonus: float = 0.0) -> tuple:
        eig = sum(expected_information_gain(p, outcome_range[k]) for k, p in predictions.items())
        novelty = sum(self.novelty_weight / (1 + data_counts.get(k, 0)) for k in predictions)
        review = self.review_weight * sum(1 for k in predictions if k in review_due)
        total = eig + novelty + review + warmup_bonus
        return total, {"eig": round(eig, 4), "novelty": round(novelty, 4), "review": review,
                       "warmup": warmup_bonus}

    @staticmethod
    def knowledge_gaps(learners: dict, max_questions: int = 10) -> list:
        """Questions the agent would like answered, derived from competing hypotheses."""
        out = []
        for key, lr in sorted(learners.items()):
            st = lr.status()
            if st in ("validated", "unpredictable"):
                continue
            post, p_null = lr.posterior()
            if not post:
                out.append(Question(key, f"What determines {key}? (no hypothesis survives yet)", []))
                continue
            seen, alts = set(), []
            for h, p in post:
                if h.probe_sig in seen:
                    continue
                seen.add(h.probe_sig)
                alts.append((str(h.expr), round(p, 3)))
                if len(alts) == 3:
                    break
            if len(alts) > 1:
                out.append(Question(key, f"For {key}: is it {alts[0][0]} or {alts[1][0]}?", alts))
            else:
                out.append(Question(key, f"Does {alts[0][0]} keep holding for {key}?", alts))
        return out[:max_questions]
