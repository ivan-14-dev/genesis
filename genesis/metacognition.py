"""Metacognitive engine and developmental curriculum.

Monitors the agent's own predictions (accuracy, calibration, learning
progress), produces an evidence-backed self report (what it knows, what it
does not, what it judged unpredictable, which beliefs were refuted), and
decides the current developmental stage from *measured* assessments only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageGate:
    stage: int
    name: str
    criteria: tuple  # ((metric, op, threshold), ...); empty = not implemented yet
    implemented: bool = True


CURRICULUM = (
    StageGate(0, "initialization", (("stage0_audit_passed", ">=", 1.0),)),
    StageGate(1, "sensory & pattern discovery",
              (("seq_rule_extrapolation_accuracy", ">=", 0.8), ("seq_false_validation_rate", "<=", 0.05))),
    StageGate(2, "relationships & associations", (("lab_passive_prediction_accuracy", ">=", 0.8),)),
    StageGate(3, "causal discovery",
              (("lab_heldout_accuracy", ">=", 0.95), ("lab_confound_resolved", ">=", 1.0))),
    StageGate(4, "memory & knowledge",
              (("retention_after_reload", ">=", 0.99), ("forgetting_after_new_world", "<=", 0.02))),
    StageGate(5, "reasoning & planning", (("planning_success_rate", ">=", 0.9),)),
    # Stage 6: allocating one budget across several worlds by learning progress must beat
    # round-robin allocation and avoid sinking attention into a world of pure noise.
    StageGate(6, "autonomous exploration",
              (("autonomy_gain_vs_round_robin", ">=", 0.02), ("autonomy_noise_share", "<=", 0.25))),
    # Stage 7: self-evaluation is reproducible, never touches safety, never deploys a holdout
    # regression, and the agent can name its own limitations from its records.
    StageGate(7, "continuous cognitive development",
              (("self_improvement_reproducible", ">=", 1.0), ("safety_policy_unchanged", ">=", 1.0),
               ("self_improvement_no_holdout_regression", ">=", 1.0), ("limitations_identified", ">=", 1.0))),
)


def _passes(value, op: str, threshold: float) -> bool:
    if value is None:
        return False
    return value >= threshold if op == ">=" else value <= threshold


class Metacognition:
    def __init__(self):
        self.records: list = []  # (key, confidence, correct)

    def record_prediction(self, key: str, confidence: float, correct: bool) -> None:
        self.records.append((key, confidence, bool(correct)))

    def calibration(self, bins: int = 10) -> dict:
        n = len(self.records)
        if n == 0:
            return {"n": 0, "accuracy": None, "brier": None, "ece": None}
        acc = sum(c for _, _, c in self.records) / n
        brier = sum((p - c) ** 2 for _, p, c in self.records) / n
        ece = 0.0
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            sel = [(p, c) for _, p, c in self.records if lo <= p < hi or (b == bins - 1 and p == 1.0)]
            if sel:
                ece += len(sel) / n * abs(sum(p for p, _ in sel) / len(sel) - sum(c for _, c in sel) / len(sel))
        return {"n": n, "accuracy": round(acc, 4), "brier": round(brier, 4), "ece": round(ece, 4)}

    def learning_progress(self, window: int = 25) -> float:
        """Accuracy over the last window minus accuracy over the window before it."""
        r = [c for _, _, c in self.records]
        if len(r) < 2 * window:
            return 0.0
        return sum(r[-window:]) / window - sum(r[-2 * window:-window]) / window

    @staticmethod
    def self_report(learners: dict) -> dict:
        report = {"validated": [], "candidate": [], "unknown": [], "unpredictable": [], "refuted": []}
        for key, lr in sorted(learners.items()):
            st = lr.status()
            top, mass, p_null = lr.best()
            entry = {"key": key, "n_observations": len(lr.data),
                     "belief": str(top.expr) if top else None, "confidence": round(mass, 4),
                     "p_unpredictable": round(p_null, 4)}
            if top is not None:
                entry["forward_predictions"] = {"correct": top.fwd_ok, "wrong": top.fwd_bad}
            report[st].append(entry)
            for r in lr.refutations:
                report["refuted"].append({"key": key, **r})
        return report

    @staticmethod
    def current_stage(assessments: dict) -> dict:
        passed = []
        for gate in CURRICULUM:
            if not gate.implemented:
                return {"stage": gate.stage, "name": gate.name, "passed": passed, "blocked_by": "not implemented"}
            failing = [c for c in gate.criteria if not _passes(assessments.get(c[0]), c[1], c[2])]
            if failing:
                return {"stage": gate.stage, "name": gate.name, "passed": passed,
                        "blocked_by": [{"metric": m, "required": f"{op} {t}", "value": assessments.get(m)}
                                       for m, op, t in failing]}
            passed.append(gate.stage)
        return {"stage": CURRICULUM[-1].stage, "name": CURRICULUM[-1].name, "passed": passed, "blocked_by": None}
