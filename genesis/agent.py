"""GENESIS agent: the cognitive cycle that ties all engines together.

OBSERVE -> PERCEIVE -> REMEMBER -> PREDICT -> EXPLORE -> ACT -> OBSERVE OUTCOME
-> COMPARE -> LEARN -> STORE -> CONSOLIDATE -> ADAPT

State persists in a single SQLite file; a new ``Genesis`` on the same path
resumes the same individual (new session) after verifying that its innate
manifest and safety policy are unchanged.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional

from .adapters import LabAdapter, PassiveLabAdapter, SequenceAdapter, sequence_rollout_probes
from .attention import Attention
from .consolidation import Consolidation
from .curiosity import Curiosity
from .environments.base import BudgetExceeded
from .induction import InductionConfig, RuleLearner
from .innate import DEFAULT_SAFETY, SafetyPolicy, innate_manifest
from .memory import LongTermMemory, WorkingMemory
from .metacognition import Metacognition
from .perception import Perception


def default_configs(safety: SafetyPolicy) -> dict:
    common = dict(max_candidates=safety.max_candidates_per_induction, max_seconds=safety.max_seconds_per_induction)
    return {
        "sequence": InductionConfig(max_size=5, **common),
        "lab": InductionConfig(max_size=3, allow_conditionals=True, outcome_range=101, **common),
    }


# The only parameters a model version (stage-7 self-improvement) may change.
# Safety limits (SafetyPolicy, search budgets) are deliberately absent.
TUNABLE = {
    "sequence.validation_k": (2, 8), "sequence.shape_bias": (0.0, 4.0),
    "sequence.max_proposed_constants": (4, 24), "lab.validation_k": (2, 8), "lab.shape_bias": (0.0, 4.0),
    "agent.default_resolution_cost": (4, 40), "agent.review_interval": (2, 100),
    "curiosity.novelty_weight": (0.0, 4.0), "curiosity.review_weight": (0.0, 4.0),
}


def validate_config(config: dict) -> dict:
    for k, v in config.items():
        if k not in TUNABLE:
            raise PermissionError(f"parameter {k!r} is not tunable (safety or architecture)")
        lo, hi = TUNABLE[k]
        if not (lo <= v <= hi):
            raise ValueError(f"{k}={v} outside allowed range [{lo}, {hi}]")
    return config


class IntegrityError(RuntimeError):
    pass


@dataclass
class SessionReport:
    steps: int = 0
    stopped_because: str = "step limit"
    utilities: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    designs: dict = field(default_factory=dict)
    seconds: float = 0.0


class Genesis:
    def __init__(self, memory_path: str, safety: SafetyPolicy = DEFAULT_SAFETY, seed: int = 0,
                 procedural_prior: bool = True, consolidate_every: int = 50, idle_patience: int = 20,
                 review_interval: int = 10, reviews_to_graduate: int = 3, config: Optional[dict] = None):
        self.safety = safety
        self.seed = seed
        self.rng = random.Random(seed)
        self.ltm = LongTermMemory(memory_path)
        self.wm = WorkingMemory(capacity=9)
        self.perception = Perception()
        self.attention = Attention()
        self.curiosity = Curiosity()
        self.metacog = Metacognition()
        self.consolidator = Consolidation()
        self.configs = default_configs(safety)
        self.procedural_prior = procedural_prior
        self.consolidate_every = consolidate_every
        self.idle_patience = idle_patience
        self.review_interval = review_interval
        self.reviews_to_graduate = reviews_to_graduate
        # Spaced re-testing of resolved judgements: key -> {"next", "interval", "count"}
        self.reviews: dict = {}
        # Metacognitive cost model: observations each past problem needed before it was resolved.
        self.resolution_costs: dict = {}
        self.context_obs: dict = {}
        self.default_resolution_cost = 12
        self.learners: dict = {}
        self.learner_env: dict = {}
        self.learner_kind: dict = {}
        self.ltm_retrievals = 0
        self._shape_cache: Optional[dict] = None
        self.config = validate_config(dict(config if config is not None else
                                           (self.ltm.get_meta("active_config") or {})))
        for k, v in self.config.items():
            scope, name = k.split(".")
            if scope in self.configs:
                self.configs[scope] = replace(self.configs[scope], **{name: v})
            elif scope == "agent":
                setattr(self, name, v)
            elif scope == "curiosity":
                setattr(self.curiosity, name, v)
        if self.ltm.get_meta("born_ts") is None:
            self._birth()
        else:
            self._wake()

    # ------------------------------------------------------------- lifecycle
    def _birth(self) -> None:
        manifest = innate_manifest()
        self.ltm.set_meta("born_ts", time.time())
        self.ltm.set_meta("innate_manifest", manifest)
        self.ltm.set_meta("safety_policy_digest", self.safety.digest())
        self.ltm.set_meta("session", 0)
        self.ltm.set_meta("interactions", 0)
        self.session = 0
        self.interactions = 0
        audit = self.audit_initial_state()
        self.ltm.set_meta("stage0_audit", audit)
        self.record_assessment("stage0_audit_passed", 1.0 if audit["passed"] else 0.0)
        self.ltm.commit()

    def audit_initial_state(self) -> dict:
        counts = {t: self.ltm.count(t) for t in ("episodes", "knowledge", "procedures", "associations",
                                                 "learner_state")}
        manifest = self.ltm.get_meta("innate_manifest")
        checks = {
            "memory_empty": all(v == 0 for v in counts.values()),
            "no_domain_knowledge_in_manifest": manifest["domain_knowledge"] == [],
            "no_pretrained_models": manifest["pretrained_models"] == [],
            "no_learners": not self.learners,
        }
        return {"passed": all(checks.values()), "checks": checks, "counts": counts}

    def _wake(self) -> None:
        if self.ltm.get_meta("innate_manifest")["digest"] != innate_manifest()["digest"]:
            raise IntegrityError("innate manifest changed since birth")
        if self.ltm.get_meta("safety_policy_digest") != self.safety.digest():
            raise IntegrityError("safety policy differs from the one fixed at birth; refusing to run")
        self.session = self.ltm.get_meta("session") + 1
        self.interactions = self.ltm.get_meta("interactions")
        self.ltm.set_meta("session", self.session)
        self.reviews = self.ltm.get_meta("reviews") or {}
        self.resolution_costs = self.ltm.get_meta("resolution_costs") or {}
        self.context_obs = {r[0]: r[1] for r in self.ltm.db.execute(
            "SELECT context, COUNT(*) FROM episodes GROUP BY context")}
        for key, env_id, state in self.ltm.load_learners():
            kind = state["kind"]
            self.learners[key] = RuleLearner.from_state(state, self.configs[kind], self._shape_prior,
                                                        self._probe_fn(kind), self._library())
            self.learner_env[key] = env_id
            self.learner_kind[key] = kind

    def close(self) -> None:
        self.consolidate()
        self.ltm.close()

    # ---------------------------------------------------------------- helpers
    def _shape_prior(self, shape: str) -> float:
        if not self.procedural_prior:
            return 0.0
        if self._shape_cache is None:
            self._shape_cache = self.ltm.shape_successes()
        return math.log1p(self._shape_cache.get(shape, 0))

    def _library(self):
        return self._shape_library if self.procedural_prior else None

    def _shape_library(self) -> list:
        """Procedural memory: shapes that led to validated knowledge, most successful first."""
        if self._shape_cache is None:
            self._shape_cache = self.ltm.shape_successes()
        return [s for s, n in sorted(self._shape_cache.items(), key=lambda t: -t[1]) if n > 0]

    @staticmethod
    def _probe_fn(kind: str):
        return sequence_rollout_probes if kind == "sequence" else None

    def learner(self, key: str, kind: str, env_id: str, var_names: list) -> RuleLearner:
        lr = self.learners.get(key)
        if lr is None:
            lr = RuleLearner(key, var_names, self.configs[kind], self._shape_prior, self._probe_fn(kind),
                             self._library())
            self.learners[key] = lr
            self.learner_env[key] = env_id
            self.learner_kind[key] = kind
        return lr

    def record_assessment(self, metric: str, value: float, detail=None) -> None:
        self.ltm.record_metric(self.session, metric, value, {"type": "assessment", "detail": detail})

    def assessments(self) -> dict:
        return self.ltm.assessments()

    # --------------------------------------------------- working memory/recall
    def recall_sequence(self, key: str) -> tuple:
        item = self.wm.get(("seq", key))
        if item is None:
            eps = self.ltm.recent_episodes(key, 3)
            window = [json.loads(e["outcome"])["term"] for e in eps]
            t = self.ltm.db.execute("SELECT COUNT(*) FROM episodes WHERE context=?", (key,)).fetchone()[0]
            item = (window, t)
            self.ltm_retrievals += 1
            self.wm.put(("seq", key), item, salience=0.5)
        else:
            self.wm.touch(("seq", key))
        return item

    def remember_term(self, key: str, y: int) -> None:
        item = self.wm.get(("seq", key))
        if item is None:
            return  # not in working memory: the next recall reads the stored episode from LTM
        window, t = item
        self.wm.put(("seq", key), ((list(window) + [y])[-3:], t + 1), salience=1.0)

    # ------------------------------------------------------------ the cycle
    def live(self, adapter, max_steps: int, strategy: str = "curiosity", stop_when_idle: bool = True) -> SessionReport:
        rep = SessionReport()
        t0 = time.perf_counter()
        idle = 0
        for _ in range(max_steps):
            if self.interactions >= self.safety.max_interactions_per_session * (self.session + 1):
                rep.stopped_because = "safety interaction limit"
                break
            # 1-2 OBSERVE + PERCEIVE (3 REMEMBER happens inside adapters via recall)
            try:
                cands = adapter.candidates()
            except BudgetExceeded:
                rep.stopped_because = "sandbox budget exhausted"
                break
            # 4 PREDICT and evaluate options
            review_due = self._review_due()
            utils = self._evaluate(adapter, cands, review_due)
            # 5 EXPLORE / DECIDE
            focus = None
            if strategy == "curiosity" and getattr(adapter, "focusable", False):
                left = max_steps - rep.steps
                if hasattr(adapter, "sb"):
                    left = min(left, adapter.sb.remaining())
                focus = self._focus(adapter.kind, cands, utils, left)
            if focus is not None:
                idx = focus
                rep.designs["focus"] = rep.designs.get("focus", 0) + 1
            elif strategy == "curiosity":
                idx = self.attention.select(cands, utils, self.rng)
            elif strategy == "random":
                idx = self.rng.randrange(len(cands))
            elif strategy == "round_robin":
                idx = rep.steps % len(cands)
            else:
                raise ValueError(strategy)
            sit = cands[idx]
            rep.utilities.append(round(utils[idx], 4))
            if strategy == "curiosity" and stop_when_idle and len(cands) > 1:
                idle = idle + 1 if utils[idx] < 0.01 else 0
                if idle >= self.idle_patience:
                    rep.stopped_because = "nothing left to learn within current hypothesis space"
                    break
            # 6-7 ACT + OBSERVE OUTCOME
            try:
                outcome = adapter.execute(sit)
            except BudgetExceeded:
                rep.stopped_because = "sandbox budget exhausted"
                break
            self.interactions += 1
            self.context_obs[sit.context] = self.context_obs.get(sit.context, 0) + 1
            rep.steps += 1
            design = sit.action.get("design", sit.provenance)
            rep.designs[design.split(":")[0]] = rep.designs.get(design.split(":")[0], 0) + 1
            # 8 COMPARE
            surprise, n_ok = 0.0, 0
            pred_log = {}
            for q in sit.queries:
                p = sit.predictions[q.key]
                y = outcome.targets[q.key]
                lr = self.learners[q.key]
                surprise += -math.log2(max(p.prob_of(y, lr.cfg.outcome_range), 1e-12))
                n_ok += p.value == y
                pred_log[q.key] = {"predicted": p.value, "confidence": round(p.confidence, 4), "actual": y}
            if sit.queries:
                rep.accuracy.append(n_ok / len(sit.queries))
            novelty = sit.percept.novelty if sit.percept else 1.0
            salience = self.attention.salience(novelty, surprise, outcome.reward or 0.0)
            # 10 STORE (episode first so learned knowledge can cite it)
            ctx = sit.context
            eid = self.ltm.record_episode(self.session, adapter.env_id, ctx, sit.provenance, sit.action,
                                          sit.percept.features if sit.percept else None,
                                          outcome.observation,
                                          pred_log, surprise, outcome.reward, salience)
            # 9 LEARN
            for q in sit.queries:
                p = sit.predictions[q.key]
                y = outcome.targets[q.key]
                res = self.learners[q.key].observe(q.inputs, y, eid, sit.provenance)
                self.metacog.record_prediction(q.key, p.confidence, p.value == y)
                self._update_reviews(q.key, res.status_after, q.key in review_due)
                if res.status_after in ("validated", "unpredictable") and res.status_before != res.status_after \
                        and res.status_before not in ("validated", "unpredictable"):
                    self.resolution_costs.setdefault(adapter.kind, []).append(self.context_obs[sit.context])
            adapter.after(sit, outcome)
            if sit.percept:
                self.perception.register(sit.percept)
            # 11 CONSOLIDATE periodically; 12 ADAPT happens via backoff, priors, review schedule
            if self.interactions % self.consolidate_every == 0:
                self.consolidate()
        self.consolidate()
        rep.seconds = time.perf_counter() - t0
        return rep

    def expected_resolution_cost(self, kind: str) -> float:
        costs = self.resolution_costs.get(kind) or []
        return statistics.median(costs) if costs else self.default_resolution_cost

    def _focus(self, kind: str, cands: list, utils: list, budget_left: int) -> Optional[int]:
        """Budget-aware attention.

        If the remaining budget cannot cover every open problem, finishing a few is worth more
        than sampling all of them a little: pick the open problem that is cheapest to finish
        (expected cost learned from past problems). Problems that already took more than twice
        the expected cost are set aside. Returns None when the budget is sufficient.
        """
        est = self.expected_resolution_cost(kind)
        open_ = []
        for i, sit in enumerate(cands):
            keys = [q.key for q in sit.queries] or [sit.context]
            if all(k in self.learners and self.learners[k].status() in ("validated", "unpredictable")
                   for k in keys):
                continue
            n = self.context_obs.get(sit.context, 0)
            if n > 2 * est:
                continue
            open_.append((max(est - n, 1.0), -utils[i], i))
        if not open_ or budget_left >= sum(c for c, _, _ in open_):
            return None
        feasible = [o for o in open_ if o[0] <= budget_left] or open_
        return min(feasible)[2]

    def _review_due(self) -> set:
        return {k for k, r in self.reviews.items() if r["next"] is not None and r["next"] <= self.interactions}

    def _evaluate(self, adapter, cands: list, review_due: set) -> list:
        """Predict the outcome of every candidate situation and score its curiosity utility."""
        utils = []
        for sit in cands:
            counts, ranges = {}, {}
            for q in sit.queries:
                lr = self.learner(q.key, adapter.kind, adapter.env_id, adapter.var_names(q.key))
                sit.predictions[q.key] = lr.predict(q.inputs)
                counts[q.key] = len(lr.data) if lr.status() not in ("validated", "unpredictable") else 10**9
                ranges[q.key] = lr.cfg.outcome_range
            u, _ = self.curiosity.utility(sit.predictions, counts, ranges, review_due, sit.warmup)
            utils.append(u)
        return utils

    def peek(self, adapter) -> float:
        """Best curiosity utility currently available in a world, without acting."""
        cands = adapter.candidates()
        return max(self._evaluate(adapter, cands, self._review_due()), default=0.0)

    def _update_reviews(self, key: str, status: str, was_review: bool) -> None:
        resolved = status in ("validated", "unpredictable")
        r = self.reviews.get(key)
        if not resolved:
            self.reviews.pop(key, None)
        elif r is None or r.get("status") != status:
            self.reviews[key] = {"next": self.interactions + self.review_interval, "interval": self.review_interval,
                                 "count": 0, "status": status}
        elif was_review:
            r["count"] += 1
            r["interval"] *= 2
            r["next"] = None if r["count"] >= self.reviews_to_graduate else self.interactions + r["interval"]

    def consolidate(self) -> dict:
        self.ltm.set_meta("interactions", self.interactions)
        self.ltm.set_meta("reviews", self.reviews)
        self.ltm.set_meta("resolution_costs", self.resolution_costs)
        summary = self.consolidator.run(self)
        self._shape_cache = None
        cal = self.metacog.calibration()
        if cal["n"]:
            self.ltm.record_metric(self.session, "calibration", cal["ece"], cal)
        self.ltm.commit()
        return summary

    # --------------------------------------------------------- convenience
    def explore_sequences(self, sandbox, max_steps: int, strategy: str = "curiosity", **kw) -> SessionReport:
        return self.live(SequenceAdapter(self, sandbox), max_steps, strategy, **kw)

    def watch_lab(self, sandbox, n: int) -> SessionReport:
        return self.live(PassiveLabAdapter(self, sandbox), n, "curiosity", stop_when_idle=False)

    def experiment_lab(self, sandbox, max_steps: int, strategy: str = "curiosity", **kw) -> SessionReport:
        return self.live(LabAdapter(self, sandbox), max_steps, strategy, **kw)

    # ---------------------------------------------------- imagination / use
    def rollout_sequence(self, key: str, k: int) -> list:
        """Predict the next k terms by feeding the agent's own predictions back in."""
        lr = self.learners.get(key)
        window, t = self.recall_sequence(key)
        window = list(window)
        out = []
        for _ in range(k):
            if lr is None or len(window) < 3:
                out.append(None)
                continue
            p = lr.predict(Perception.sequence_features(window, t))
            out.append(p.value)
            window = (window + [p.value if p.value is not None else 0])[-3:]
            t += 1
        return out

    def predict_transition(self, env_id: str, props: list, state: dict, action: str) -> tuple:
        nxt, conf = {}, 1.0
        for p in props:
            lr = self.learners.get(f"{env_id}/{action}/{p}")
            pred = lr.predict(state) if lr else None
            if pred is None or pred.value is None:
                return None, 0.0
            nxt[p] = pred.value
            conf *= pred.confidence
        return nxt, conf

    def plan(self, env_id: str, props: list, actions: list, state: dict, goal: tuple) -> Optional[list]:
        """Breadth-first search through the agent's learned transition model."""
        g, v = goal
        if state[g] == v:
            return []
        frontier = deque([(state, [])])
        seen = {tuple(state[p] for p in props)}
        while frontier:
            s, path = frontier.popleft()
            if len(path) >= self.safety.max_plan_depth:
                continue
            for a in actions:
                nxt, conf = self.predict_transition(env_id, props, s, a)
                if nxt is None or conf < 0.5:
                    continue
                if nxt[g] == v:
                    return path + [a]
                sig = tuple(nxt[p] for p in props)
                if sig not in seen:
                    seen.add(sig)
                    frontier.append((nxt, path + [a]))
        return None

    def pursue_goal(self, sandbox, task_id: int, strategy: str = "plan", max_steps: int = 8) -> dict:
        spec = sandbox.call("spec")
        props, actions, env_id = spec["properties"], spec["actions"], sandbox.env_id
        task = sandbox.call("begin_task", task_id)
        state, goal = task["state"], tuple(task["goal"])
        steps = replans = 0
        success = False
        first_plan = None
        while steps < max_steps:
            if strategy == "plan":
                plan = self.plan(env_id, props, actions, state, goal)
                if first_plan is None:
                    first_plan = plan
                if not plan:
                    break
                a = plan[0]
                expected, _ = self.predict_transition(env_id, props, state, a)
            else:
                a, expected = self.rng.choice(actions), None
            res = sandbox.call("act", a)
            steps += 1
            self.interactions += 1
            eid = self.ltm.record_episode(self.session, env_id, f"{env_id}/{a}", "goal-directed",
                                          {"action": a, "goal": list(goal)}, state, {"next": res["state"]},
                                          {"expected": expected}, None, res["reward"], None)
            for p in props:
                self.learner(f"{env_id}/{a}/{p}", "lab", env_id, props).observe(state, res["state"][p], eid,
                                                                               "goal-directed")
            if expected is not None and expected != res["state"]:
                replans += 1
            state = res["state"]
            if res["reward"] > 0:
                success = True
                break
            if res["done"]:
                break
        self.ltm.record_metric(self.session, "goal_reward", 1.0 if success else 0.0,
                               {"task": task_id, "steps": steps, "strategy": strategy})
        return {"task": task_id, "success": success, "steps": steps, "replans": replans,
                "plan": first_plan, "goal": list(goal)}

    # --------------------------------------------------------- observability
    def report(self) -> dict:
        ltm = self.ltm
        return {
            "session": self.session,
            "interactions": self.interactions,
            "stage": Metacognition.current_stage(self.assessments()),
            "assessments": self.assessments(),
            "memory": {t: ltm.count(t) for t in ("episodes", "knowledge", "evidence", "procedures",
                                                 "associations", "metacog")} | {"bytes": ltm.size_bytes(),
                                                                              "working_memory": self.wm.keys(),
                                                                              "ltm_retrievals": self.ltm_retrievals},
            "calibration": self.metacog.calibration(),
            "self_report": Metacognition.self_report(self.learners),
            "questions": [q.text for q in Curiosity.knowledge_gaps(self.learners)],
            "procedural_memory": ltm.shape_successes(),
        }
