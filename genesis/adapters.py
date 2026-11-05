"""Sensorimotor adapters: how the agent turns a sandboxed world into situations.

An adapter knows the *format* of a world's observations and actions (its
"body"), never its rules. Each candidate :class:`Situation` lists the learning
problems (queries) the agent can make a prediction about before acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .perception import Percept, Perception

SEQ_VARS = ["x[t-1]", "x[t-2]", "x[t-3]", "t"]


@dataclass
class Query:
    key: str
    inputs: dict


@dataclass
class Situation:
    action: dict
    queries: list
    provenance: str
    context: str
    percept: Optional[Percept] = None
    warmup: float = 0.0
    predictions: dict = field(default_factory=dict)


@dataclass
class Outcome:
    targets: dict
    observation: dict
    reward: Optional[float] = None


def sequence_rollout_probes(lr, expr, k: int = 12) -> list:
    """On-manifold probes: the inputs the stream would produce if ``expr`` were true."""
    if not lr.data:
        return []
    inp, y = lr.data[-1]
    w = [inp["x[t-3]"], inp["x[t-2]"], inp["x[t-1]"], y]
    t = inp["t"] + 1
    probes = []
    for _ in range(k):
        p = {"x[t-1]": w[-1], "x[t-2]": w[-2], "x[t-3]": w[-3], "t": t}
        probes.append(p)
        v = expr.evaluate(p)
        if v is None:
            break
        w.append(v)
        t += 1
    return probes


class SequenceAdapter:
    kind = "sequence"

    def __init__(self, agent, sandbox):
        self.agent = agent
        self.sb = sandbox
        self.env_id = sandbox.env_id
        self.streams = sandbox.call("stream_ids")

    def key(self, sid: str) -> str:
        return f"{self.env_id}/{sid}"

    def var_names(self, key: str) -> list:
        return SEQ_VARS

    def candidates(self) -> list:
        out = []
        for sid in self.streams:
            key = self.key(sid)
            window, t = self.agent.recall_sequence(key)
            if len(window) >= 3:
                feats = Perception.sequence_features(window, t)
                percept = self.agent.perception.perceive("sequence", feats, key)
                out.append(Situation({"attend": sid}, [Query(key, feats)], "observation", key, percept))
            else:
                out.append(Situation({"attend": sid}, [], "observation", key, None, warmup=1.0))
        return out

    def execute(self, sit: Situation) -> Outcome:
        sid = sit.action["attend"]
        y = self.sb.call("next_term", sid)
        return Outcome({self.key(sid): y}, {"term": y})

    def after(self, sit: Situation, outcome: Outcome) -> None:
        key = self.key(sit.action["attend"])
        self.agent.remember_term(key, outcome.targets[key])


class LabAdapter:
    """Controlled experiments: the agent sets up an object and applies an action."""

    kind = "lab"

    def __init__(self, agent, sandbox, n_random: int = 30):
        self.agent = agent
        self.sb = sandbox
        self.env_id = sandbox.env_id
        spec = sandbox.call("spec")
        self.props = spec["properties"]
        self.actions = spec["actions"]
        self.lo, self.hi = spec["settable_range"]
        self.n_random = n_random
        self.history: list = []

    def key(self, action: str, prop: str) -> str:
        return f"{self.env_id}/{action}/{prop}"

    def var_names(self, key: str) -> list:
        return self.props

    def _situation(self, state: dict, action: str, design: str, provenance: str = "experiment") -> Situation:
        percept = self.agent.perception.perceive("state", {**state, "_action": self.actions.index(action)},
                                                 self.env_id)
        return Situation({"experiment": action, "state": dict(state), "design": design},
                         [Query(self.key(action, p), dict(state)) for p in self.props], provenance,
                         f"{self.env_id}/{action}", percept)

    def candidates(self) -> list:
        rng = self.agent.rng
        sits = []
        for _ in range(self.n_random):
            state = {p: rng.randint(self.lo, self.hi) for p in self.props}
            sits.append(self._situation(state, rng.choice(self.actions), "random-configuration"))
        # Controlled variations: repeat a recent experiment changing exactly one variable.
        for state, action in self.history[-4:]:
            for p in self.props:
                s2 = dict(state)
                s2[p] = rng.choice([v for v in range(self.lo, self.hi + 1) if v != state[p]])
                sits.append(self._situation(s2, action, f"one-variable-change:{p}"))
        return sits

    def execute(self, sit: Situation) -> Outcome:
        state, action = sit.action["state"], sit.action["experiment"]
        nxt = self.sb.call("experiment", state, action)
        self.history.append((dict(state), action))
        return Outcome({self.key(action, p): nxt[p] for p in self.props}, {"state": state, "next": nxt})

    def after(self, sit: Situation, outcome: Outcome) -> None:
        pass


class PassiveLabAdapter(LabAdapter):
    """Watching the world: no choice of experiment, natural (confounded) states."""

    def candidates(self) -> list:
        state, action, nxt = self.sb.call("passive_observation")
        sit = self._situation(state, action, "passive", provenance="observation")
        sit.action["_observed_next"] = nxt
        return [sit]

    def execute(self, sit: Situation) -> Outcome:
        nxt = sit.action.pop("_observed_next")
        action = sit.action["experiment"]
        return Outcome({self.key(action, p): nxt[p] for p in self.props}, {"state": sit.action["state"], "next": nxt})
