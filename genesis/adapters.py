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
        v = expr.evaluate(p, True)
        if v is None:
            break
        w.append(v)
        t += 1
    return probes


def sequence_dream(expr, rng, length: int = 16) -> list:
    """Generative hook for dreaming: the stream ``expr`` would produce from random starting terms."""
    w = [rng.randint(-9, 9) for _ in range(3)]
    ys = []
    for t in range(3, 3 + length):
        v = expr.evaluate({"x[t-1]": w[-1], "x[t-2]": w[-2], "x[t-3]": w[-3], "t": t})
        if v is None:
            break
        w.append(v)
        ys.append(v)
    return ys


class SequenceAdapter:
    kind = "sequence"
    focusable = True  # each candidate is an independent learning problem

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
        spec = self.spec = sandbox.call("spec")
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


class PixelLabAdapter(LabAdapter):
    """Controlled experiments seen through pixels (phase 5).

    The query inputs are the motor command the agent issued (an efference copy, as in :class:`LabAdapter`);
    the outcome is *read* from the "after" image by the agent's calibrated :class:`~genesis.vision.VisualReader`.
    An unreadable panel yields no target for that property, so nothing is learned from it. The "before" image
    shows the object exactly as the agent's motors set it: it is first read (a check of perception, counted in
    ``perception_checks``) and then used to refine the readout calibration.
    """

    kind = "lab"

    def __init__(self, agent, sandbox, n_random: int = 30):
        super().__init__(agent, sandbox, n_random)
        self.panels = self.spec["image"]["panels"]
        self.reader = agent.visual_reader(self.env_id)
        if self.reader is None:
            raise RuntimeError(f"no calibrated vision for {self.env_id}: call Genesis.calibrate_vision first")
        self.perception_checks = {"read": 0, "agreed": 0, "abstained": 0}
        self.abstentions = {"unreadable_situations": 0, "unreadable_outcomes": 0}

    def read(self, image: dict) -> tuple:
        from .vision import panel_patches
        values, glances = {}, {}
        for p in self.props:
            glances[p] = self.reader.glance(*panel_patches(image, self.panels[p]))
            values[p], _ = self.reader.read(glances[p])
        return values, glances

    def _outcome(self, action: str, state: dict, after: dict, digests: list) -> Outcome:
        nxt, _ = self.read(after)
        unread = [p for p in self.props if nxt[p] is None]
        self.abstentions["unreadable_outcomes"] += len(unread)
        return Outcome({self.key(action, p): nxt[p] for p in self.props},
                       {"state": state, "next": nxt, "unreadable": unread, "images_sha256": digests})

    def execute(self, sit: Situation) -> Outcome:
        from .vision import image_digest
        state, action = sit.action["state"], sit.action["experiment"]
        obs = self.sb.call("experiment", state, action)
        self.history.append((dict(state), action))
        seen, glances = self.read(obs["before"])
        for p in self.props:
            self.perception_checks["read"] += 1
            if seen[p] is None:
                self.perception_checks["abstained"] += 1
            else:
                self.perception_checks["agreed"] += seen[p] == state[p]
            self.reader.calibrate(glances[p], state[p])
        return self._outcome(action, state, obs["after"], [image_digest(obs["before"]), image_digest(obs["after"])])


class PixelPassiveAdapter(PixelLabAdapter):
    """Watching the pixel world: both states must be read; nothing calibrates the readout here."""

    def candidates(self) -> list:
        from .vision import image_digest
        obs = self.sb.call("passive_observation")
        state, _ = self.read(obs["before"])
        action = obs["action"]
        self._pending = (obs["after"], [image_digest(obs["before"]), image_digest(obs["after"])])
        if any(v is None for v in state.values()):
            # Unreadable situation: remembered as an episode, but it poses no learning problem.
            self.abstentions["unreadable_situations"] += 1
            return [Situation({"experiment": action, "state": state, "design": "passive"}, [], "observation",
                              f"{self.env_id}/{action}")]
        sit = self._situation(state, action, "passive", provenance="observation")
        return [sit]

    def execute(self, sit: Situation) -> Outcome:
        after, digests = self._pending
        self._pending = None
        if not sit.queries:
            return Outcome({}, {"state": sit.action["state"], "next": None, "unreadable": "before",
                                "images_sha256": digests})
        return self._outcome(sit.action["experiment"], sit.action["state"], after, digests)
