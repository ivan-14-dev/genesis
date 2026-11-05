"""Stage 6 — autonomous exploration across several environments.

The agent is given a portfolio of sandboxed worlds and a single interaction
budget. Nobody tells it which world is worth studying. It decides, chunk by
chunk, where to spend its attention, using only signals it can observe:

* **Learning progress** (Oudeyer & Kaplan's intrinsic motivation): how much its
  prediction accuracy changed and how many open questions it resolved per
  interaction during its recent chunks in that world. Worlds where nothing
  improves (e.g. pure noise) lose priority even though they stay surprising —
  which is exactly the "noisy TV" trap that pure surprise-seeking falls into.
* **Remaining opportunity**: the best curiosity utility still available there
  (``Genesis.peek``). A world whose utility has collapsed is *exhausted*.
* **Untried worlds** get a fixed number of probing chunks before being judged.

It also turns its knowledge gaps into explicit, persistent **learning
objectives** (one per open question) and marks them achieved or abandoned.

Limitations: chunk granularity is fixed; progress is a short-horizon estimate,
so a world that only pays off after a long plateau can be under-explored.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .curiosity import Curiosity


@dataclass
class WorldRecord:
    name: str
    adapter: object
    chunks: list = field(default_factory=list)  # (steps, accuracy_delta, resolved) of chunks that made predictions
    steps: int = 0
    last_acc: float | None = None
    exhausted: bool = False


class Explorer:
    def __init__(self, agent, adapters: dict, chunk: int = 10, probe_chunks: int = 2, window: int = 2,
                 exhausted_utility: float = 0.01, optimism: float = 0.1, min_progress: float = 0.05):
        self.agent = agent
        self.worlds = {name: WorldRecord(name, a) for name, a in adapters.items()}
        self.chunk = chunk
        self.probe_chunks = probe_chunks
        self.window = window
        self.exhausted_utility = exhausted_utility
        self.min_progress = min_progress
        self.optimism = optimism
        self.log: list = []

    # ----------------------------------------------------------- signals
    def _resolved(self, env_id: str) -> int:
        # Only rules count: concluding "this is unpredictable" closes a question but teaches nothing usable.
        return sum(1 for k, lr in self.agent.learners.items()
                   if k.startswith(env_id + "/") and lr.status() == "validated")

    def progress(self, w: WorldRecord) -> float:
        recent = w.chunks[-self.window:]
        steps = sum(c[0] for c in recent) or 1
        return sum(abs(c[1]) * c[0] + c[2] for c in recent) / steps

    def scores(self) -> dict:
        out = {}
        for name, w in self.worlds.items():
            if w.exhausted:
                continue
            # Unprobed worlds first; afterwards measured progress plus an optimism bonus that shrinks
            # with evidence, so an unlucky early sample does not condemn a world forever.
            out[name] = float("inf") if len(w.chunks) < self.probe_chunks else \
                self.progress(w) + self.optimism / math.sqrt(len(w.chunks))
        return out

    # ---------------------------------------------------------- objectives
    def update_objectives(self) -> None:
        ltm, learners = self.agent.ltm, self.agent.learners
        for q in Curiosity.knowledge_gaps(learners, max_questions=10 ** 6):
            ltm.add_objective(self.agent.session, q.key.split("/")[0], q.key, q.text)
        for o in ltm.objectives("open"):
            lr = learners.get(o["key"])
            if lr is None:
                continue
            st = lr.status()
            if st == "validated":
                top, mass, _ = lr.best()
                ltm.update_objective(o["key"], "achieved", {"belief": str(top.expr), "confidence": round(mass, 4)})
            elif st == "unpredictable":
                ltm.update_objective(o["key"], "abandoned", {"reason": "judged unpredictable from evidence"})

    # --------------------------------------------------------------- loop
    def choose(self, strategy: str, rng: random.Random, turn: int) -> str | None:
        live = [n for n, w in self.worlds.items() if not w.exhausted]
        if not live:
            return None
        if strategy == "random":
            return rng.choice(live)
        if strategy == "round_robin":
            return live[turn % len(live)]
        scores = self.scores()
        best = max(scores.values())
        if best <= self.min_progress:
            return None  # every world has been probed and none shows measurable learning progress
        return min((n for n in live if scores[n] == best), key=lambda n: self.worlds[n].steps)

    def run(self, budget: int, strategy: str = "progress") -> dict:
        rng = random.Random(self.agent.seed + 6)
        spent = turn = 0
        stopped = "budget spent"
        while spent < budget:
            name = self.choose(strategy, rng, turn)
            if name is None:
                stopped = ("no world shows measurable learning progress"
                           if any(not w.exhausted for w in self.worlds.values()) else "all worlds exhausted")
                break
            w = self.worlds[name]
            env_id = w.adapter.env_id
            before = self._resolved(env_id)
            rep = self.agent.live(w.adapter, min(self.chunk, budget - spent), "curiosity", stop_when_idle=True)
            acc = sum(rep.accuracy) / len(rep.accuracy) if rep.accuracy else None
            delta = 0.0 if acc is None or w.last_acc is None else acc - w.last_acc
            if acc is not None:
                w.last_acc = acc
            resolved = self._resolved(env_id) - before
            if acc is not None or resolved:
                w.chunks.append((max(rep.steps, 1), delta, resolved))
            w.steps += rep.steps
            spent += rep.steps
            if rep.steps == 0 or self.agent.peek(w.adapter) < self.exhausted_utility \
                    or rep.stopped_because == "sandbox budget exhausted":
                w.exhausted = True
            self.update_objectives()
            self.log.append({"turn": turn, "world": name, "steps": rep.steps, "accuracy": acc,
                             "resolved": resolved, "exhausted": w.exhausted})
            turn += 1
        return {"strategy": strategy, "interactions": spent, "stopped_because": stopped,
                "allocation": {n: w.steps for n, w in self.worlds.items()},
                "exhausted": [n for n, w in self.worlds.items() if w.exhausted], "turns": len(self.log)}
