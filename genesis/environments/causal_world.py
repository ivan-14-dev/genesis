"""Stage-2/3 world: objects with properties, opaque actions and hidden causal rules.

* Properties ``p0..p{K-1}`` hold small integers; actions ``a0..a{A-1}`` are opaque tokens.
* Each action changes one property according to a hidden rule sampled from the
  seed: add a constant, set a constant, copy another property or add another
  property, optionally gated by a condition ``p_q == v``.
* Passive observation shows "natural" states in which a distractor property is
  always equal to the true condition property (a confounder). Only controlled
  experiments can tell the cause from the correlate.
* Goal tasks reward the agent for bringing a property to a target value.
"""

from __future__ import annotations

import random

VALUE_MIN, VALUE_MAX = -50, 50


class CausalWorld:
    AGENT_METHODS = ("spec", "passive_observation", "experiment", "begin_task", "act")

    def __init__(self, seed: int, n_props: int = 4, n_values: int = 6, n_actions: int = 4, noise: float = 0.0):
        self.env_id = f"causal-world-{seed}"
        self._rng = random.Random(seed)
        self._seed = seed
        self.n_values = n_values
        self.noise = noise
        self.props = [f"p{i}" for i in range(n_props)]
        self.actions = [f"a{i}" for i in range(n_actions)]
        rng = self._rng
        targets = rng.sample(self.props, n_actions) if n_actions <= n_props else [rng.choice(self.props) for _ in
                                                                                    range(n_actions)]
        self._rules = {}
        conditional_forced = rng.randrange(n_actions)
        for i, (a, tgt) in enumerate(zip(self.actions, targets)):
            others = [p for p in self.props if p != tgt]
            kind = rng.choice(["add", "set", "copy", "sum"])
            rule = {"target": tgt, "kind": kind}
            if kind == "add":
                rule["c"] = rng.choice([1, 2, 3, -1, -2])
            elif kind == "set":
                rule["c"] = rng.randrange(n_values)
            else:
                rule["src"] = rng.choice(others)
            if i == conditional_forced or rng.random() < 0.4:
                rule["cond"] = (rng.choice(others), rng.randrange(n_values))
            self._rules[a] = rule
        cond_rule = self._rules[self.actions[conditional_forced]]
        q = cond_rule["cond"][0]
        candidates = [p for p in self.props if p not in (q, cond_rule["target"])]
        self._confound = (q, rng.choice(candidates))  # natural states: confound[1] mirrors confound[0]
        self._task_state = None
        self._task_goal = None
        self._task_steps = 0

    # ------------------------------------------------------------------ physics
    def _apply(self, state: dict, action: str, rng: random.Random | None = None) -> dict:
        r = self._rules[action]
        nxt = dict(state)
        if "cond" in r and state[r["cond"][0]] != r["cond"][1]:
            return nxt
        t = r["target"]
        if r["kind"] == "add":
            v = state[t] + r["c"]
        elif r["kind"] == "set":
            v = r["c"]
        elif r["kind"] == "copy":
            v = state[r["src"]]
        else:
            v = state[t] + state[r["src"]]
        if rng is not None and self.noise and rng.random() < self.noise:
            v += rng.choice([-3, -2, -1, 1, 2, 3])
        nxt[t] = max(VALUE_MIN, min(VALUE_MAX, v))
        return nxt

    def _validate(self, state: dict) -> dict:
        if set(state) != set(self.props):
            raise ValueError("state must assign every property")
        return {p: max(VALUE_MIN, min(VALUE_MAX, int(state[p]))) for p in self.props}

    # ----------------------------------------------------------- agent-visible
    def spec(self) -> dict:
        return {"properties": list(self.props), "actions": list(self.actions),
                "settable_range": [0, self.n_values - 1]}

    def passive_observation(self) -> tuple:
        s = {p: self._rng.randrange(self.n_values) for p in self.props}
        s[self._confound[1]] = s[self._confound[0]]
        a = self._rng.choice(self.actions)
        return dict(s), a, self._apply(s, a, self._rng)

    def experiment(self, state: dict, action: str) -> dict:
        if action not in self._rules:
            raise ValueError(f"unknown action {action!r}")
        return self._apply(self._validate(state), action, self._rng)

    def begin_task(self, task_id: int) -> dict:
        rng = random.Random(self._seed * 100003 + task_id)
        while True:
            start = {p: rng.randrange(self.n_values) for p in self.props}
            s = dict(start)
            for _ in range(rng.randint(2, 4)):
                s = self._apply(s, rng.choice(self.actions))
            changed = [p for p in self.props if s[p] != start[p]]
            if changed:
                g = rng.choice(changed)
                break
        self._task_state, self._task_goal, self._task_steps = dict(start), (g, s[g]), 0
        return {"state": dict(start), "goal": [g, s[g]]}

    def act(self, action: str) -> dict:
        if self._task_state is None:
            raise RuntimeError("no active task")
        self._task_state = self._apply(self._task_state, action, self._rng)
        self._task_steps += 1
        g, v = self._task_goal
        reached = self._task_state[g] == v
        done = reached or self._task_steps >= 8
        return {"state": dict(self._task_state), "reward": 1.0 if reached else 0.0, "done": done}

    # ------------------------------------------------------------ harness-only
    def _true_next(self, state: dict, action: str) -> dict:
        return self._apply(dict(state), action)

    def _describe_rules(self) -> dict:
        out = {}
        for a, r in self._rules.items():
            t = r["target"]
            eff = {"add": lambda: f"{t} + {r['c']}", "set": lambda: f"{r.get('c')}", "copy": lambda: f"{r.get('src')}",
                   "sum": lambda: f"{t} + {r.get('src')}"}[r["kind"]]()
            out[a] = (f"if {r['cond'][0]} == {r['cond'][1]}: " if "cond" in r else "") + f"{t} := {eff}"
        return out
