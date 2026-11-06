"""Disposable structured objects with hidden, seed-sampled action effects.

Sensor names and action tokens carry no semantics. Each experiment acts on a
fresh specimen, so changing one channel is a genuine simulator intervention.
This is a categorical toy world, not a physical simulation.
"""
import itertools
import random


class ObjectWorld:
    AGENT_METHODS = ("spec", "experiment")

    def __init__(self, seed):
        self.env_id = f"object-world-{seed}"
        self.features = [f"s{i}" for i in range(5)]
        self.actions = ["a0", "a1"]
        rng = random.Random(seed)
        order = rng.sample(self.features, len(self.features))
        self._rules = {
            "a0": {"channels": order[:1], "flip": rng.randrange(2)},
            "a1": {"channels": order[1:3], "flip": rng.randrange(2)},
        }

    def spec(self):
        return {
            "features": list(self.features), "actions": list(self.actions),
            "objects": [{"id": f"specimen-{i}", "features": dict(zip(self.features, values))}
                        for i, values in enumerate(itertools.product((0, 1), repeat=len(self.features)))],
        }

    def _true_next(self, features, action):
        rule = self._rules[action]
        effect = int(all(features[c] == 1 for c in rule["channels"]))
        return effect ^ rule["flip"]

    def experiment(self, object_id, features, action):
        if action not in self.actions:
            raise ValueError("unknown object action")
        if set(features) != set(self.features) or any(type(v) is not int or v not in (0, 1, 2)
                                                     for v in features.values()):
            raise ValueError("invalid object sensors")
        if not isinstance(object_id, str) or not object_id:
            raise ValueError("object ID required")
        return {"object_id": object_id, "features": dict(features),
                "action": action, "effect": self._true_next(features, action)}

    def _heldout(self, seed, n=64):
        rng = random.Random(seed)
        causal = {c for rule in self._rules.values() for c in rule["channels"]}
        return [{"id": f"unseen-{seed}-{i}",
                 "features": {c: rng.randrange(2) if c in causal else 2 for c in self.features}}
                for i in range(n)]

    def _change(self, action):
        self._rules[action]["flip"] ^= 1
