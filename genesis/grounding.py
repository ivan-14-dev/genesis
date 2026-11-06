"""Learn task-relative object concepts by minimal predictive feature abstraction.

No object laws or feature relevance are supplied here. The innate bias is a
bounded search for deterministic categorical projections. Episodes, model
revisions and future validation evidence remain in the ordinary GENESIS memory.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections import Counter

from .environments.base import BudgetExceeded

MAX_FEATURES = 8
MAX_ACTIONS = 4
MAX_OBJECTS = 64
MAX_EPISODES = 512
VALIDATION_K = 4


def manifest():
    from .grounded_language import language_manifest
    body = {
        "version": 1,
        "mechanisms": ["minimal consistent sensor projection", "predictive equivalence classes",
                       "disagreement/novelty/one-channel experiment selection",
                       "novel-input forward validation", "deterministic conflict revision"],
        "limits": [MAX_FEATURES, MAX_ACTIONS, MAX_OBJECTS, MAX_EPISODES, VALIDATION_K],
        "domain_knowledge": [], "pretrained_models": [],
        "grounded_language": language_manifest(),
    }
    return {**body, "digest": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()}


def register(ltm, session):
    current, old = manifest(), ltm.get_meta("innate_grounding")
    if old is None or old["digest"] != current["digest"]:
        history = ltm.get_meta("innate_grounding_history") or []
        history.append({"session": session, "ts": time.time(),
                        "from": old and old["digest"], "to": current["digest"]})
        ltm.set_meta("innate_grounding_history", history)
        ltm.set_meta("innate_grounding", current)


class Projection:
    def __init__(self, features, rows):
        self.features = list(features)
        self.rows = rows
        self.models = []
        if not rows:
            return
        for size in range(len(features) + 1):
            for mask in itertools.combinations(features, size):
                table = {}
                for row in rows:
                    key = tuple(row["features"][f] for f in mask)
                    if key in table and table[key] != row["effect"]:
                        break
                    table[key] = row["effect"]
                else:
                    self.models.append((mask, table))
            if self.models:
                break

    def predict(self, features):
        votes = [table.get(tuple(features[f] for f in mask)) for mask, table in self.models]
        known = [v for v in votes if v is not None]
        value = known[0] if known and len(known) == len(votes) and len(set(known)) == 1 else None
        disagreement = 1 - max(Counter(known).values(), default=0) / max(len(votes), 1)
        return {"value": value, "uncertainty": disagreement,
                "alternatives": len(self.models)}

    def description(self):
        if len(self.models) != 1:
            return None
        mask, table = self.models[0]
        return {"features": list(mask),
                "groups": [{"values": list(k), "effect": v} for k, v in sorted(table.items())]}

    def contrasts(self):
        """Episodes witnessing a changed effect when exactly one sensor changes."""
        pairs = {}
        for i, a in enumerate(self.rows):
            for b in self.rows[:i]:
                if a["effect"] == b["effect"]:
                    continue
                changed = [f for f in self.features if a["features"][f] != b["features"][f]]
                if len(changed) == 1:
                    pairs.setdefault(changed[0], [b["episode"], a["episode"]])
        return pairs


class GroundedExplorer:
    def __init__(self, agent, sandbox):
        self.agent, self.ltm, self.sb = agent, agent.ltm, sandbox
        self.env_id = sandbox.env_id
        spec = sandbox.call("spec")
        self.features = sorted(spec["features"])
        self.actions = sorted(spec["actions"])
        self.objects = spec["objects"]
        if not (0 < len(self.features) <= MAX_FEATURES and 0 < len(self.actions) <= MAX_ACTIONS
                and 0 < len(self.objects) <= MAX_OBJECTS):
            raise ValueError("grounded experiment exceeds fixed representation bounds")
        if len(set(self.features)) != len(self.features) or len(set(self.actions)) != len(self.actions):
            raise ValueError("duplicate sensors or actions")
        for obj in self.objects:
            if not isinstance(obj["id"], str) or set(obj["features"]) != set(self.features) \
                    or any(type(v) is not int for v in obj["features"].values()):
                raise ValueError("invalid grounded object")
        self.key = f"grounding/{self.env_id}"
        stored = self.ltm.get_meta(self.key)
        schema = {"features": self.features, "actions": self.actions}
        if stored and stored["schema"] != schema:
            raise ValueError("grounded sensor/action schema changed")
        self.state = stored or {"schema": schema, "rows": [], "revisions": {a: 0 for a in self.actions},
                                "validation": {}, "last": None}
        self.models = {a: Projection(self.features, self.rows(a)) for a in self.actions}
        register(self.ltm, agent.session)

    def rows(self, action):
        return [r for r in self.state["rows"] if r["action"] == action
                and r["revision"] == self.state["revisions"][action]]

    def predict(self, features, action):
        if action not in self.actions or set(features) != set(self.features):
            raise ValueError("unknown sensors or action")
        return self.models[action].predict(features)

    def choose_action(self, features, target):
        return next((a for a in self.actions if self.predict(features, a)["value"] == target), None)

    def select(self):
        visits = Counter((r["action"], tuple(r["features"][f] for f in self.features))
                         for r in self.state["rows"]
                         if r["revision"] == self.state["revisions"][r["action"]])
        counts = Counter(r["action"] for r in self.state["rows"])
        last = self.state["last"]
        ranked = []
        for obj in self.objects:
            for action in self.actions:
                pred = self.predict(obj["features"], action)
                key = action, tuple(obj["features"][f] for f in self.features)
                contrast = bool(last and last["action"] == action and sum(
                    obj["features"][f] != last["features"][f] for f in self.features) == 1)
                score = (pred["uncertainty"] + 0.25 / (1 + visits[key])
                         + 0.1 * contrast + 0.05 / (1 + counts[action]))
                ranked.append((score, obj["id"], action, obj))
        _, _, action, obj = max(ranked, key=lambda x: x[:3])
        return obj, action

    def experience(self, obj, action):
        if len(self.state["rows"]) >= MAX_EPISODES:
            raise BudgetExceeded("grounded episode limit reached")
        if self.agent.interactions >= self.agent.safety.max_interactions_per_session * (self.agent.session + 1):
            raise BudgetExceeded("agent safety interaction limit reached")
        features = obj["features"]
        pred = self.predict(features, action)  # necessarily before the experiment
        old_model = self.models[action]
        old_description = old_model.description()
        old_rows = self.rows(action)
        novel = not any(r["features"] == features for r in old_rows)
        outcome = self.sb.call("experiment", obj["id"], features, action)
        effect = outcome["effect"]
        if type(effect) is not int:
            raise ValueError("grounded effects must be integers")
        changed = any(r["features"] == features and r["effect"] != effect for r in old_rows)
        last = self.state["last"]
        controlled = bool(last and last["action"] == action and sum(
            last["features"][f] != features[f] for f in self.features) == 1)
        ep = self.ltm.record_episode(
            self.agent.session, self.env_id, self.key, "experiment",
            action={"object_id": obj["id"], "action": action, "features": dict(features),
                    "design": "one-channel-change" if controlled else "specimen-test"},
            observation=outcome, outcome={"effect": effect},
            predictions={action: {"predicted": pred["value"], "actual": effect, "confidence": None,
                                  "uncertainty": pred["uncertainty"], "novel_input": novel}},
            surprise=float(pred["value"] is not None and pred["value"] != effect))
        if changed:
            self.state["revisions"][action] += 1
            for k in self.ltm.knowledge(key=f"{self.key}/{action}"):
                if k.status != "refuted":
                    self.ltm.set_status(k.id, "refuted", "incompatible repeated state: new model revision")
                    self.ltm.add_evidence(k.id, [ep], "refuted_by")
            for k in self.ltm.knowledge(kind="grounded-word"):
                if k.context == self.env_id and k.status not in ("refuted", "dormant"):
                    self.ltm.set_status(k.id, "dormant", "effect model revised; lexical revalidation required")
        row = {"episode": ep, "object_id": obj["id"], "features": dict(features), "action": action,
               "effect": effect, "revision": self.state["revisions"][action]}
        self.state["rows"].append(row)
        self.state["last"] = row
        model = self.models[action] = Projection(self.features, self.rows(action))
        description = model.description()
        validation = self.state["validation"].get(action, {"ok": [], "bad": []})
        if changed or description != old_description:
            validation = {"ok": [], "bad": []}
        elif novel and pred["value"] is not None:
            validation["ok" if pred["value"] == effect else "bad"].append(ep)
        self.state["validation"][action] = validation
        self._consolidate(action, description, validation, ep)
        self.ltm.set_meta(self.key, self.state)
        self.agent.interactions += 1
        self.ltm.set_meta("interactions", self.agent.interactions)
        self.ltm.commit()
        return {"episode": ep, "prediction": pred, "effect": effect, "revised": changed}

    def _consolidate(self, action, description, validation, ep):
        key = f"{self.key}/{action}"
        current = json.dumps({"revision": self.state["revisions"][action], "representation": description},
                             sort_keys=True)
        for k in self.ltm.knowledge(key=key):
            if k.statement == current or k.status == "refuted":
                continue
            row = self.state["last"]
            expr = k.expr
            values = [row["features"][f] for f in expr["features"]]
            old_effect = next((g["effect"] for g in expr["groups"] if g["values"] == values), None)
            contradicted = old_effect is not None and old_effect != row["effect"]
            self.ltm.set_status(k.id, "refuted" if contradicted else "dormant",
                                "prediction contradicted" if contradicted else "representation revised")
            if contradicted:
                self.ltm.add_evidence(k.id, [ep], "refuted_by")
        if description is None:
            return
        contrasts = self.models[action].contrasts()
        status = ("validated" if len(validation["ok"]) >= VALIDATION_K and not validation["bad"]
                  and all(f in contrasts for f in description["features"]) else "candidate")
        rows = self.rows(action)
        kid = self.ltm.upsert_knowledge(
            key, "grounded-concept", current, description,
            len(validation["ok"]) / max(len(validation["ok"]) + len(validation["bad"]), 1),
            status, "induced from controlled object experiments", self.env_id,
            n_support=len(rows), forward_ok=len(validation["ok"]), forward_bad=len(validation["bad"]),
            reason="minimal predictive abstraction with future checks and one-channel contrasts")
        self.ltm.add_evidence(kid, [r["episode"] for r in rows if r["episode"] not in validation["ok"]],
                              "induced_from")
        self.ltm.add_evidence(kid, validation["ok"], "supports")

    def run(self, steps, strategy="active"):
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if strategy not in ("active", "random"):
            raise ValueError("unknown grounded exploration strategy")
        report = {"steps": 0, "correct": 0, "abstained": 0, "revisions": 0, "stopped": "step limit"}
        for _ in range(steps):
            obj, action = (self.select() if strategy == "active" else
                           (self.agent.rng.choice(self.objects), self.agent.rng.choice(self.actions)))
            try:
                result = self.experience(obj, action)
            except BudgetExceeded as exc:
                report["stopped"] = str(exc)
                break
            report["steps"] += 1
            report["abstained"] += result["prediction"]["value"] is None
            report["correct"] += result["prediction"]["value"] == result["effect"]
            report["revisions"] += result["revised"]
        self.ltm.record_metric(self.agent.session, "grounded_exploration", report["correct"], report)
        self.ltm.commit()
        return report
