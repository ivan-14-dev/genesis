"""Induce flat linguistic constructions from utterance/meaning demonstrations.

Token alignment and variable substitution are innate. Surface frames, slot
order and negation assignments are learned, not a supplied FR/EN grammar.
Only object descriptions can be expressed; this is not general syntax learning.
"""
from __future__ import annotations

import json
import re

from .language import fold

MAX_EXAMPLES = 256
MAX_FORMS = 64
MAX_TOKENS = 24
CHECKS = 3


def manifest():
    return {"version": 1, "constructions_at_birth": [],
            "mechanisms": ["word alignment", "slot substitution", "prospective validation",
                           "conjunction of signed lexical predicates"],
            "limits": [MAX_EXAMPLES, MAX_FORMS, MAX_TOKENS, CHECKS]}


def tokens(text):
    text = fold(text).strip().rstrip(".!?")
    parts = text.split()
    if not 1 <= len(parts) <= MAX_TOKENS or any(
            not re.fullmatch(r"[^\W\d_]{1,32}", p) for p in parts):
        raise ValueError("use 1-24 alphabetic tokens in a construction")
    return parts


def canonical(terms):
    return sorted([[word, negative] for word, negative in terms])


class ConstructionLearner:
    def __init__(self, lexicon):
        self.lexicon, self.ltm = lexicon, lexicon.ltm
        self.key = f"constructions/{lexicon.g.env_id}"
        self.state = self.ltm.get_meta(self.key) or {"examples": [], "validation": {}}

    def examples(self, key):
        return [e for e in self.state["examples"] if e["active"] and e["key"] == key]

    def forms(self):
        return sorted({e["key"] for e in self.state["examples"] if e["active"]})

    def meaning(self, key):
        meanings = {tuple(e["negative"]) for e in self.examples(key)}
        return list(next(iter(meanings))) if len(meanings) == 1 else None

    def validated(self, key):
        v = self.state["validation"].get(key, {})
        examples = self.examples(key)
        meaning = self.meaning(key)
        return (meaning is not None and v.get("meaning") == meaning and len(v.get("episodes", [])) >= CHECKS
                and all(len({e["bindings"][i] for e in examples}) >= 2 for i in range(len(meaning))))

    def parse(self, text):
        try:
            parts = tokens(text)
        except ValueError:
            return None  # Not a supported construction; the ordinary dialogue can still handle it.
        known = {e["word"] for e in self.lexicon.examples()}
        matches = []
        for key in self.forms():
            pattern = json.loads(key)
            if len(pattern) != len(parts):
                continue
            bindings = []
            for token, part in zip(pattern, parts):
                if isinstance(token, int):
                    if part not in known:
                        break
                    bindings.append(part)
                elif token != part:
                    break
            else:
                negative = self.meaning(key)
                terms = canonical(zip(bindings, negative)) if negative is not None else None
                matches.append((key, terms))
        if not matches:
            return None
        meanings = {json.dumps(terms) for _, terms in matches}
        terms = matches[0][1] if len(meanings) == 1 and None not in [t for _, t in matches] else None
        validated = terms is not None and all(self.validated(key) for key, _ in matches)
        return {"terms": terms, "validated": validated,
                "status": "inconsistent" if terms is None else "validated" if validated else "candidate",
                "forms": [key for key, _ in matches]}

    def teach(self, text, terms, correction=False, source="user-construction-teaching"):
        if source not in ("user-construction-teaching", "simulated-construction-teaching"):
            raise ValueError("unknown construction provenance")
        parts = tokens(text)
        if not 1 <= len(terms) <= 4 or any(type(n) is not bool for _, n in terms):
            raise ValueError("one to four signed word meanings are required")
        terms = canonical((fold(w), n) for w, n in terms)
        words = [w for w, _ in terms]
        known = {e["word"] for e in self.lexicon.examples()}
        if len(set(words)) != len(words) or any(w not in known or parts.count(w) != 1 for w in words):
            raise ValueError("each distinct target word must be learned and occur exactly once")
        bindings = [p for p in parts if p in words]
        pattern = [bindings.index(p) if p in words else p for p in parts]
        if not any(isinstance(p, str) for p in pattern):
            raise ValueError("a construction needs at least one literal token")
        key = json.dumps(pattern)
        keys = {e["key"] for e in self.state["examples"]}
        if len(self.state["examples"]) >= MAX_EXAMPLES or key not in keys and len(keys) >= MAX_FORMS:
            raise ValueError("construction memory budget exceeded")
        prior = self.parse(text)
        previous = self.examples(key)
        novel = bindings not in [e["bindings"] for e in previous]
        episode = self.ltm.record_episode(
            self.lexicon.g.agent.session, self.lexicon.g.env_id, self.key, source,
            action={"correction": correction}, observation={"tokens": parts},
            outcome={"terms": terms}, predictions={"terms": prior and prior["terms"], "novel_binding": novel})
        if correction:
            for e in previous:
                if e["bindings"] == bindings:
                    e["active"] = False
        self.state["examples"].append({"key": key, "bindings": bindings,
                                       "negative": [dict(terms)[w] for w in bindings],
                                       "active": True, "episode": episode})
        meaning = self.meaning(key)
        v = self.state["validation"].get(key, {})
        if correction or v.get("meaning") != meaning or meaning is None:
            v = {"meaning": meaning, "episodes": []}
        elif novel and prior and prior["terms"] == terms:
            v["episodes"].append(episode)
        self.state["validation"][key] = v
        status = "inconsistent" if meaning is None else "validated" if self.validated(key) else "candidate"
        statement = json.dumps({"pattern": pattern, "negative": meaning}, sort_keys=True)
        for item in self.ltm.knowledge(key=f"{self.key}/{key}"):
            if item.statement != statement and item.status != "refuted":
                self.ltm.set_status(item.id, "refuted", "construction interpretation changed")
                self.ltm.add_evidence(item.id, [episode], "refuted_by")
        kid = self.ltm.upsert_knowledge(
            f"{self.key}/{key}", "construction", statement,
            {"pattern": pattern, "negative": meaning}, 1.0 if v["episodes"] else 0.0, status,
            "external utterance/meaning demonstrations", self.lexicon.g.env_id,
            n_support=len(self.examples(key)), forward_ok=len(v["episodes"]),
            reason="annotation agreement on novel bindings, not general language understanding")
        self.ltm.add_evidence(kid, [e["episode"] for e in self.examples(key)], "induced_from")
        self.ltm.add_evidence(kid, v["episodes"], "supports")
        self.ltm.set_meta(self.key, self.state)
        self.ltm.commit()
        return {"knowledge_id": kid, "episode_id": episode, "status": status,
                "validated": self.validated(key), "forward_ok": len(v["episodes"]), "pattern": pattern}
