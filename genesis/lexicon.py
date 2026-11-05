"""Learned vocabulary: which words signal which action, acquired from the user's teaching.

GENESIS has no language model. What is **innate** here is only: splitting text into letter tokens (accents
folded), a fixed set of actions the chat can perform (the *intents*), and the designer-provided interface words
(yes/no and the command keywords in ``chat.py``). Everything else, i.e. what *your* words mean, is
**acquired**:

* When a sentence is not understood, the user answers with the command they meant. Every word of the sentence
  is then associated with that intent (cross-situational word learning, Siskind 1996; Yu & Smith 2007).
* Each word predicts an intent ``P(i | w) = (s(w,i) + α) / (S(w) + c(w,i) + αK)``, where ``s`` counts
  positive episodes, ``c`` explicit rejections and ``K`` the number of intents. Words seen with many intents
  ("tu", "the") therefore carry little weight without any stop-word list. A sentence is interpreted by
  naive-Bayes combination of its known words.
* As with rules, a word→intent mapping is only **validated** by forward predictions: the word predicted the
  intent *before* the user confirmed it. Unvalidated interpretations are proposed as questions ("do you mean
  `stage`?"), never executed silently. Repeated wrong predictions refute the mapping.

Each mapping is a knowledge item (``kind="lexical"``, provenance ``user-taught``). Each teaching, confirmation
or rejection is an episode (env ``language``, provenance ``user-provided``) linked as evidence, so
``explain`` shows why a word is believed to mean something.

This is word→action association, not language understanding. There is no syntax, no compositional meaning,
and no knowledge of words never taught.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Optional

ALPHA = 0.1
VALIDATE_OK = 2           # correct forward predictions needed to validate a word→intent mapping
ACT_POSTERIOR = 0.9       # act without asking only above this and with a validated word
ASK_POSTERIOR = 0.4       # below this the sentence is treated as not understood
MAX_TOKENS = 12
MAX_WORD_LEN = 30
MAX_VOCAB = 5000


def tokens(text: str) -> list:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    out = []
    for w in re.findall(r"[a-z]+", t):
        if len(w) <= MAX_WORD_LEN and w not in out:
            out.append(w)
    return out[:MAX_TOKENS]


@dataclass
class Interpretation:
    intent: str
    posterior: float
    words: list           # known words that contributed
    validated: bool       # at least one of them has a validated mapping to ``intent``
    kids: list            # knowledge ids of the contributing mappings

    @property
    def act(self) -> bool:
        return self.validated and self.posterior >= ACT_POSTERIOR


class Lexicon:
    def __init__(self, ltm, intents: tuple, session: Callable[[], int], ignore: frozenset = frozenset()):
        self.ltm = ltm
        self.ignore = set(ignore)
        self.intents = tuple(intents)
        self.session = session
        self.stats: dict = {}   # word -> intent -> {id, s, c, ok, bad, status}
        for k in ltm.knowledge(kind="lexical"):
            w, i = k.expr["word"], k.expr["intent"]
            self.stats.setdefault(w, {})[i] = {"id": k.id, "s": k.n_support, "c": k.n_contra,
                                               "ok": k.forward_ok, "bad": k.forward_bad, "status": k.status}

    def words(self, text: str) -> list:
        return [w for w in tokens(text) if w not in self.ignore]

    # --------------------------------------------------------------- model
    def p(self, w: str, i: str) -> float:
        d = self.stats.get(w, {})
        e = d.get(i, {})
        total = sum(x["s"] for x in d.values())
        return (e.get("s", 0) + ALPHA) / (total + e.get("c", 0) + ALPHA * len(self.intents))

    def known(self, w: str) -> bool:
        return any(x["s"] > 0 for x in self.stats.get(w, {}).values())

    def predicted(self, w: str) -> Optional[str]:
        seen = [i for i in self.intents if self.stats.get(w, {}).get(i, {}).get("s", 0) > 0]
        if not seen:
            return None
        return max(seen, key=lambda i: (self.p(w, i), -self.intents.index(i)))

    def interpret(self, text: str) -> Optional[Interpretation]:
        known = [w for w in self.words(text) if self.known(w)]
        if not known:
            return None
        logs = {i: sum(math.log(self.p(w, i)) for w in known) for i in self.intents}
        m = max(logs.values())
        z = sum(math.exp(v - m) for v in logs.values())
        top = max(self.intents, key=lambda i: (logs[i], -self.intents.index(i)))
        entries = [self.stats[w].get(top) for w in known]
        return Interpretation(top, math.exp(logs[top] - m) / z, known,
                              any(e and e["status"] == "validated" for e in entries),
                              [e["id"] for e in entries if e])

    @staticmethod
    def _status(e: dict) -> str:
        if e["bad"] >= 2 and e["bad"] > e["ok"]:
            return "refuted"
        if e["ok"] >= VALIDATE_OK and e["ok"] >= 3 * e["bad"]:
            return "validated"
        return "candidate"

    # ------------------------------------------------------------ learning
    def learn(self, text: str, intent: str, how: str, posterior: Optional[float] = None) -> dict:
        """Update from one episode. ``how`` is ``taught`` (the user named the intended command after a
        sentence was not understood), ``confirmed`` (yes to a proposed interpretation) or ``rejected``
        (no to a proposed or executed interpretation)."""
        assert intent in self.intents and how in ("taught", "confirmed", "rejected")
        words = self.words(text)
        before = {w: self.predicted(w) for w in words}
        ep = self.ltm.record_episode(
            self.session(), "language", "lexicon", "user-provided", action={"how": how},
            observation={"utterance": text[:200], "words": words},
            outcome={"intent": intent, "accepted": how != "rejected"},
            predictions={"words": before, "posterior": posterior},
            surprise=None if posterior is None else (1 - posterior if how != "rejected" else posterior))
        touched, new_words, changes = set(), [], []
        for w in words:
            if how == "rejected":
                e = self.stats.get(w, {}).get(intent)
                if e is None:
                    continue
                e["c"] += 1
                if before[w] == intent:
                    e["bad"] += 1
                touched.add(w)
                continue
            if w not in self.stats and len(self.stats) >= MAX_VOCAB:
                continue
            if not self.known(w):
                new_words.append(w)
            e = self.stats.setdefault(w, {}).setdefault(intent, {"id": None, "s": 0, "c": 0, "ok": 0, "bad": 0,
                                                                  "status": "candidate"})
            e["s"] += 1
            if how == "confirmed":  # only predictions actually put to the user are scored
                if before[w] == intent:
                    e["ok"] += 1
                elif before[w] is not None:
                    self.stats[w][before[w]]["bad"] += 1
            touched.add(w)
        for w in touched:
            for i, e in self.stats[w].items():
                old, e["status"] = e["status"], self._status(e)
                rel = ("contradicts" if how == "rejected" and i == intent else
                       "supports" if i == intent and how != "rejected" else
                       "contradicts" if how == "confirmed" and before[w] == i else None)
                created = e["id"] is None
                e["id"] = self.ltm.upsert_knowledge(
                    f"lexicon/{w}", "lexical", f"« {w} » → {i}", {"word": w, "intent": i}, self.p(w, i),
                    e["status"], "user-taught", "language", e["s"], e["c"], e["ok"], e["bad"],
                    reason=f"{how} ({e['ok']} correct / {e['bad']} wrong forward predictions)")
                if rel:
                    self.ltm.add_evidence(e["id"], [ep], "induced_from" if created else rel)
                if old != e["status"] and not created:
                    changes.append((w, i, e["status"]))
        self.ltm.commit()
        return {"episode": ep, "words": words, "new_words": new_words, "changes": changes}

    # ------------------------------------------------------------ inspection
    def entries(self) -> list:
        out = []
        for w, d in self.stats.items():
            for i, e in d.items():
                if e["s"] > 0:
                    out.append({"word": w, "intent": i, "p": self.p(w, i), **e})
        return out

    def size(self) -> int:
        return sum(1 for w in self.stats if self.known(w))
