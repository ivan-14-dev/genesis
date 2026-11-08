"""Experience-backed symbolic teaching, not pretrained world knowledge.

Only explicit class inclusion licenses inheritance. Instance membership and spatial
relations are never treated as universal rules. Answers use current premises, not
cached conclusions; their proofs are retained as episodes.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .language import fold


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    negative: bool = False

    @property
    def key(self):
        return "semantic/" + json.dumps(
            [self.subject, self.relation, self.object], ensure_ascii=True)

    @property
    def statement(self):
        return f"{self.subject} {'NOT ' if self.negative else ''}{self.relation} {self.object}"


def noun(text, plural=False):
    word = fold(text.strip())
    if plural:
        if word.endswith("aux"):
            word = word[:-3] + "al"
        elif word.endswith("s") and not word.endswith(("ss", "us")):
            word = word[:-1]
    return word


class SemanticMemory:
    MAX_FACTS = 512

    def __init__(self, agent):
        self.agent = agent
        self.ltm = agent.ltm

    def items(self):
        return self.ltm.knowledge(kind="semantic")

    def teach(self, facts, cid, text, correction=False):
        # Reject the entire message before recording anything if its budget is exceeded.
        existing = {(k.key, k.statement) for k in self.items()}
        if len(existing | {(f.key, f.statement) for f in facts}) > self.MAX_FACTS:
            raise ValueError("semantic memory budget exceeded")
        predictions = {}
        for i, fact in enumerate(facts):
            prior = self.answer(fact)
            predictions[str(i)] = {
                "predicted": True if prior["status"] == "supported"
                else False if prior["status"] == "opposed" else None,
                "actual": True, "confidence": None,
                "knowledge_ids": prior["knowledge_ids"],
                "meaning": "agreement with user assertion, not objective truth"}
        ep = self.ltm.record_episode(
            self.agent.session, "semantic", f"conversation-{cid}", "user-taught",
            action={"correction": correction}, observation={"text": text},
            outcome=[asdict(f) for f in facts], predictions=predictions)
        ids = []
        for fact in facts:
            peers = self.ltm.knowledge(key=fact.key)
            opposite = [k for k in peers if k.expr["negative"] != fact.negative
                        and k.status != "refuted"]
            if correction:
                for k in opposite:
                    self.ltm.set_status(k.id, "refuted", "explicit user correction")
                    self.ltm.add_evidence(k.id, [ep], "refuted_by")
                opposite = []
            old = next((k for k in peers if k.statement == fact.statement), None)
            kid = self.ltm.upsert_knowledge(
                fact.key, "semantic", fact.statement, asdict(fact),
                0.5, "disputed" if opposite else "taught", "user-taught",
                f"conversation-{cid}", n_support=(old.n_support if old else 0) + 1,
                n_contra=old.n_contra if old else 0,
                reason="explicit user correction" if correction else "user assertion, not verified")
            self.ltm.add_evidence(kid, [ep], "supports")
            for k in opposite:
                self.ltm.set_status(k.id, "disputed", "opposing user assertions")
                self.ltm.add_evidence(k.id, [ep], "contradicts")
                prior_episodes = [r[0] for r in self.ltm.db.execute(
                    "SELECT episode_id FROM evidence WHERE knowledge_id=? AND relation='supports'", (k.id,))]
                self.ltm.add_evidence(kid, prior_episodes, "contradicts")
            for updated in [kid] + [k.id for k in opposite]:
                self.ltm.db.execute(
                    "UPDATE knowledge SET n_contra=(SELECT COUNT(*) FROM evidence "
                    "WHERE knowledge_id=? AND relation='contradicts') WHERE id=?", (updated, updated))
            ids.append(kid)
        self.ltm.record_metric(self.agent.session, "semantic_teaching", len(facts),
                               {"episode": ep, "knowledge_ids": ids})
        return ids

    def answer(self, fact):
        items = [k for k in self.items() if k.status != "refuted"]
        positive, negative = {}, {}
        for k in items:
            f = Fact(**k.expr)
            if f.subject == fact.subject and f.relation == fact.relation:
                (negative if f.negative else positive)[f.object] = [k.id]
        # Cycles terminate because each class is visited once. No unrelated facts
        # follow from a contradiction, and a contradicted intermediate is not used.
        if fact.relation in ("instance", "subclass"):
            rules = [k for k in items if k.expr["relation"] == "subclass"
                     and not k.expr["negative"] and k.status != "disputed"]
            changed = True
            while changed:
                changed = False
                for k in rules:
                    f = Fact(**k.expr)
                    if f.subject in positive and f.subject not in negative and f.object not in positive:
                        positive[f.object] = positive[f.subject] + [k.id]
                        changed = True
        pos, neg = positive.get(fact.object), negative.get(fact.object)
        status = "conflict" if pos and neg else "supported" if pos else "opposed" if neg else "unknown"
        if fact.negative and status in ("supported", "opposed"):
            status = "opposed" if status == "supported" else "supported"
        ids = list(dict.fromkeys((pos or []) + (neg or [])))
        return {"status": status, "knowledge_ids": ids,
                "inferred": bool(pos and len(pos) > 1),
                "premises": [k.statement for i in ids for k in items if k.id == i]}

    def query(self, fact, cid, text):
        answer = self.answer(fact)
        ep = self.ltm.record_episode(
            self.agent.session, "semantic", f"conversation-{cid}", "inference",
            action={"query": asdict(fact), "text": text}, outcome=answer)
        return {**answer, "episode_id": ep}

    def retract(self, kid, cid, text):
        item = next((k for k in self.items() if k.id == kid), None)
        if item is None:
            raise ValueError("no semantic teaching with that ID")
        ep = self.ltm.record_episode(
            self.agent.session, "semantic", f"conversation-{cid}", "user-taught",
            action={"retract": kid}, observation={"text": text})
        self.ltm.set_status(kid, "refuted", "explicit user retraction")
        self.ltm.add_evidence(kid, [ep], "refuted_by")
        # A previous contradiction is resolved if only one active polarity remains.
        peers = [k for k in self.ltm.knowledge(key=item.key) if k.status != "refuted"]
        if len(peers) == 1:
            self.ltm.set_status(peers[0].id, "taught", "opposing assertion retracted")


# Deliberately bounded grammar: new nouns carry no preloaded meanings.
LABEL = r"[^\W\d_]+(?:-[^\W\d_]+)*"
PRONOUNS = {"il", "elle", "lui", "it", "he", "she"}
RESERVED = {"je", "tu", "vous", "nous", "on", "i", "you", "we", "who", "what",
            "qui", "quoi", "ce", "that", "this"}


def parse_fact(text):
    t = fold(text).replace("’", "'").strip().rstrip(".").strip()
    universal = re.fullmatch(
        rf"(?:tous les|toutes les|les|all) ({LABEL}) (?:sont des|sont les|are) ({LABEL})", t)
    if universal:
        return Fact(noun(universal[1], True), "subclass", noun(universal[2], True))
    m = re.fullmatch(
        rf"({LABEL}) (?:(n'est pas|est|is not|is) (un|une|a|an) ({LABEL})"
        rf"|(n'est pas dans|est dans|is not in|is in) ({LABEL}))", t)
    if not m or m[1] in RESERVED:
        return None
    if m[2]:
        return Fact(m[1], "instance", m[4], m[2] in ("n'est pas", "is not"))
    return Fact(m[1], "in", m[6], m[5] in ("n'est pas dans", "is not in"))


def parse_query(text):
    t = fold(text).replace("’", "'").strip().rstrip("?.").strip()
    # A question mark prevents declarative teaching even if no inversion is used.
    if t.startswith("est-ce que "):
        return parse_fact(t[11:])
    m = re.fullmatch(rf"({LABEL}) est-(?:il|elle) (.+)", t)
    if m:
        return parse_fact(f"{m[1]} est {m[2]}")
    m = re.fullmatch(rf"is ({LABEL}) (.+)", t)
    if m:
        return parse_fact(f"{m[1]} is {m[2]}")
    if text.rstrip().endswith("?"):
        return parse_fact(t)
    return None


class SemanticDialogue:
    def __init__(self, agent):
        self.memory = SemanticMemory(agent)
        self.ltm = agent.ltm
        self.require = lambda action, target=None, detail=None: None  # set by the chat engine's policy guard

    def context(self, cid):
        for m in reversed(self.ltm.messages(cid)):
            meta = m.get("meta") or {}
            if "semantic_subject" in meta:
                return meta["semantic_subject"]
        return None

    def previous(self, cid):
        return next((m.get("meta") or {} for m in reversed(self.ltm.messages(cid))
                     if m["role"] == "genesis"), {})

    def respond(self, cid, text, lang):
        tr = lambda fr, en: fr if lang == "fr" else en
        previous = self.previous(cid)
        normalized = fold(text).strip().rstrip("?.").strip()
        retract = re.fullmatch(r"(?:retire le fait|retract fact) #?(\d+)", normalized)
        if retract and not text.rstrip().endswith("?"):
            kid = int(retract[1])
            self.require("semantic.teach", f"knowledge:{kid}", {"op": "retract"})
            try:
                self.memory.retract(kid, cid, text)
            except ValueError as exc:
                return tr("Rétractation refusée : ", "Retraction refused: ") + str(exc), {
                    "intent": "refused", "reason": str(exc)}
            return tr("Enseignement rétracté ; historique et épisodes conservés.",
                      "Teaching retracted; history and episodes preserved."), {
                          "intent": "semantic-retract", "knowledge_ids": [kid]}
        if normalized in ("pourquoi", "why") and previous.get("intent", "").startswith("semantic-"):
            ids = previous.get("knowledge_ids", [])
            if not ids:
                return tr("Je n'ai pas de prémisse utilisable pour cette réponse.",
                          "I have no usable premise for that answer."), {"intent": "semantic-explain"}
            lines = []
            for kid in ids:
                ex = self.ltm.explain(kid)
                episodes = ex["evidence"]["supports"]["examples"]
                lines.append(f"- #{kid} `{ex['item']['statement']}` ({ex['item']['status']})")
                lines.extend(f"  {json.loads(ep['observation'])['text']}" for ep in episodes)
            return tr("Voici les enseignements utilisés, pas des observations vérifiées :\n",
                      "These are the teachings used, not verified observations:\n") + "\n".join(lines), {
                          "intent": "semantic-explain", "knowledge_ids": ids}
        recall = re.fullmatch(
            rf"(?:que sais-tu (?:de|sur)|what do you know about) ({LABEL})", normalized)
        if recall:
            subject = recall[1]
            if subject in PRONOUNS:
                subject = self.context(cid)
            if not subject:
                return tr("De qui parles-tu ? Utilise son nom.",
                          "Who do you mean? Please use the name."), {"intent": "semantic-clarify"}
            items = [k for k in self.memory.items() if k.expr["subject"] == subject]
            lines = [f"- #{k.id} `{k.statement}` ({k.status})" for k in items]
            return tr("Enseignements externes conservés (non vérifiés) :",
                      "Stored external teachings (unverified):") + "\n" + (
                          "\n".join(lines) if lines else tr("Aucun.", "None.")), {
                              "intent": "semantic-recall", "semantic_subject": subject,
                              "knowledge_ids": [k.id for k in items]}
        question = parse_query(text)
        correction = bool(re.match(r"^(?:correction|corrige)\s*:", text, re.I))
        body = re.sub(r"^(?:correction|corrige)\s*:\s*", "", text, flags=re.I)
        parts = re.split(r"\s*;\s*|(?<=[.!?])\s+", body.strip())
        facts = [parse_fact(part) for part in parts]
        if question:
            facts = [question]
        elif "?" in body or not all(facts):
            if correction or any(facts):
                return tr(
                    "Je n'ai rien enregistré : sépare les enseignements simples par des points ou "
                    "des points-virgules, et pose les questions dans un autre message.",
                    "I saved nothing: separate simple teachings with periods or semicolons, "
                    "and ask questions in another message."), {"intent": "semantic-clarify"}
            return None
        subject = self.context(cid)
        resolved = []
        for fact in facts:
            if fact.subject in PRONOUNS:
                if not subject:
                    return tr("De qui parles-tu ? Utilise son nom.",
                              "Who do you mean? Please use the name."), {"intent": "semantic-clarify"}
                fact = Fact(subject, fact.relation, fact.object, fact.negative)
            resolved.append(fact)
            if fact.relation != "subclass":
                subject = fact.subject
        subjects = {f.subject for f in resolved if f.relation != "subclass"}
        meta = {"semantic_subject": next(iter(subjects)) if len(subjects) == 1
                else subject if not subjects else None}
        if question:
            result = self.memory.query(resolved[0], cid, text)
            status = result["status"]
            wording = {
                "supported": tr("Oui, selon les enseignements en mémoire.",
                                "Yes, according to the teachings in memory."),
                "opposed": tr("Non, selon les enseignements en mémoire.",
                              "No, according to the teachings in memory."),
                "unknown": tr("Je ne sais pas : aucune preuve utilisable en mémoire.",
                              "I don't know: there is no usable evidence in memory."),
                "conflict": tr("Mes sources se contredisent : je ne peux pas trancher.",
                               "My sources conflict: I cannot decide."),
            }[status]
            if result["knowledge_ids"]:
                wording += "\n\n" + tr("Prémisses : ", "Premises: ") + "; ".join(
                    f"#{i} `{s}`" for i, s in zip(result["knowledge_ids"], result["premises"]))
                wording += "\n\n" + tr(
                    "Conclusion conditionnelle, pas une observation vérifiée.",
                    "Conditional conclusion, not a verified observation.")
            return wording, {**meta, **result, "intent": "semantic-query"}
        self.require("semantic.teach", None, {"facts": [f.statement for f in resolved], "correction": correction})
        try:
            ids = self.memory.teach(resolved, cid, text, correction)
        except ValueError as exc:
            return tr("Enseignement refusé : ", "Teaching refused: ") + str(exc), {
                "intent": "refused", "reason": str(exc)}
        results = [self.memory.answer(f) for f in resolved]
        conflict = any(r["status"] == "conflict" for r in results)
        reply = tr("Enseignement conservé (source : toi, non vérifiée).",
                   "Teaching saved (source: you, unverified).")
        reply += "\n" + "\n".join(f"- #{i} `{f.statement}`" for i, f in zip(ids, resolved))
        if conflict:
            reply += "\n\n" + tr(
                "Contradiction conservée. Pour corriger explicitement : `correction : <fait>`.",
                "Conflict preserved. To correct explicitly: `correction: <fact>`.")
        if correction:
            reply += "\n\n" + tr("L'affirmation opposée est réfutée, sans effacer son historique.",
                                 "The opposite assertion is refuted without erasing its history.")
        return reply, {**meta, "intent": "semantic-teach", "knowledge_ids": ids,
                       "conflict": conflict}
