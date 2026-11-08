"""Learn lexical predicates from paired objects, experiments and annotations.

The predicate grammar is supplied; no word has a supplied meaning. Object
identity is evidence/context, never a semantic feature. Labels are external
teaching, not observations of physical truth.
"""
from __future__ import annotations

import json
import re
from itertools import chain

from .language import fold

MAX_WORDS = 64
MAX_LABELS = 512
MAX_MEANINGS = 128
FORWARD_CHECKS = 3
WORD = r"[^\W\d_]{1,32}"


def language_manifest():
    from .constructions import manifest
    return {"version": 2, "meanings_at_birth": [],
            "mechanisms": ["sensory/effect predicate version space", "positive/negative grounded examples",
                           "novel-input forward checks", "three-valued conjunction and negation"],
            "limits": [MAX_WORDS, MAX_LABELS, MAX_MEANINGS, FORWARD_CHECKS],
            "construction_learning": manifest()}


class GroundedLexicon:
    def __init__(self, explorer):
        self.g = explorer
        self.ltm = explorer.ltm
        self.key = f"grounded-language/{explorer.env_id}"
        self.state = self.ltm.get_meta(self.key) or {"examples": [], "validation": {}}

    def revision(self):
        return dict(self.g.state["revisions"])

    def examples(self, word=None):
        return [e for e in self.state["examples"] if e["active"] and e["revision"] == self.revision()
                and (word is None or e["word"] == word)]

    def snapshot(self, obj):
        if set(obj["features"]) != set(self.g.features):
            raise ValueError("object has unknown sensors")
        effects, episodes = {}, []
        for action in self.g.actions:
            rows = [r for r in self.g.rows(action) if r["object_id"] == obj["id"]
                    and r["features"] == obj["features"]]
            if rows:
                effects[action] = rows[-1]["effect"]
                episodes.append(rows[-1]["episode"])
        return {"features": dict(obj["features"]), "effects": effects, "experiment_episodes": episodes}

    def inventory(self, extra=None):
        sensors, effects = set(), set()
        for row in chain.from_iterable(self.g.rows(a) for a in self.g.actions):
            sensors.update(row["features"].items())
            effects.add((row["action"], row["effect"]))
        for example in self.examples() + ([extra] if extra else []):
            sensors.update(example["features"].items())
            effects.update(example["effects"].items())
        meanings = ([("sensor", k, v) for k, v in sorted(sensors)]
                    + [("effect", k, v) for k, v in sorted(effects)])
        if len(meanings) > MAX_MEANINGS:
            raise ValueError("grounded lexical hypothesis budget exceeded")
        return meanings

    @staticmethod
    def observed(meaning, example):
        kind, name, value = meaning
        source = example["features"] if kind == "sensor" else example["effects"]
        return source[name] == value if name in source else None

    def candidates(self, word):
        examples = self.examples(word)
        if not examples:
            return []
        return [m for m in self.inventory()
                if all(self.observed(m, e) in (None, e["label"]) for e in examples)]

    def predict(self, word, obj):
        word = fold(word)
        meanings = self.candidates(word)
        votes = []
        for kind, name, value in meanings:
            observed = (obj["features"][name] if kind == "sensor" else
                        self.effect_value(obj, name))
            votes.append(None if observed is None else observed == value)
        prediction = votes[0] if votes and None not in votes and len(set(votes)) == 1 else None
        validation = self.state["validation"].get(word, {})
        signature = json.dumps([self.revision(), meanings], sort_keys=True)
        validated = (validation.get("signature") == signature and validation.get("ok", 0) >= FORWARD_CHECKS
                     and validation.get("bad", 0) == 0
                     and {e["label"] for e in self.examples(word)} == {True, False})
        return {"value": prediction, "validated": validated, "candidates": meanings,
                "status": "validated" if validated else "candidate" if meanings else
                "inconsistent" if self.examples(word) else "unknown"}

    def effect_value(self, obj, action):
        observed = [r for r in self.g.rows(action) if r["object_id"] == obj["id"]
                    and r["features"] == obj["features"]]
        if observed:
            return observed[-1]["effect"]
        if not self.ltm.knowledge(key=f"{self.g.key}/{action}", status="validated"):
            return None
        return self.g.predict(obj["features"], action)["value"]

    def teach(self, word, obj, label, correction=False, source="user-grounded-teaching"):
        word = fold(word.strip())
        if not re.fullmatch(WORD, word) or type(label) is not bool:
            raise ValueError("one alphabetic word and a boolean label are required")
        if source not in ("user-grounded-teaching", "simulated-grounded-teaching"):
            raise ValueError("unknown label provenance")
        words = {e["word"] for e in self.state["examples"]}
        if len(self.state["examples"]) >= MAX_LABELS or word not in words and len(words) >= MAX_WORDS:
            raise ValueError("grounded vocabulary memory budget exceeded")
        snapshot = self.snapshot(obj)
        self.inventory(snapshot)  # Check the resource bound before any change.
        prior = self.predict(word, obj)
        previous = self.examples(word)
        novel = not any(e["features"] == obj["features"] for e in previous)
        ep = self.ltm.record_episode(
            self.g.agent.session, self.g.env_id, self.key, source,
            action={"word": word, "correction": correction},
            observation={"object_id": obj["id"], **snapshot},
            outcome={"label": label},
            predictions={word: {"predicted": prior["value"], "actual": label,
                                "confidence": None, "novel_input": novel}})
        if correction:
            for e in previous:
                if e["features"] == obj["features"]:
                    e["active"] = False
        self.state["examples"].append(
            {**snapshot, "word": word, "object_id": obj["id"], "label": label, "episode": ep,
             "revision": self.revision(), "active": True})
        meanings = self.candidates(word)
        signature = json.dumps([self.revision(), meanings], sort_keys=True)
        validation = self.state["validation"].get(word, {})
        if validation.get("signature") != signature or prior["candidates"] != meanings:
            validation = {"signature": signature, "ok": 0, "bad": 0, "episodes": []}
        elif novel and prior["value"] is not None:
            validation["ok" if prior["value"] == label else "bad"] += 1
            validation["episodes"].append(ep)
        self.state["validation"][word] = validation
        result = self.predict(word, obj)
        key = f"{self.key}/{word}"
        statement = json.dumps({"word": word, "revision": self.revision(), "meanings": meanings}, sort_keys=True)
        for item in self.ltm.knowledge(key=key):
            if item.statement != statement and item.status not in ("refuted", "dormant"):
                self.ltm.set_status(item.id, "refuted" if correction or not meanings else "dormant",
                                    "grounded word interpretation revised")
                if correction or not meanings:
                    self.ltm.add_evidence(item.id, [ep], "refuted_by")
        kid = self.ltm.upsert_knowledge(
            key, "grounded-word", statement, {"word": word, "meanings": meanings},
            validation["ok"] / max(validation["ok"] + validation["bad"], 1), result["status"],
            "external labels linked to sensory/action evidence", self.g.env_id,
            n_support=len(self.examples(word)), forward_ok=validation["ok"], forward_bad=validation["bad"],
            reason="grounded predicate learning; agreement with teacher, not verified word meaning")
        self.ltm.add_evidence(kid, [e["episode"] for e in self.examples(word)], "induced_from")
        self.ltm.add_evidence(kid, list(chain.from_iterable(
            e["experiment_episodes"] for e in self.examples(word))), "induced_from")
        self.ltm.add_evidence(kid, validation["episodes"], "supports")
        self.ltm.set_meta(self.key, self.state)
        self.ltm.commit()
        return {**result, "knowledge_id": kid, "episode_id": ep,
                "forward_ok": validation["ok"], "forward_bad": validation["bad"]}

    def match(self, obj, terms, require_validated=True):
        values = []
        for word, negative in terms:
            result = self.predict(word, obj)
            value = result["value"] if result["validated"] or not require_validated else None
            values.append(None if value is None else (not value if negative else value))
        return False if False in values else True if values and all(v is True for v in values) else None

    def select(self, objects, terms):
        return [obj for obj in objects if self.match(obj, terms) is True]

    def suggest_example(self, word):
        seen = [e["features"] for e in self.examples(word)]
        unseen = [obj for obj in self.g.objects if obj["features"] not in seen]
        # A disagreement is especially useful; otherwise a new configuration
        # gives an honest forward check rather than repeated training evidence.
        return max(unseen, key=lambda o: (self.predict(word, o)["value"] is None, o["id"])) if unseen else None


def parse_terms(text):
    terms = []
    for part in re.split(r"\s+(?:et|and)\s+", fold(text).strip()):
        m = re.fullmatch(rf"(?:(pas|not)\s+)?({WORD})", part)
        if not m:
            return None
        terms.append((m[2], bool(m[1])))
    return terms if 0 < len(terms) <= 4 else None


class ObjectDialogue:
    def __init__(self, agent, factory):
        self.agent, self.ltm, self.factory = agent, agent.ltm, factory
        self.worlds = {}
        self.require = lambda action, target=None, detail=None: None  # set by the chat engine's policy guard

    def workshop(self, seed):
        from .grounding import GroundedExplorer
        if seed not in self.worlds:
            self.worlds[seed] = GroundedExplorer(self.agent, self.factory(seed))
        g = self.worlds[seed]
        return g, GroundedLexicon(g)

    def respond(self, cid, text, lang):
        from .constructions import ConstructionLearner
        from .environments.base import BudgetExceeded
        tr = lambda fr, en: fr if lang == "fr" else en
        t = fold(text).replace("’", "'").strip().rstrip(".!")
        start = re.fullmatch(r"(?:atelier objets|object workshop)(?: (\d{1,6}))?", t)
        explore = re.fullmatch(r"(?:explore objets|explore objects) (\d{1,4})", t)
        observe = re.fullmatch(r"(?:observe|regarde) ([a-z0-9-]+)", t)
        test = re.fullmatch(r"(?:teste|test) ([a-z0-9-]+)", t)
        label = re.fullmatch(
            rf"(?:(correction)\s*:\s*)?(?:cet objet (est|n'est pas)|this object (is|is not)) ({WORD})", t)
        query = re.fullmatch(r"(?:cet objet est-il|is this object) (.+)\s*\?", t)
        select = re.fullmatch(r"(?:montre un objet|show an? object) (.+?)[?]?", t)
        explain = re.fullmatch(rf"(?:pourquoi le mot|explain word) ({WORD})\s*[?]?", t)
        teaching = re.fullmatch(
            r"(?:(correction)\s*:\s*)?(?:apprends phrase|learn phrase)\s*:\s*(.+?)\s*=>\s*(.+)", t)
        context_key = f"object-dialogue/{cid}"
        context = self.ltm.get_meta(context_key)
        learned = None
        if not any((start, explore, observe, test, label, query, select, explain, teaching)):
            if self.factory is None or context is None:
                return None
            try:
                _, lexicon = self.workshop(context["seed"])
                learned = ConstructionLearner(lexicon).parse(t)
            except (ValueError, BudgetExceeded) as exc:
                return tr("Demande refusée : ", "Request refused: ") + str(exc), {"intent": "refused"}
            if learned is None:
                return None
        if self.factory is None:
            return tr("L'atelier objets n'est pas configuré dans cette interface.",
                      "The object workshop is not configured in this interface."), {"intent": "refused"}
        if start:
            seed = int(start[1] or 0)
            self.require("objects.explore", f"workshop:{seed}")
            try:
                g, _ = self.workshop(seed)
            except (ValueError, BudgetExceeded) as exc:
                return tr("Atelier refusé : ", "Workshop refused: ") + str(exc), {
                    "intent": "refused", "reason": str(exc)}
            self.ltm.set_meta(context_key, {"seed": seed, "selected": None})
            return tr(
                "Atelier prêt. Aucun sens de mot n'est fourni. Explore les effets, observe un objet, "
                "puis enseigne « cet objet est dax » ou « cet objet n'est pas dax ». "
                "Les sens restent des hypothèses jusqu'aux vérifications sur d'autres objets.",
                "Workshop ready. No word meanings are supplied. Explore effects, observe an object, "
                "then teach “this object is dax” or “this object is not dax”. "
                "Meanings remain hypotheses until checked on other objects."), {
                    "intent": "object-workshop", "env_id": g.env_id,
                    "choices": [tr("explore objets 80", "explore objects 80"), "observe specimen-0"]}
        if context is None:
            return tr("Ouvre d'abord `atelier objets`.", "Open `object workshop` first."), {
                "intent": "object-workshop", "choices": [tr("atelier objets", "object workshop")]}
        try:
            g, lexicon = self.workshop(context["seed"])
            obj = next((o for o in g.objects if o["id"] == context["selected"]), None)
            if teaching or label:
                self.require("objects.teach", context["seed"], {"text": t[:120]})
            elif explore or observe or test:
                self.require("objects.explore", context["seed"], {"text": t[:120]})
            if teaching:
                terms = parse_terms(teaching[3])
                if terms is None:
                    raise ValueError("use learned words with et/and and pas/not after =>")
                result = ConstructionLearner(lexicon).teach(teaching[2], terms, correction=bool(teaching[1]))
                return tr("Construction enregistrée : ", "Construction recorded: ") + (
                    f"#{result['knowledge_id']} — {result['status']}; {result['forward_ok']} "
                ) + tr("prédictions correctes sur de nouvelles combinaisons.",
                       "correct predictions on new bindings."), {"intent": "construction-teach", **result}
            if explore:
                steps = int(explore[1])
                if not 1 <= steps <= 128:
                    raise ValueError("exploration requires 1-128 steps")
                report = g.run(steps)
                return tr(
                    f"{report['steps']} expériences ; arrêt : {report['stopped']}. Choisis un objet à observer.",
                    f"{report['steps']} experiments; stop: {report['stopped']}. Choose an object to observe."), {
                        "intent": "object-explore", **report,
                        "choices": [f"observe {o['id']}" for o in g.objects[:6]]}
            if observe:
                obj = next((o for o in g.objects if o["id"] == observe[1]), None)
                if obj is None:
                    raise ValueError("unknown specimen ID")
                self.ltm.set_meta(context_key, {**context, "selected": obj["id"]})
                snapshot = lexicon.snapshot(obj)
                return f"**{obj['id']}**\n\n" + tr("Capteurs : ", "Sensors: ") + (
                    f"`{json.dumps(obj['features'], sort_keys=True)}`\n\n"
                ) + tr("Effets réellement observés : ", "Actually observed effects: ") + (
                    f"`{json.dumps(snapshot['effects'], sort_keys=True)}`"
                ), {"intent": "object-observe", "object": obj, **snapshot,
                    "choices": [tr(f"teste {a}", f"test {a}") for a in g.actions]}
            if select or learned:
                if learned and not learned["validated"]:
                    return tr(
                        "Cette construction apprise est ambiguë ou non validée. "
                        "Enseigne d'autres exemples avec `apprends phrase : phrase => sens`.",
                        "This learned construction is ambiguous or unvalidated. "
                        "Teach more examples with `learn phrase: sentence => meaning`."
                    ), {"intent": "construction-uncertain", **learned}
                terms = learned["terms"] if learned else parse_terms(select[1])
                if terms is None:
                    raise ValueError("use up to four words joined with 'et/and', optionally 'pas/not'")
                objects = lexicon.select(g.objects, terms)
                if objects:
                    self.ltm.set_meta(context_key, {**context, "selected": objects[0]["id"]})
                prefix = tr("Construction apprise. ", "Learned construction. ") if learned else ""
                return prefix + ((tr("Objets correspondant aux sens appris : ", "Objects matching learned meanings: ")
                        + ", ".join(o["id"] for o in objects)) if objects else tr(
                            "Aucun objet sélectionné : pas de correspondance confirmée, ou sens encore inconnu.",
                            "No object selected: no confirmed match, or a meaning is still unknown.")), {
                                "intent": "object-select", "objects": [o["id"] for o in objects], "terms": terms,
                                "construction": learned}
            if explain:
                meanings = lexicon.candidates(explain[1])
                items = self.ltm.knowledge(key=f"{lexicon.key}/{explain[1]}")
                refs = ", ".join(f"#{k.id}" for k in items)
                return tr("Sens compatibles avec les exemples (pas une définition innée) : ",
                          "Meanings compatible with the examples (not an innate definition): ") + (
                              f"`{json.dumps(meanings)}`\n\n{refs}"
                          ), {"intent": "object-word-explain", "candidates": meanings,
                              "knowledge_ids": [k.id for k in items]}
            if obj is None:
                raise ValueError("observe a specimen first")
            if test:
                result = g.experience(obj, test[1])
                return tr("Prédiction avant action : ", "Prediction before action: ") + (
                    f"`{result['prediction']['value']}`; "
                ) + tr("effet observé : ", "observed effect: ") + f"`{result['effect']}`", {
                    "intent": "object-test", **result}
            if label:
                result = lexicon.teach(label[4], obj, (label[2] or label[3]) in ("est", "is"),
                                       correction=bool(label[1]))
                suggestion = lexicon.suggest_example(label[4])
                return tr("Exemple enregistré, avec ses observations. ", "Example saved with its observations. ") + (
                    f"#{result['knowledge_id']} — {result['status']}; "
                    f"{len(result['candidates'])} "
                ) + tr("sens compatibles. ", "compatible meanings. ") + (
                    tr("Exemples contradictoires ou sens hors de mon espace d'hypothèses.",
                       "Conflicting examples or a meaning outside my hypothesis space.")
                    if result["status"] == "inconsistent" else ""
                ), {"intent": "object-word-teach", **result,
                    "choices": [f"observe {suggestion['id']}"] if suggestion else []}
            terms = parse_terms(query[1].strip())
            if terms is None:
                raise ValueError("unsupported object description")
            value = lexicon.match(obj, terms, require_validated=False)
            meanings = [lexicon.predict(w, obj) for w, _ in terms]
            verified = all(m["validated"] for m in meanings)
            response = tr("Je ne sais pas.", "I don't know.") if value is None else (
                tr("Oui", "Yes") if value else tr("Non", "No"))
            return response + tr(
                " — interprétation apprise, pas une définition universelle." if verified else " — hypothèse non validée.",
                " — learned interpretation, not a universal definition." if verified else " — unvalidated hypothesis."
            ), {"intent": "object-word-query", "value": value, "validated": verified, "meanings": meanings}
        except (ValueError, BudgetExceeded) as exc:
            return tr("Demande refusée : ", "Request refused: ") + str(exc), {
                "intent": "refused", "reason": str(exc)}
