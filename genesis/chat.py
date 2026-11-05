"""Conversational entry point to GENESIS.

This is **not** a language model and does not pretend to be one. The agent has never learned words, so
the interface understands only a small command grammar (French or English keywords plus integers). That
grammar is part of the *interface*, fixed by the designers and listed in the innate manifest's spirit:
it is not knowledge the agent acquired, and it is documented as such to the user.

What the agent says about the world comes only from its own memory:

* A sequence of integers typed by the user is fed term by term through the normal cognitive cycle
  (``Genesis.live``): the agent predicts each term *before* reading it, learns from the error, and
  stores episodes with provenance ``user-provided`` (external information, kept distinct from its own
  observations and experiments).
* Questions about knowledge, stage, limitations or curiosity are answered from long-term memory.
* ``explore`` lets it run curiosity-driven experiments in a fresh sandboxed world supplied by the harness.

Every answer carries ``meta`` (intent, keys, knowledge ids) so it can be traced back to memory rows.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Callable, Optional

from .adapters import SEQ_VARS, Outcome, Query, Situation
from .curiosity import Curiosity
from .metacognition import Metacognition
from .perception import Perception

MAX_TEXT = 2000
MAX_TERMS = 200
MAX_ABS = 10 ** 9
MAX_EXPLORE = 300
DEFAULT_EXPLORE = 100

INT_RE = re.compile(r"(?<![\w.\-])-?\d+(?![\w.])")

FR = {"bonjour", "salut", "coucou", "bonsoir", "que", "quoi", "sais", "tu", "explique", "pourquoi", "aide",
      "stade", "etat", "limites", "suite", "merci", "je", "les", "des", "est", "connais", "connaissances",
      "predis", "prochain", "prochains", "suivant", "suivants", "curiosite", "objectifs", "comment", "qui", "es",
      "ton", "ta", "tes", "une", "et", "le", "la", "du", "ca", "va", "explorer", "appris", "ignores", "peux",
      "faire", "quelles", "quels", "quel", "quelle", "voici", "trouve", "regle", "encore", "croyances"}
EN = {"hello", "hi", "hey", "what", "know", "explain", "why", "help", "stage", "status", "limitations",
      "continue", "more", "thanks", "thank", "you", "the", "is", "are", "predict", "next", "who", "how", "do",
      "your", "knowledge", "learned", "can", "find", "rule", "here", "beliefs", "objectives"}

EXPLAIN = {"explain", "explique", "expliquer", "pourquoi", "why", "evidence", "preuve", "preuves", "justify",
           "justifie"}
CONTINUE = {"continue", "suite", "encore", "more", "then", "puis"}
PREDICT = {"predict", "predis", "prediction", "predictions", "next", "suivant", "suivants", "prochain",
           "prochains", "apres", "after"}
EXPLORE = {"explore", "explorer", "exploration", "experimente", "experiment", "experiments", "experiences"}
KNOW = {"know", "sais", "connais", "knowledge", "connaissances", "learned", "appris", "beliefs", "croyances",
        "savoir", "learnt"}
STAGE = {"stage", "stade", "status", "etat", "progress", "progres", "developpement", "development", "niveau",
         "level", "memoire", "memory"}
LIMITS = {"limitations", "limites", "limits", "faiblesses", "weaknesses", "echecs", "failures", "limitation",
          "limite"}
QUESTIONS = {"questions", "curiosity", "curiosite", "objectifs", "objectives", "gaps", "lacunes", "ignores",
             "uncertain", "incertain", "doutes", "doubts"}
HELP = {"help", "aide", "commands", "commandes", "capacites", "capabilities", "faire", "usage", "mode", "emploi"}
GREET = {"bonjour", "salut", "hello", "hi", "hey", "coucou", "bonsoir", "yo"}
WHO = {"qui", "who"}
THANKS = {"merci", "thanks", "thank", "thx"}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _words(text: str) -> list:
    return re.findall(r"[a-z]+", _norm(text))


def detect_lang(text: str, fallback: Optional[str] = None) -> str:
    ws = set(_words(text))
    fr, en = len(ws & FR), len(ws & EN)
    if any(c in text for c in "éèêàçùôîâ"):
        fr += 2
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return fallback if fallback in ("fr", "en") else "en"


def _t(lang: str, fr: str, en: str) -> str:
    return fr if lang == "fr" else en


STATUS_FR = {"validated": "validée", "candidate": "candidate", "unpredictable": "imprévisible", "refuted": "réfutée",
             "hypothesis": "hypothèse", "dormant": "dormante", "unknown": "inconnue"}
REASONS_FR = {"step limit": "budget atteint", "sandbox budget exhausted": "budget du bac à sable épuisé",
              "nothing left to learn within current hypothesis space":
                  "plus rien à apprendre dans mon espace d'hypothèses actuel",
              "safety interaction limit": "limite de sécurité atteinte"}
NOTES_FR = {
    "hypothesis space is bounded by max_size; longer programs are unreachable":
        "mon espace d'hypothèses est borné (max_size) : les programmes plus longs me sont inaccessibles",
    "constants are proposed from observed data only; unseen moduli cannot be discovered":
        "je ne propose des constantes qu'à partir des données : un modulo jamais observé m'échappe",
    "search budgets are safety-locked and cannot be raised by self-improvement":
        "mes budgets de recherche sont verrouillés par la sécurité ; je ne peux pas les augmenter moi-même"}


def _st(lang: str, status: str) -> str:
    return STATUS_FR.get(status, status) if lang == "fr" else status


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x is not None else "?"


class UserSequenceAdapter:
    """Presents terms typed by the user to the cognitive cycle, one at a time, as external information."""

    kind = "sequence"
    focusable = False

    def __init__(self, agent, env_id: str, key: str, terms: list):
        self.agent = agent
        self.env_id = env_id
        self.key_ = key
        self.terms = list(terms)
        self.pos = 0

    def var_names(self, key: str) -> list:
        return SEQ_VARS

    def candidates(self) -> list:
        window, t = self.agent.recall_sequence(self.key_)
        if len(window) >= 3:
            feats = Perception.sequence_features(window, t)
            percept = self.agent.perception.perceive("sequence", feats, self.key_)
            return [Situation({"read": "user term"}, [Query(self.key_, feats)], "user-provided", self.key_, percept)]
        return [Situation({"read": "user term"}, [], "user-provided", self.key_, None, warmup=1.0)]

    def execute(self, sit: Situation) -> Outcome:
        y = self.terms[self.pos]
        self.pos += 1
        return Outcome({self.key_: y}, {"term": y})

    def after(self, sit: Situation, outcome: Outcome) -> None:
        self.agent.remember_term(self.key_, outcome.targets[self.key_])


class ChatEngine:
    """Turns a message into an action of the agent and an answer grounded in its memory."""

    def __init__(self, agent, world_factory: Optional[Callable] = None):
        self.agent = agent
        self.ltm = agent.ltm
        self.world_factory = world_factory

    # ------------------------------------------------------------ transcript
    def new_conversation(self, lang: Optional[str] = None) -> dict:
        cid = self.ltm.add_conversation(None, lang if lang in ("fr", "en") else None)
        self.ltm.commit()
        return self.ltm.conversation(cid)

    def conversations(self) -> list:
        return self.ltm.conversations()

    def messages(self, cid: int) -> list:
        return self.ltm.messages(cid)

    def send(self, cid: int, text: str, lang_hint: Optional[str] = None) -> dict:
        conv = self.ltm.conversation(cid)
        if conv is None:
            raise KeyError(cid)
        text = (text or "").strip()[:MAX_TEXT]
        if not text:
            raise ValueError("empty message")
        lang = detect_lang(text, conv.get("lang") or lang_hint)
        user = self.ltm.add_message(cid, "user", text)
        reply, meta = self.respond(cid, text, lang)
        meta = {"lang": lang, **meta}
        bot = self.ltm.add_message(cid, "genesis", reply, meta)
        self.ltm.update_conversation(cid, lang=lang, title=conv["title"] or text[:60])
        self.ltm.commit()
        return {"user": user, "reply": bot, "conversation": self.ltm.conversation(cid)}

    # ---------------------------------------------------------------- intents
    def respond(self, cid: int, text: str, lang: str) -> tuple:
        ws = set(_words(text))
        nums = [int(m) for m in INT_RE.findall(text)]
        last_seq = self._last(cid, "key")
        if ws & EXPLAIN:
            kid = nums[0] if nums else self._last(cid, "knowledge_id")
            if kid is not None:
                return self.explain(kid, lang)
        if ws & CONTINUE and nums and last_seq:
            return self.learn_sequence(cid, nums, lang, key=last_seq)
        if len(nums) >= 4:
            return self.learn_sequence(cid, nums, lang)
        if 1 <= len(nums) <= 3 and not (ws - {"et", "and"}) and last_seq:
            return self.learn_sequence(cid, nums, lang, key=last_seq)
        if ws & PREDICT and last_seq:
            return self.predict(last_seq, min(max(nums[0], 1), 20) if nums else 5, lang)
        if ws & EXPLORE:
            return self.explore(min(max(nums[0], 10), MAX_EXPLORE) if nums else DEFAULT_EXPLORE, lang)
        if ws & LIMITS:
            return self.limitations(lang)
        if ws & QUESTIONS:
            return self.questions(lang)
        if ws & KNOW:
            return self.knowledge(lang)
        if ws & STAGE:
            return self.stage(lang)
        if ws & HELP or text.strip() == "?":
            return self.help(lang), {"intent": "help"}
        if ws & GREET or (ws & WHO and ws & {"es", "are", "you", "tu"}):
            return self.intro(lang), {"intent": "intro"}
        if ws & THANKS:
            return _t(lang, "Avec plaisir. Donne-moi d'autres suites à analyser quand tu veux.",
                      "You're welcome. Send me more sequences whenever you like."), {"intent": "thanks"}
        if 0 < len(nums) < 4:
            return _t(lang, "Il me faut au moins 4 nombres entiers pour chercher une règle : je prédis chaque terme "
                            "à partir des 3 précédents.",
                      "I need at least 4 integers to look for a rule: I predict each term from the previous 3."), \
                {"intent": "too-short"}
        return self.not_understood(lang), {"intent": "not-understood"}

    def _last(self, cid: int, field: str):
        for m in reversed(self.ltm.messages(cid)):
            if m["role"] == "genesis" and m["meta"] and m["meta"].get(field) is not None:
                return m["meta"][field]
        return None

    # ----------------------------------------------------------- sequences
    def learn_sequence(self, cid: int, terms: list, lang: str, key: Optional[str] = None) -> tuple:
        if len(terms) > MAX_TERMS or any(abs(v) > MAX_ABS for v in terms):
            return _t(lang, f"Je refuse cette entrée : au plus {MAX_TERMS} termes, chacun entre -10⁹ et 10⁹.",
                      f"Input refused: at most {MAX_TERMS} terms, each between -10⁹ and 10⁹."), \
                {"intent": "refused"}
        agent = self.agent
        env_id = f"conversation-{cid}"
        appended = key is not None
        if key is None:
            n = sum(1 for k in agent.learners if k.startswith(env_id + "/"))
            key = f"{env_id}/seq{n + 1}"
        first_ep = self.ltm.db.execute("SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()[0]
        adapter = UserSequenceAdapter(agent, env_id, key, terms)
        agent.live(adapter, len(terms), "curiosity", stop_when_idle=False)
        trace = []
        for r in self.ltm.db.execute("SELECT predictions, outcome FROM episodes WHERE context=? AND id>? ORDER BY id",
                                     (key, first_ep)):
            pred = (json.loads(r["predictions"]) if r["predictions"] else {}).get(key)
            y = json.loads(r["outcome"])["term"]
            trace.append({"term": y, "predicted": pred["predicted"] if pred else None})
        lr = agent.learners.get(key)
        status = lr.status() if lr else "unknown"
        top, mass, p_null = lr.best() if lr else (None, 0.0, 1.0)
        rule = str(top.expr) if top else None
        kid = None
        if rule:
            kid = next((k.id for k in self.ltm.knowledge(key=key) if k.statement == rule), None)
        if status == "unpredictable":
            kid = next((k.id for k in self.ltm.knowledge(key=key) if k.status == "unpredictable"), None)
        future = agent.rollout_sequence(key, 5) if status in ("validated", "candidate") else []
        predicted = [t for t in trace if t["predicted"] is not None]
        n_ok = sum(t["predicted"] == t["term"] for t in predicted)
        n_total = len(lr.data) if lr else 0
        meta = {"intent": "sequence", "key": key, "status": status, "rule": rule, "confidence": round(mass, 4),
                "p_null": round(p_null, 4), "knowledge_id": kid, "predicted_before_seeing": len(predicted),
                "correct_before_seeing": n_ok, "next_terms": future, "appended": appended}
        lines = []
        if appended:
            lines.append(_t(lang, f"J'ai lu {len(terms)} terme(s) de plus de la même suite.",
                            f"I read {len(terms)} more term(s) of the same sequence."))
        if predicted:
            marks = " ".join(f"{t['term']}{'·' if t['predicted'] is None else '✓' if t['predicted'] == t['term'] else '✗'}"
                             for t in trace[-12:])
            lines.append(_t(lang,
                            f"Avant de lire chaque terme, j'ai essayé de le prédire : **{n_ok}/{len(predicted)}** "
                            f"correct(s).\n`{marks}`",
                            f"Before reading each term I tried to predict it: **{n_ok}/{len(predicted)}** correct.\n"
                            f"`{marks}`"))
        ref = f" (#{kid})" if kid else ""
        fwd = (top.fwd_ok, top.fwd_ok + top.fwd_bad) if top else (0, 0)
        if status == "validated":
            lines.append(_t(lang,
                            f"**Règle validée{ref} :** `{rule}` — confiance {_pct(mass)}, confirmée par {fwd[0]} "
                            f"prédiction(s) faite(s) avant de voir le résultat.",
                            f"**Validated rule{ref}:** `{rule}` — confidence {_pct(mass)}, confirmed by {fwd[0]} "
                            f"prediction(s) made before seeing the outcome."))
        elif status == "candidate":
            need = max(lr.cfg.validation_k - fwd[0], 1)
            lines.append(_t(lang,
                            f"**Hypothèse candidate{ref} :** `{rule}` — confiance {_pct(mass)}. Elle n'est **pas "
                            f"encore validée** : il me faut encore environ {need} prédiction(s) correcte(s). "
                            f"Envoie la suite avec `suite 1 2 3…`.",
                            f"**Candidate hypothesis{ref}:** `{rule}` — confidence {_pct(mass)}. It is **not "
                            f"validated yet**: I need about {need} more correct prediction(s). Send more terms "
                            f"with `continue 1 2 3…`."))
        elif status == "unpredictable":
            lines.append(_t(lang,
                            f"**Aucune règle trouvée{ref}.** Parmi les compositions de mes primitives, aucune "
                            f"n'explique ces {n_total} observations mieux que le hasard (p(aléatoire) = "
                            f"{_pct(p_null)}). Je la juge imprévisible plutôt que d'inventer une règle.",
                            f"**No rule found{ref}.** No composition of my primitives explains these {n_total} "
                            f"observations better than chance (p(random) = {_pct(p_null)}). I judge it "
                            f"unpredictable rather than invent a rule."))
        else:
            alt = f" `{rule}` ({_pct(mass)})" if rule else ""
            lines.append(_t(lang,
                            f"**Pas encore de conclusion.** Avec {n_total} observation(s), plusieurs hypothèses "
                            f"restent possibles{', la meilleure :' + alt if alt else ''}. Donne-moi plus de "
                            f"termes (`suite …`).",
                            f"**No conclusion yet.** With {n_total} observation(s), several hypotheses remain"
                            f"{', best so far:' + alt if alt else ''}. Give me more terms (`continue …`)."))
        if future and None not in future:
            lines.append(_t(lang, f"Termes suivants prédits : **{', '.join(map(str, future))}**",
                            f"Predicted next terms: **{', '.join(map(str, future))}**"))
        if rule and status in ("validated", "candidate"):
            lines.append(_t(lang, "_Notation : `x[t-1]` = terme précédent, `x[t-2]` = celui d'avant, `t` = position "
                                  "(le premier terme est à t = 0)._",
                            "_Notation: `x[t-1]` = previous term, `x[t-2]` = the one before, `t` = position "
                            "(first term is t = 0)._"))
        lines.append(_t(lang, "_Source : données fournies par toi (provenance « user-provided »), distinctes de mes "
                              "propres observations._",
                        "_Source: data you provided (provenance “user-provided”), kept separate from my own "
                        "observations._"))
        return "\n\n".join(lines), meta

    def predict(self, key: str, k: int, lang: str) -> tuple:
        lr = self.agent.learners.get(key)
        status = lr.status() if lr else "unknown"
        if status not in ("validated", "candidate"):
            return _t(lang, "Je n'ai pas de règle assez sûre pour cette suite ; je préfère ne pas deviner.",
                      "I have no sufficiently reliable rule for this sequence; I prefer not to guess."), \
                {"intent": "predict", "key": key, "status": status}
        future = self.agent.rollout_sequence(key, k)
        top, mass, _ = lr.best()
        return _t(lang, f"Avec `{top.expr}` ({_st(lang, status)}, confiance {_pct(mass)}), les {k} termes suivants seraient : "
                        f"**{', '.join(map(str, future))}**",
                  f"Using `{top.expr}` ({status}, confidence {_pct(mass)}), the next {k} terms would be: "
                  f"**{', '.join(map(str, future))}**"), \
            {"intent": "predict", "key": key, "status": status, "next_terms": future}

    # -------------------------------------------------------- self-knowledge
    def explain(self, kid: int, lang: str) -> tuple:
        ex = self.ltm.explain(kid)
        if not ex:
            return _t(lang, f"Je n'ai aucune connaissance #{kid} en mémoire.",
                      f"There is no knowledge item #{kid} in my memory."), {"intent": "explain", "found": False}
        it = ex["item"]
        lines = [_t(lang, f"**#{kid} — `{it['statement']}`** ({_st(lang, it['status'])}, confiance {_pct(it['confidence'])})",
                    f"**#{kid} — `{it['statement']}`** ({it['status']}, confidence {_pct(it['confidence'])})"),
                 _t(lang, f"Contexte : `{it['key']}` · {it['provenance']}",
                    f"Context: `{it['key']}` · {it['provenance']}")]
        if it.get("forward_ok") is not None:
            lines.append(_t(lang, f"Prédictions faites avant de voir le résultat : {it['forward_ok']} justes, "
                                  f"{it['forward_bad']} fausses.",
                            f"Predictions made before seeing the outcome: {it['forward_ok']} right, "
                            f"{it['forward_bad']} wrong."))
        names = {"induced_from": _t(lang, "induite à partir de", "induced from"),
                 "supports": _t(lang, "confirmée par", "supported by"),
                 "contradicts": _t(lang, "contredite par", "contradicted by"),
                 "refuted_by": _t(lang, "réfutée par", "refuted by")}
        ev = []
        for rel, d in ex["evidence"].items():
            if d["total"]:
                ids = ", ".join(str(e['id']) for e in d["examples"][:5])
                more = "…" if d["total"] > len(d["examples"][:5]) else ""
                ev.append(f"- {names[rel]} **{d['total']}** " + _t(lang, "épisode(s)", "episode(s)")
                          + f" (" + _t(lang, "épisodes", "episodes") + f" {ids}{more})")
        if ev:
            lines.append(_t(lang, "Preuves :", "Evidence:") + "\n" + "\n".join(ev))
        if ex["history"]:
            hist = [f"- {_st(lang, h['old_status']) if h['old_status'] else '∅'} → {_st(lang, h['new_status'])}"
                    f"{' : ' + h['reason'] if h['reason'] else ''}" for h in ex["history"][-5:]]
            lines.append(_t(lang, "Historique :", "History:") + "\n" + "\n".join(hist))
        return "\n\n".join(lines), {"intent": "explain", "knowledge_id": kid, "found": True}

    def knowledge(self, lang: str) -> tuple:
        items = self.ltm.knowledge()
        by = {}
        for k in items:
            by[k.status] = by.get(k.status, 0) + 1
        validated = [k for k in items if k.status == "validated"]
        if not items:
            return _t(lang, "Je ne sais encore rien : ma mémoire est vide. Tout ce que je saurai viendra de "
                            "l'expérience. Donne-moi une suite de nombres ou demande-moi d'`explorer`.",
                      "I don't know anything yet: my memory is empty. Everything I will know must come from "
                      "experience. Give me a number sequence or ask me to `explore`."), \
                {"intent": "knowledge", "counts": by}
        summary = ", ".join((f"{_st(lang, s)} : {n}" if lang == "fr" else f"{n} {s}")
                            for s, n in sorted(by.items(), key=lambda t: -t[1]))
        lines = [_t(lang, f"Ma mémoire contient **{len(items)}** croyances : {summary}.",
                    f"My memory holds **{len(items)}** beliefs: {summary}."),
                 _t(lang, "Dernières règles validées :", "Latest validated rules:")]
        for k in sorted(validated, key=lambda k: -k.id)[:10]:
            lines.append(f"- #{k.id} `{k.key}` : `{k.statement}` ({_pct(k.confidence)})")
        lines.append(_t(lang, "Demande `explique <numéro>` pour voir les épisodes qui justifient une croyance.",
                        "Ask `explain <number>` to see the episodes that justify a belief."))
        return "\n".join(lines), {"intent": "knowledge", "counts": by}

    def stage(self, lang: str) -> tuple:
        a = self.agent
        st = Metacognition.current_stage(a.assessments())
        mem = {t: a.ltm.count(t) for t in ("episodes", "knowledge", "procedures")}
        lines = [_t(lang, f"Stade de développement actuel : **{st['stage']} — {st['name']}**.",
                    f"Current developmental stage: **{st['stage']} — {st['name']}**."),
                 _t(lang, f"Portes franchies (mesurées) : {st['passed']}.", f"Gates passed (measured): {st['passed']}.")]
        if isinstance(st["blocked_by"], list) and st["blocked_by"]:
            b = "; ".join(f"`{x['metric']}` = {x['value']} (requis {x['required']})" for x in st["blocked_by"])
            lines.append(_t(lang, f"Bloqué par : {b}.", f"Blocked by: {b}."))
        lines.append(_t(lang, f"Interactions vécues : {a.interactions} · épisodes : {mem['episodes']} · croyances : "
                              f"{mem['knowledge']} · formes de règles mémorisées : {mem['procedures']}.",
                        f"Interactions: {a.interactions} · episodes: {mem['episodes']} · beliefs: "
                        f"{mem['knowledge']} · remembered rule shapes: {mem['procedures']}."))
        return "\n\n".join(lines), {"intent": "stage", "stage": st["stage"]}

    def limitations(self, lang: str) -> tuple:
        from .self_improvement import limitations
        lim = limitations(self.agent)
        lines = [_t(lang, "Ce que mes propres archives disent de mes limites :",
                    "What my own records say about my limits:")]
        lines.append(_t(lang, f"- problèmes bloqués : {len(lim['stuck_problems'])}",
                        f"- stuck problems: {len(lim['stuck_problems'])}"))
        for s in lim["stuck_problems"][:5]:
            lines.append(f"  - `{s['key']}` ({s['observations']} obs.)")
        lines.append(_t(lang, f"- suites jugées imprévisibles : {len(lim['judged_unpredictable'])}",
                        f"- streams judged unpredictable: {len(lim['judged_unpredictable'])}"))
        lines.append(_t(lang, "- limites d'architecture connues :", "- known architectural limits:"))
        lines += [f"  - {NOTES_FR.get(n, n) if lang == 'fr' else n}" for n in lim["architectural_notes"]]
        lines.append(_t(lang, "- je ne comprends pas le langage : je n'ai appris aucun mot, seulement quelques "
                              "commandes fixées par mes concepteurs.",
                        "- I do not understand language: I learned no words, only a few commands fixed by my "
                        "designers."))
        return "\n".join(lines), {"intent": "limitations", "n": lim["n_identified_from_records"]}

    def questions(self, lang: str) -> tuple:
        qs = Curiosity.knowledge_gaps(self.agent.learners)
        if not qs:
            return _t(lang, "Je n'ai aucune question ouverte : chaque problème rencontré est soit résolu, soit jugé "
                            "imprévisible. Donne-moi quelque chose de nouveau.",
                      "I have no open questions: every problem I met is either solved or judged unpredictable. "
                      "Give me something new."), {"intent": "questions", "n": 0}
        lines = [_t(lang, "Ce que je voudrais savoir (hypothèses encore en concurrence) :",
                    "What I would like to find out (hypotheses still competing):")]
        lines += [f"- {q.text}" for q in qs]
        return "\n".join(lines), {"intent": "questions", "n": len(qs)}

    # ------------------------------------------------------------ exploration
    def explore(self, budget: int, lang: str) -> tuple:
        if self.world_factory is None:
            return _t(lang, "Aucun monde d'expérimentation n'est connecté à cette interface.",
                      "No experimental world is connected to this interface."), {"intent": "explore"}
        n = (self.ltm.get_meta("chat_explorations") or 0) + 1
        self.ltm.set_meta("chat_explorations", n)
        sb = self.world_factory(20000 + n)
        rep = self.agent.explore_sequences(sb, budget)
        env = sb.env_id
        res = {k: lr.status() for k, lr in self.agent.learners.items() if self.agent.learner_env.get(k) == env}
        val = sorted(k for k, s in res.items() if s == "validated")
        unp = sorted(k for k, s in res.items() if s == "unpredictable")
        lines = [_t(lang, f"J'ai exploré un nouveau monde isolé (`{env}`, règles cachées) pendant **{rep.steps}** "
                          f"interactions, en choisissant moi-même quoi observer.",
                    f"I explored a new sandboxed world (`{env}`, hidden rules) for **{rep.steps}** interactions, "
                    f"choosing what to observe myself."),
                 _t(lang, f"- règles validées : **{len(val)}** / {len(res)} flux",
                    f"- rules validated: **{len(val)}** / {len(res)} streams"),
                 _t(lang, f"- flux jugés imprévisibles : {len(unp)}", f"- streams judged unpredictable: {len(unp)}"),
                 _t(lang, f"- arrêt : {REASONS_FR.get(rep.stopped_because, rep.stopped_because or 'budget atteint')}",
                    f"- stopped: {rep.stopped_because or 'budget reached'}")]
        for k in val[:8]:
            top = self.agent.learners[k].best()[0]
            lines.append(f"  - `{k}` : `{top.expr}`")
        return "\n".join(lines), {"intent": "explore", "env_id": env, "steps": rep.steps, "validated": len(val),
                                  "unpredictable": len(unp)}

    # ---------------------------------------------------------------- framing
    def intro(self, lang: str) -> str:
        return _t(lang,
                  "Bonjour. Je suis **GENESIS**, un système cognitif expérimental. Je ne suis **pas** un modèle de "
                  "langage : je n'ai appris aucun mot et je ne comprends que quelques commandes simples.\n\n"
                  "Tout ce que je sais, je l'ai appris par expérience, et je peux montrer les épisodes qui le "
                  "justifient. Donne-moi une suite de nombres entiers, par exemple `3, 7, 11, 15, 19, 23`, et "
                  "j'essaierai d'en découvrir la règle.",
                  "Hello. I am **GENESIS**, an experimental cognitive system. I am **not** a language model: I "
                  "learned no words and understand only a few simple commands.\n\n"
                  "Everything I know I learned from experience, and I can show the episodes that justify it. Give "
                  "me a sequence of integers, e.g. `3, 7, 11, 15, 19, 23`, and I will try to discover its rule.")

    def help(self, lang: str) -> str:
        return _t(lang,
                  "Ce que je peux faire (commandes fixées par mes concepteurs, pas apprises) :\n"
                  "- une **suite d'entiers** (≥ 4) → je cherche la règle en prédisant chaque terme\n"
                  "- `suite 30 34` → je lis d'autres termes de la dernière suite\n"
                  "- `prédis 10` → les 10 termes suivants selon ma règle\n"
                  "- `explique 12` / `pourquoi` → les épisodes qui justifient la croyance #12\n"
                  "- `que sais-tu ?` → mes connaissances\n"
                  "- `explorer 100` → j'expérimente seul dans un monde isolé aux règles cachées\n"
                  "- `stade` · `limites` · `questions`",
                  "What I can do (commands fixed by my designers, not learned):\n"
                  "- a **sequence of integers** (≥ 4) → I look for its rule by predicting each term\n"
                  "- `continue 30 34` → I read more terms of the last sequence\n"
                  "- `predict 10` → the next 10 terms according to my rule\n"
                  "- `explain 12` / `why` → the episodes that justify belief #12\n"
                  "- `what do you know?` → my knowledge\n"
                  "- `explore 100` → I experiment on my own in a sandboxed world with hidden rules\n"
                  "- `stage` · `limitations` · `questions`")

    def not_understood(self, lang: str) -> str:
        return _t(lang,
                  "Je ne comprends pas cette phrase. Je n'ai pas de modèle de langage et je n'ai jamais appris de "
                  "mots : je préfère le dire plutôt que de faire semblant.\n\n" + self.help(lang),
                  "I don't understand that sentence. I have no language model and have never learned words; I'd "
                  "rather say so than pretend.\n\n" + self.help(lang))
