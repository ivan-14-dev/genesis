"""Conversational entry point to GENESIS.

This is **not** a language model and does not pretend to be one. Language understanding comes from the
innate bilingual faculty (``language.py``): a hand-written French/English lexicon and grammar about GENESIS's
own interface and cognition, fixed by the designers, declared at birth and kept separate from acquired
knowledge. It parses questions, negation, elisions and typos, but carries no knowledge about the world.

What the agent says about the world comes only from its own memory:

* A sequence of integers typed by the user is fed term by term through the normal cognitive cycle
  (``Genesis.live``): the agent predicts each term *before* reading it, learns from the error, and
  stores episodes with provenance ``user-provided`` (external information, kept distinct from its own
  observations and experiments).
* Questions about knowledge, stage, limitations or curiosity are answered from long-term memory.
* ``explore`` lets it run curiosity-driven experiments in a fresh sandboxed world supplied by the harness.
* Sentences outside the grammar are not guessed at: the user can teach which command they meant, and the
  word→command associations are learned, validated by forward predictions and stored with evidence
  (``lexicon.py``).

Every answer carries ``meta`` (intent, keys, knowledge ids) so it can be traced back to memory rows.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Callable, Optional

from .adapters import SEQ_VARS, Outcome, Query, Situation
from . import governance, language
from .curiosity import Curiosity
from .governance import PolicyDenied
from .language import FUNCTION, INT_RE, Parse
from .lexicon import ASK_POSTERIOR, Lexicon
from .metacognition import Metacognition
from .perception import Perception
from .semantic import SemanticDialogue
from .grounded_language import ObjectDialogue

MAX_TEXT = 2000
MAX_TERMS = 200
MAX_ABS = 10 ** 9
MAX_EXPLORE = 300
DEFAULT_EXPLORE = 100

YES = language.words_of("YES")
NO = language.words_of("NO")
TOPICAL = ("LIMIT", "QUESTION", "STAGE", "MEMORY", "WORD", "LANGUAGE", "HELP", "EXPLORE")
ACTIONS = ("explore", "predict", "explain", "sequence", "continue")

# Intents a learned word may point to (the innate actions the chat can perform).
TEACHABLE = ("knowledge", "stage", "limitations", "questions", "vocabulary", "help", "intro", "thanks", "explore",
             "predict", "explain")
# Intents that change memory beyond the transcript, and the policy action each one needs.
INTENT_ACTION = {"sequence": "sequence.learn", "continue": "sequence.learn", "explore": "world.explore"}
# Explicit prefix for the in-house language model: free text never reaches it implicitly.
GENERATE_RE = re.compile(r"^\s*(compl[eè]te|g[eé]n[eè]re|generate)\s*:\s*(.+)$", re.I | re.S)
OUTPUT_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
DENIAL_FR = {"role not permitted": "ton rôle ne le permet pas", "domain disabled by the admin":
             "ce domaine est désactivé par l'administrateur", "blocked term": "ton message contient un terme "
             "interdit par l'administrateur", "message too long": "ton message est trop long",
             "unknown action (default deny)": "action inconnue (refus par défaut)"}
INTENT_FR = {"knowledge": "que sais-tu", "stage": "stade", "limitations": "limites", "questions": "questions",
             "vocabulary": "vocabulaire", "help": "aide", "intro": "présentation", "thanks": "merci",
             "explore": "explorer", "predict": "prédis", "explain": "explique"}
COMMANDS = (("que sais-tu ?", "what do you know?"), ("stade", "stage"), ("limites", "limitations"),
            ("questions", "questions"), ("vocabulaire", "vocabulary"), ("aide", "help"),
            ("explorer 100", "explore 100"))

GLOSSARY = {
    "RULE": ("Pour moi, une **règle** est un petit programme (fait de mes primitives arithmétiques et logiques) qui "
             "prédit un terme à partir des précédents. Je n'en considère une comme validée qu'après des "
             "prédictions justes faites *avant* de voir les valeurs.",
             "To me, a **rule** is a small program (built from my arithmetic and logic primitives) that predicts a "
             "term from the previous ones. I only call one validated after correct predictions made *before* "
             "seeing the values."),
    "BELIEF": ("Une **croyance** est une connaissance que j'ai acquise, avec sa provenance, sa confiance, son statut "
               "(hypothèse, candidate, validée, réfutée) et les épisodes qui la justifient (`explique N`).",
               "A **belief** is a piece of knowledge I acquired, with its provenance, confidence, status "
               "(hypothesis, candidate, validated, refuted) and the episodes supporting it (`explain N`)."),
    "MEMORY": ("Ma **mémoire** à long terme (SQLite) contient des épisodes (ce que j'ai vécu, jamais effacés), des "
               "connaissances sémantiques, des procédures, des associations et mon historique d'apprentissage.",
               "My long-term **memory** (SQLite) holds episodes (what I experienced, never erased), semantic "
               "knowledge, procedures, associations and my learning history."),
    "STAGE": ("Un **stade** est une étape de mon développement (0 à 7). Je passe au suivant sur des critères "
              "mesurés, pas avec le temps. Tape `stade` pour voir où j'en suis.",
              "A **stage** is a step of my development (0 to 7). I move on based on measured criteria, not elapsed "
              "time. Type `stage` to see where I am."),
    "SEQUENCE": ("Une **suite** est une liste d'entiers ; je cherche le programme qui produit chaque terme à partir "
                 "des précédents. Envoie-m'en une, par exemple `2, 5, 8, 11, 14`.",
                 "A **sequence** is a list of integers; I look for the program producing each term from the "
                 "previous ones. Send me one, e.g. `2, 5, 8, 11, 14`."),
    "PREDICT": ("Une **prédiction** est ce que j'annonce *avant* d'observer. L'erreur entre prédiction et "
                "observation est mon principal signal d'apprentissage.",
                "A **prediction** is what I announce *before* observing. The error between prediction and "
                "observation is my main learning signal."),
    "QUESTION": ("Ma **curiosité** choisit les expériences qui réduisent le plus mon incertitude (gain "
                 "d'information attendu), pas au hasard. Tape `questions` pour voir mes lacunes actuelles.",
                 "My **curiosity** picks the experiments expected to reduce my uncertainty the most (expected "
                 "information gain), not random ones. Type `questions` to see my current gaps."),
    "WORD": ("Je connais de naissance un petit vocabulaire français/anglais sur moi-même ; les autres **mots** "
             "s'apprennent : tu m'enseignes à quelle commande ils renvoient, et je valide par mes prédictions.",
             "I'm born with a small French/English vocabulary about myself; other **words** are learned: you teach "
             "me which command they refer to and I validate them through my predictions."),
    "EXPLORE": ("**Explorer**, c'est expérimenter seul dans un monde isolé aux règles cachées, en changeant une "
                "variable à la fois pour distinguer causes et corrélations.",
                "**Exploring** means experimenting on my own in a sandboxed world with hidden rules, changing one "
                "variable at a time to tell causes from correlations."),
}


def detect_lang(text: str, fallback: Optional[str] = None) -> str:
    return language.parse(text, fallback).lang


def _t(lang: str, fr: str, en: str) -> str:
    return fr if lang == "fr" else en


STATUS_FR = {"validated": "validée", "candidate": "candidate", "unpredictable": "imprévisible", "refuted": "réfutée",
             "hypothesis": "hypothèse", "dormant": "dormante", "unknown": "inconnue"}
REASONS_FR = {"step limit": "budget atteint", "sandbox budget exhausted": "budget du bac à sable épuisé",
              "time budget": "budget de temps atteint",
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
    """Turns a message into an action of the agent and an answer grounded in its memory.

    ``guard`` (``governance.Guard``) is the identity the engine currently acts for. Every write is checked
    against the admin policy *before* it happens; refusals are answered, never silently dropped, and logged
    in the audit chain. By default the engine runs under the installed policy as the local user."""

    def __init__(self, agent, world_factory: Optional[Callable] = None, object_factory: Optional[Callable] = None,
                 guard: Optional[governance.Guard] = None):
        self.agent = agent
        self.ltm = agent.ltm
        self.world_factory = world_factory
        self.guard = guard or governance.Guard(governance.load_policy(self.ltm), self.ltm)
        self.lexicon = Lexicon(self.ltm, TEACHABLE, lambda: agent.session, ignore=YES | NO)
        self.semantic = SemanticDialogue(agent)
        self.objects = ObjectDialogue(agent, object_factory)
        self.semantic.require = self.objects.require = self._require

    def _require(self, action: str, target=None, detail: Optional[dict] = None) -> None:
        self.guard.require(action, target, detail)

    def _precheck(self, intent: str) -> None:
        """Refuses (and logs) an intent the identity may not perform, before anything is learned about it."""
        action = INTENT_ACTION.get(intent)
        if action and not self.guard.allowed(action):
            self.guard.require(action)

    @property
    def owner(self) -> str:
        return self.guard.actor

    # ------------------------------------------------------------ transcript
    def new_conversation(self, lang: Optional[str] = None) -> dict:
        self.guard.require("chat.read")
        cid = self.ltm.add_conversation(None, lang if lang in ("fr", "en") else None, owner=self.owner)
        self.ltm.commit()
        return self.ltm.conversation(cid)

    def conversation(self, cid: int) -> Optional[dict]:
        return self.ltm.conversation(cid, owner=self.owner)

    def conversations(self) -> list:
        return self.ltm.conversations(owner=self.owner)

    def messages(self, cid: int) -> list:
        if self.conversation(cid) is None:
            raise KeyError(cid)
        return self.ltm.messages(cid)

    def send(self, cid: int, text: str, lang_hint: Optional[str] = None) -> dict:
        conv = self.conversation(cid)
        if conv is None:
            raise KeyError(cid)
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        gen = GENERATE_RE.match(text)
        # The prompt may be in any language: the command word decides the language of the reply.
        lang = (("en" if gen.group(1).lower() in ("complete", "generate") else "fr") if gen
                else detect_lang(text[:MAX_TEXT], conv.get("lang") or lang_hint))
        try:
            self.guard.require("chat.read", f"conversation:{cid}")
            self.guard.check_text(text)
        except PolicyDenied as exc:
            # The refused content is not stored: it may be exactly what the admin wants kept out of memory.
            user = self.ltm.add_message(cid, "user", _t(lang, "[message refusé par la politique]",
                                                        "[message refused by policy]"), {"refused": exc.reason})
            reply, meta = self.denied(exc, lang)
        else:
            user = self.ltm.add_message(cid, "user", text)
            try:
                reply, meta = self.respond(cid, text, lang)
            except PolicyDenied as exc:
                reply, meta = self.denied(exc, lang)
        meta = {"lang": lang, **meta}
        bot = self.ltm.add_message(cid, "genesis", reply, meta)
        self.ltm.update_conversation(cid, lang=lang, title=conv["title"] or user["text"][:60])
        self.ltm.commit()
        return {"user": user, "reply": bot, "conversation": self.ltm.conversation(cid)}

    def denied(self, exc: PolicyDenied, lang: str) -> tuple:
        reason = DENIAL_FR.get(exc.reason, exc.reason) if lang == "fr" else exc.reason
        if exc.reason.startswith("role '"):
            reason = _t(lang, "ton rôle ne le permet pas", exc.reason)
        return _t(lang, f"Je ne peux pas faire cela (`{exc.action}`) : {reason}. Les règles sont fixées par "
                        "l'administrateur ; rien n'a été appris ni modifié.",
                  f"I can't do that (`{exc.action}`): {reason}. The rules are set by the administrator; nothing "
                  "was learned or changed."), {"intent": "policy-denied", "action": exc.action,
                                               "reason": exc.reason}

    # ---------------------------------------------------------------- intents
    def _innate(self, cid: int, p: Parse, text: str) -> Optional[str]:
        """Maps the innate language faculty's parse (concepts, question type, negation) to an intent."""
        intent = self._intent(cid, p, text)
        if intent in ACTIONS and p.negated and p.qtype is None:
            return "declined"
        return intent

    def _intent(self, cid: int, p: Parse, text: str) -> Optional[str]:
        nums, has = p.numbers, p.has
        last_seq = self._last(cid, "key")
        content = [t for t in p.tokens if t not in FUNCTION]
        if has("EXPLAIN") and (nums or not has(*TOPICAL)):
            return "explain"
        if p.qtype == "why" and (nums or self._last(cid, "knowledge_id") is not None) and not has(*TOPICAL):
            return "explain"
        if has("CONTINUE") and nums and last_seq:
            return "continue"
        if len(nums) >= 4:
            return "sequence"
        if 1 <= len(nums) <= 3 and not content and last_seq:
            return "continue"
        if has("PREDICT") or (last_seq and (has("NEXT") or (has("CONTINUE") and not nums))):
            return "predict"
        if has("HOWAREYOU"):
            return "smalltalk"
        if any(g in p.concepts for g in GLOSSARY) and (has("MEANING") or (
                p.qtype == "what" and not has("SELF", "USER", "KNOW", "LEARN", "CAN"))):
            return "define"
        if has("LANGUAGE") or (has("SELF") and has("UNDERSTAND", "SPEAK") and not p.negated
                               and not has(*TOPICAL, "KNOW", "RULE", "BELIEF")):
            return "languages"
        if has("USER") and has("UNDERSTAND", "KNOW") and p.negated:
            return "help"
        for concepts, intent in ((("EXPLORE",), "explore"), (("WORD",), "vocabulary"), (("LIMIT",), "limitations"),
                                 (("QUESTION",), "questions")):
            if has(*concepts):
                return intent
        if has("KNOW") and p.negated:
            return "questions"
        if has("HELP") or (has("CAN") and has("SELF")) or text.strip() == "?":
            return "help"
        if has("KNOW", "LEARN", "BELIEF", "RULE"):
            return "knowledge"
        if has("STAGE", "MEMORY"):
            return "stage"
        if has("GREET") or (p.qtype == "who" and has("SELF")) or (has("NAME") and has("SELF")):
            return "intro"
        if has("THANKS"):
            return "thanks"
        if has("BYE"):
            return "bye"
        if 0 < len(nums) < 4:
            return "too-short"
        return None

    def run(self, cid: int, intent: str, nums: list, lang: str, p: Optional[Parse] = None) -> tuple:
        last_seq = self._last(cid, "key")
        if intent in ("define", "languages", "smalltalk", "bye", "declined"):
            return getattr(self, intent)(p or language.parse("", lang), lang)
        if intent == "explain":
            kid = nums[0] if nums else self._last(cid, "knowledge_id")
            if kid is None:
                return _t(lang, "Quelle croyance ? Donne son numéro, par exemple `explique 3`.",
                          "Which belief? Give its number, e.g. `explain 3`."), {"intent": "explain", "found": False}
            return self.explain(kid, lang)
        if intent == "sequence":
            return self.learn_sequence(cid, nums, lang)
        if intent == "continue":
            return self.learn_sequence(cid, nums, lang, key=last_seq)
        if intent == "predict":
            if not last_seq:
                return _t(lang, "Donne-moi d'abord une suite de nombres : je ne peux prédire que ce que j'ai appris.",
                          "Give me a number sequence first: I can only predict what I have learned."), \
                    {"intent": "predict"}
            return self.predict(last_seq, min(max(nums[0], 1), 20) if nums else 5, lang)
        if intent == "explore":
            return self.explore(min(max(nums[0], 10), MAX_EXPLORE) if nums else DEFAULT_EXPLORE, lang)
        if intent in ("vocabulary", "limitations", "questions", "knowledge", "stage"):
            return getattr(self, intent)(lang)
        if intent == "help":
            return self.help(lang), {"intent": "help"}
        if intent == "intro":
            return self.intro(lang), {"intent": "intro"}
        if intent == "thanks":
            return _t(lang, "Avec plaisir. Donne-moi d'autres suites à analyser quand tu veux.",
                      "You're welcome. Send me more sequences whenever you like."), {"intent": "thanks"}
        return _t(lang, "Il me faut au moins 4 nombres entiers pour chercher une règle : je prédis chaque terme "
                        "à partir des 3 précédents.",
                  "I need at least 4 integers to look for a rule: I predict each term from the previous 3."), \
            {"intent": "too-short"}

    def respond(self, cid: int, text: str, lang: str) -> tuple:
        gen = GENERATE_RE.match(text)
        if gen:
            return self.generate(gen.group(2).strip(), lang)
        objects = self.objects.respond(cid, text, lang)
        if objects is not None:
            return objects
        semantic = self.semantic.respond(cid, text, lang)
        if semantic is not None:
            return semantic
        reply, meta = self._respond(cid, text, lang)
        p = language.parse(text, lang)
        if p.corrected and meta.get("intent") not in ("not-understood", "unknown-topic"):
            fixes = ", ".join(f"« {a} » → « {b} »" for a, b in p.corrected.items())
            reply += "\n\n" + _t(lang, f"_(J'ai corrigé une faute probable : {fixes}.)_",
                                  f"_(I corrected a probable typo: {fixes}.)_")
            meta = {**meta, "corrected": p.corrected}
        return reply, meta

    def _respond(self, cid: int, text: str, lang: str) -> tuple:
        p = language.parse(text, lang)
        nums = p.numbers
        n_tok = len(p.tokens)
        prev = self._prev_meta(cid)
        innate = self._innate(cid, p, text)
        teachable = innate if innate in TEACHABLE else None
        if prev.get("intent") == "unknown-topic" and text.strip().lower() not in (prev.get("choices") or []):
            # I understood the sentence's structure, so only an explicit pick of a command counts as teaching.
            teachable = None
        yes = p.has("YES") and not p.has("NO") and n_tok <= 4 and not nums
        no = p.has("NO") and not p.has("YES") and n_tok <= 4 and not nums
        u = prev.get("utterance")

        if prev.get("intent") == "clarify" and u:
            proposed = prev["proposed"]
            if yes:
                self._precheck(proposed)
                info = self._teach_word(u, proposed, "confirmed", prev.get("posterior"))
                return self._with_note(self.run(cid, proposed, prev.get("nums") or [], lang),
                                       self._learn_note(info, u, proposed, "confirmed", lang), info is not None)
            if no:
                info = self._teach_word(u, proposed, "rejected", prev.get("posterior"))
                return self._ask_teacher(u, prev.get("nums") or [], lang, rejected=proposed, info=info)
            # Any other message is not taken as an answer: ignoring a question is not evidence.
        elif prev.get("intent") in ("not-understood", "unknown-topic", "teach-request") and u and teachable:
            self._precheck(teachable)
            info = self._teach_word(u, teachable, "taught")
            return self._with_note(self.run(cid, teachable, nums or prev.get("nums") or [], lang),
                                   self._learn_note(info, u, teachable, "taught", lang), info is not None)
        elif prev.get("interpreted") and u and no and not innate:
            info = self._teach_word(u, prev["intent"], "rejected", prev.get("posterior"))
            return self._ask_teacher(u, prev.get("nums") or [], lang, rejected=prev["intent"], info=info)

        if prev.get("intent") in ("not-understood", "unknown-topic", "teach-request") and u and (yes or no):
            return self._ask_teacher(u, prev.get("nums") or [], lang, rejected=None, info=None)
        if innate:
            return self.run(cid, innate, nums, lang, p)
        if yes or no:
            return _t(lang, "Je ne t'ai rien demandé auquel répondre par oui ou non.",
                      "I didn't ask you anything to answer yes or no to."), {"intent": "yes-no"}
        guess = self.lexicon.interpret(text)
        if guess and guess.act:
            reply, meta = self.run(cid, guess.intent, nums, lang)
            words = ", ".join(f"« {w} »" for w in guess.words)
            note = _t(lang, f"_J'ai compris ta phrase comme `{INTENT_FR[guess.intent]}` grâce aux mots que tu m'as "
                            f"appris ({words} ; confiance {_pct(guess.posterior)}). Réponds `non` si je me "
                            f"trompe._",
                      f"_I took your sentence to mean `{guess.intent}` thanks to words you taught me ({words}; "
                      f"confidence {_pct(guess.posterior)}). Reply `no` if I'm wrong._")
            return note + "\n\n" + reply, {**meta, "interpreted": True, "utterance": text, "nums": nums,
                                            "posterior": guess.posterior, "lexical_ids": guess.kids}
        if guess and guess.posterior >= ASK_POSTERIOR:
            words = ", ".join(f"« {w} »" for w in guess.words)
            name = INTENT_FR[guess.intent] if lang == "fr" else guess.intent
            return _t(lang, f"Je ne suis pas sûr. D'après les mots que tu m'as appris ({words}), je pense que tu "
                            f"veux dire **`{name}`** (P = {_pct(guess.posterior)} parmi mes commandes), mais cette "
                            f"association n'est pas encore validée par mes prédictions. C'est bien ça ?",
                      f"I'm not sure. From the words you taught me ({words}), I think you mean **`{name}`** "
                      f"(P = {_pct(guess.posterior)} among my commands), but that mapping isn't validated by my "
                      f"predictions yet. Is that right?"), \
                {"intent": "clarify", "proposed": guess.intent, "utterance": text, "nums": nums,
                 "posterior": guess.posterior, "lexical_ids": guess.kids,
                 "choices": [_t(lang, "oui", "yes"), _t(lang, "non", "no")]}
        if p.topic and (p.qtype or p.understood >= 0.5):
            return self.unknown_topic(p, lang), {"intent": "unknown-topic", "utterance": text, "nums": nums,
                                                 "topic": p.topic, "qtype": p.qtype,
                                                 "choices": self._command_choices(lang)}
        return self.not_understood(lang), {"intent": "not-understood", "utterance": text, "nums": nums,
                                           "choices": self._command_choices(lang)}

    # ------------------------------------------------------- word learning
    def _prev_meta(self, cid: int) -> dict:
        for m in reversed(self.ltm.messages(cid)):
            if m["role"] == "genesis":
                return m["meta"] or {}
        return {}

    @staticmethod
    def _command_choices(lang: str) -> list:
        return [_t(lang, fr, en) for fr, en in COMMANDS]

    @staticmethod
    def _with_note(answer: tuple, note: str, learned: bool = True) -> tuple:
        reply, meta = answer
        return reply + "\n\n" + note, {**meta, "learned_word": learned}

    def _teach_word(self, utterance: str, intent: str, how: str, posterior=None) -> Optional[dict]:
        """Word learning is a write governed by ``lexicon.teach``; without it the command still runs."""
        try:
            self.guard.require("lexicon.teach", intent, {"utterance": utterance[:80], "how": how})
        except PolicyDenied:
            return None
        return self.lexicon.learn(utterance, intent, how, posterior)

    def _learn_note(self, info: Optional[dict], utterance: str, intent: str, how: str, lang: str) -> str:
        name = INTENT_FR[intent] if lang == "fr" else intent
        quoted = utterance if len(utterance) <= 60 else utterance[:57] + "…"
        if info is None:
            return _t(lang, f"_« {quoted} » → `{name}` : association pas mémorisée, ton rôle ne permet pas de "
                            "m'enseigner du vocabulaire._",
                      f"_“{quoted}” → `{name}`: mapping not stored, your role may not teach me vocabulary._")
        parts = [_t(lang, f"📘 Noté : « {quoted} » → `{name}`.", f"📘 Noted: “{quoted}” → `{name}`.")]
        if info["new_words"]:
            parts.append(_t(lang, f"Nouveaux mots : {', '.join(info['new_words'])}.",
                            f"New words: {', '.join(info['new_words'])}."))
        val = [w for w, i, st in info["changes"] if i == intent and st == "validated"]
        if val:
            parts.append(_t(lang, f"Associations validées par mes prédictions : {', '.join(val)}.",
                            f"Mappings validated by my predictions: {', '.join(val)}."))
        else:
            parts.append(_t(lang, "Je te demanderai confirmation avant de m'en servir seul.",
                            "I'll ask for confirmation before relying on it."))
        return "_" + " ".join(parts) + "_"

    def _ask_teacher(self, utterance: str, nums: list, lang: str, rejected: Optional[str], info) -> tuple:
        name = (INTENT_FR[rejected] if lang == "fr" else rejected) if rejected else None
        head = (_t(lang, f"Compris, ce n'était pas `{name}` : je l'ai noté comme une erreur de prédiction. ",
                   f"Understood, it wasn't `{name}`: I recorded it as a prediction error. ") if name else "")
        return head + _t(lang, "Quelle commande voulais-tu dire ? Choisis-la ci-dessous ou tape-la, et je "
                               "l'associerai à ta phrase.",
                         "Which command did you mean? Pick it below or type it, and I'll associate it with your "
                         "sentence."), \
            {"intent": "teach-request", "utterance": utterance, "nums": nums,
             "choices": self._command_choices(lang), "rejected": rejected}

    def vocabulary(self, lang: str) -> tuple:
        es = self.lexicon.entries()
        grounded = [k for k in self.ltm.knowledge(kind="grounded-word")
                    if k.status not in ("refuted", "dormant")]
        constructions = [k for k in self.ltm.knowledge(kind="construction") if k.status != "refuted"]
        if not es and not grounded and not constructions:
            return _t(lang, "Je n'ai aucune interprétation de mot active. Quand je ne comprends pas une phrase, réponds "
                            "avec la commande que tu voulais dire : j'associerai tes mots à cette commande.",
                     "I have no active word interpretations. When I don't understand a sentence, reply with the command "
                      "you meant and I'll associate your words with it."), {"intent": "vocabulary", "n": 0}
        n_val = sum(1 for e in es if e["status"] == "validated")
        lines = [_t(lang, f"**{self.lexicon.size()}** mot(s) associé(s) à des commandes "
                          f"({n_val} association(s) validée(s) par mes prédictions).",
                    f"**{self.lexicon.size()}** word(s) associated with commands "
                    f"({n_val} mapping(s) validated by my predictions).")]
        es.sort(key=lambda e: (e["status"] != "validated", -e["p"], e["word"]))
        for e in es[:15]:
            name = INTENT_FR[e["intent"]] if lang == "fr" else e["intent"]
            lines.append(f"- #{e['id']} « {e['word']} » → `{name}` — {_st(lang, e['status'])}, P = {_pct(e['p'])}, "
                         + _t(lang, f"prédictions {e['ok']} justes / {e['bad']} fausses",
                              f"predictions {e['ok']} right / {e['bad']} wrong"))
        if grounded:
            lines.append(_t(lang, "Interprétations apprises à partir d'objets (accord avec les exemples, "
                                  "pas de définition universelle) :",
                            "Interpretations learned from objects (agreement with examples, "
                            "not universal definitions):"))
            for k in grounded[:15]:
                lines.append(f"- #{k.id} « {k.expr['word']} » — `{k.context}` — {_st(lang, k.status)}; "
                             f"`{json.dumps(k.expr['meanings'])}`")
        if constructions:
            lines.append(_t(lang, "Constructions apprises (nombres = positions variables) :",
                            "Learned constructions (numbers = variable slots):"))
            for k in constructions[:15]:
                lines.append(f"- #{k.id} `{k.statement}` — `{k.context}` — {_st(lang, k.status)}")
        return "\n".join(lines), {"intent": "vocabulary", "n": len(es) + len(grounded),
                                  "grounded": len(grounded), "constructions": len(constructions),
                                  "validated": n_val + sum(k.status == "validated" for k in grounded)}

    def _last(self, cid: int, field: str):
        for m in reversed(self.ltm.messages(cid)):
            if m["role"] == "genesis" and m["meta"] and m["meta"].get(field) is not None:
                return m["meta"][field]
        return None

    # ----------------------------------------------------------- sequences
    def learn_sequence(self, cid: int, terms: list, lang: str, key: Optional[str] = None) -> tuple:
        max_terms = min(MAX_TERMS, self.guard.limit("max_terms_per_message"))
        if len(terms) > max_terms or any(abs(v) > MAX_ABS for v in terms):
            return _t(lang, f"Je refuse cette entrée : au plus {max_terms} termes, chacun entre -10⁹ et 10⁹.",
                      f"Input refused: at most {max_terms} terms, each between -10⁹ and 10⁹."), \
                {"intent": "refused"}
        agent = self.agent
        env_id = f"conversation-{cid}"
        appended = key is not None
        if key is None:
            n = sum(1 for k in agent.learners if k.startswith(env_id + "/"))
            key = f"{env_id}/seq{n + 1}"
        already = self.ltm.db.execute("SELECT COUNT(*) FROM episodes WHERE context=?", (key,)).fetchone()[0]
        cap = self.guard.limit("max_sequence_terms")
        if already + len(terms) > cap:
            # Every term re-scores the hypotheses on the whole history: the admin bounds that cost.
            return _t(lang, f"Je refuse : cette suite dépasserait {cap} termes (limite fixée par l'administrateur ; "
                            f"déjà lus : {already}). Commence une nouvelle suite.",
                      f"Refused: this sequence would exceed {cap} terms (limit set by the administrator; already "
                      f"read: {already}). Start a new sequence."), {"intent": "refused", "reason": "max_sequence_terms"}
        self.guard.require("sequence.learn", key, {"terms": len(terms)})
        first_ep = self.ltm.db.execute("SELECT COALESCE(MAX(id), 0) FROM episodes").fetchone()[0]
        adapter = UserSequenceAdapter(agent, env_id, key, terms)
        rep = agent.live(adapter, len(terms), "curiosity", stop_when_idle=False,
                         max_seconds=self.guard.limit("max_seconds_per_message"))
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
            lines.append(_t(lang, f"J'ai lu {len(trace)} terme(s) de plus de la même suite.",
                            f"I read {len(trace)} more term(s) of the same sequence."))
        if rep.stopped_because == "time budget":
            meta["stopped"] = "time budget"
            lines.append(_t(lang, f"⏱️ Budget de temps atteint : je n'ai lu que {len(trace)}/{len(terms)} terme(s). "
                                  "Renvoie les suivants avec `suite …`.",
                            f"⏱️ Time budget reached: I only read {len(trace)}/{len(terms)} term(s). Send the rest "
                            "with `continue …`."))
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
        confidence = (_t(lang, "enseignement externe non vérifié", "unverified external teaching")
                      if it["kind"] == "semantic" else
                      _t(lang, f"confiance {_pct(it['confidence'])}", f"confidence {_pct(it['confidence'])}"))
        if it["kind"] in ("grounded-word", "construction"):
            confidence = _t(
                lang, f"accord avec les annotations nouvelles {_pct(it['confidence'])}, pas une certitude",
                f"agreement with novel annotations {_pct(it['confidence'])}, not certainty")
        lines = [f"**#{kid} — `{it['statement']}`** ({_st(lang, it['status'])}, {confidence})",
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
        items = [k for k in self.ltm.knowledge() if k.kind != "lexical"]
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
                 _t(lang, "Dernières croyances validées :", "Latest validated beliefs:")]
        for k in sorted(validated, key=lambda k: -k.id)[:10]:
            lines.append(f"- #{k.id} `{k.key}` : `{k.statement}` ({_pct(k.confidence)})")
        taught = [k for k in items if k.kind == "semantic"]
        if taught:
            lines.append(_t(lang, "Enseignements externes (non vérifiés) :",
                            "External teachings (unverified):"))
            for k in taught[-10:]:
                lines.append(f"- #{k.id} `{k.statement}` ({k.status})")
        if self.lexicon.size():
            lines.append(_t(lang, f"Et **{self.lexicon.size()}** mot(s) appris de toi (voir `vocabulaire`).",
                            f"Plus **{self.lexicon.size()}** word(s) learned from you (see `vocabulary`)."))
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
        n = self.lexicon.size()
        lines.append(_t(lang, f"- j'ai appris {n} mot(s) associés à des commandes. Je peux aussi acquérir des "
                              "relations enseignées et des sens de mots liés aux objets de l'atelier. "
                              "Je peux apprendre des constructions simples par des exemples. "
                              "L'alignement des mots et les opérations logiques restent fournis par mes concepteurs : "
                              "ce n'est pas une compréhension humaine générale.",
                        f"- I learned {n} word(s) associated with commands. I can also acquire taught relations "
                        "and word meanings linked to workshop objects, and learn simple constructions from examples. "
                        "Word alignment and logical operations remain designer-provided: "
                        "this is not human-level general understanding."))
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
        budget = min(budget, self.guard.limit("max_explore_steps"))
        self.guard.require("world.explore", None, {"budget": budget})
        n = (self.ltm.get_meta("chat_explorations") or 0) + 1
        self.ltm.set_meta("chat_explorations", n)
        sb = self.world_factory(20000 + n)
        rep = self.agent.explore_sequences(sb, budget, max_seconds=self.guard.limit("max_seconds_per_message"))
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

    # ------------------------------------------------- in-house language model
    def generate(self, prompt: str, lang: str) -> tuple:
        """Continues a text with the admin-approved model trained here. The output is labelled as unverified,
        filtered (personal data, blocked terms) and never stored as knowledge: only the transcript keeps it."""
        from . import corpus
        from .lm import registry
        self.guard.require("lm.generate", None, {"prompt_chars": len(prompt)})
        meta = {"intent": "lm-generate"}
        try:
            model = registry.load_active(self.ltm)
        except (registry.LMIntegrityError, KeyError) as exc:
            return _t(lang, f"Mon modèle de langage interne est refusé : son intégrité ne se vérifie plus ({exc}). "
                            "L'administrateur doit le réapprouver ou en entraîner un autre.",
                      f"My in-house language model is refused: its integrity no longer verifies ({exc}). The "
                      "administrator must re-approve it or train another one."), {**meta, "error": "integrity"}
        if model is None:
            return _t(lang, "Aucun modèle de langage interne n'est actif. Il doit être entraîné ici, sur des "
                            "documents approuvés par l'administrateur (`python -m genesis corpus add`, "
                            "`corpus approve`, `lm train`, `lm approve`). Je n'utilise aucun modèle externe.",
                      "No in-house language model is active. It has to be trained here, on documents approved by "
                      "the administrator (`python -m genesis corpus add`, `corpus approve`, `lm train`, "
                      "`lm approve`). I use no external model."), {**meta, "active": False}
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        out = model.generate(prompt, self.guard.limit("max_generate_tokens"), seed=seed,
                             max_seconds=self.guard.limit("max_seconds_per_message"))
        text, redactions = corpus.redact(OUTPUT_CONTROL.sub("", out["text"]))
        meta = {**meta, "model_id": model.id, "tokens": out["tokens"], "stopped": out["stopped"], "seed": seed}
        if redactions:
            meta["redactions"] = redactions
        if self.guard.policy.blocked(text):
            return _t(lang, "Le texte généré contenait un terme interdit par l'administrateur : je ne l'affiche pas.",
                      "The generated text contained a term blocked by the administrator: I won't show it."), \
                {**meta, "withheld": True}
        head = _t(lang, f"_Texte généré par mon modèle interne {model.describe('fr')} — **non vérifié, ce n'est "
                        "pas une connaissance** et rien n'est mémorisé comme vrai._",
                  f"_Text generated by my in-house model {model.describe('en')} — **unverified, not knowledge**, "
                  "and nothing is stored as true._")
        return f"{head}\n\n{prompt}{text}", meta

    # ------------------------------------------------------ language faculty
    def define(self, p: Parse, lang: str) -> tuple:
        term = next(g for g in GLOSSARY if g in p.concepts)
        fr, en = GLOSSARY[term]
        return _t(lang, fr, en), {"intent": "define", "term": term}

    def languages(self, p: Parse, lang: str) -> tuple:
        m = language.faculty_manifest()
        return _t(lang,
                  f"Je comprends le **français** et l'**anglais** de base : c'est une faculté **innée**, écrite par "
                  f"mes concepteurs ({m['n_forms']} formes de mots, {len(m['concepts'])} concepts, plus la grammaire "
                  "des questions, de la négation, des élisions et des fautes de frappe courantes). Elle est déclarée "
                  "comme telle dans mon état initial, séparément de ce que j'ai appris.\n\n"
                  "Ce n'est **pas** un modèle de langage : je comprends ce qui parle de moi (connaissances, règles, "
                  "suites, stade, limites, curiosité, exploration...), pas n'importe quel sujet. Les mots qui me "
                  f"manquent, tu peux me les apprendre ({self.lexicon.size()} mots associés aux commandes). "
                  "Dans `atelier objets`, tu peux aussi m'enseigner des mots à partir d'exemples observés.",
                  f"I understand basic **French** and **English**: it is an **innate** faculty written by my "
                  f"designers ({m['n_forms']} word forms, {len(m['concepts'])} concepts, plus the grammar of "
                  "questions, negation, contractions and common typos). It is declared as such in my initial state, "
                  "separately from what I have learned.\n\n"
                  "It is **not** a language model: I understand what is about me (knowledge, rules, sequences, "
                  "stage, limitations, curiosity, exploring...), not any topic. You can teach me the words I lack "
                  f"({self.lexicon.size()} words associated with commands). In `object workshop`, "
                  "you can also teach me words through observed examples."), {"intent": "languages"}

    def smalltalk(self, p: Parse, lang: str) -> tuple:
        n_ep = self.ltm.count("episodes")
        n_val = self.ltm.db.execute("SELECT COUNT(*) FROM knowledge WHERE status='validated' AND kind!='lexical'"
                                    ).fetchone()[0]
        extra = _t(lang, f" et {n_val} connaissance(s) validée(s)", f" and {n_val} validated belief(s)")
        return _t(lang, f"Je n'ai pas d'émotions, donc je ne vais ni bien ni mal. Ce que je peux mesurer : "
                        f"{n_ep} épisode(s) vécus en mémoire{extra}. Donne-moi une suite si tu veux que j'apprenne "
                        "quelque chose de nouveau.",
                  f"I don't have feelings, so I'm neither well nor unwell. What I can measure: {n_ep} episode(s) "
                  f"experienced in memory{extra}. Give me a sequence if you'd like me to learn something new."), \
            {"intent": "smalltalk"}

    def bye(self, p: Parse, lang: str) -> tuple:
        return _t(lang, "Au revoir. Tout ce que j'ai appris aujourd'hui reste dans ma mémoire pour la prochaine fois.",
                  "Goodbye. Everything I learned today stays in my memory for next time."), {"intent": "bye"}

    def declined(self, p: Parse, lang: str) -> tuple:
        return _t(lang, "D'accord, je ne le fais pas.", "Alright, I won't."), {"intent": "declined"}

    def unknown_topic(self, p: Parse, lang: str) -> str:
        words = ", ".join(f"« {w} »" for w in p.topic[:6])
        if p.qtype:
            head = _t(lang, f"Je repère une question à propos de {words}, mais je ne peux pas répondre à cette "
                            "formulation depuis ma mémoire. Essaie une question simple comme "
                            "« Nala est-elle un animal ? » ou « que sais-tu de Nala ? ».",
                      f"I detect a question about {words}, but I cannot answer this wording from my memory. "
                      "Try a simple question like “Is Nala an animal?” or “what do you know about Nala?”.")
        else:
            head = _t(lang, f"Je comprends la structure de ta phrase, mais pas les mots {words} : ils ne sont ni "
                            "dans mon vocabulaire inné ni dans ce que tu m'as appris.",
                      f"I understand the structure of your sentence, but not the words {words}: they are neither in "
                      "my innate vocabulary nor among those you taught me.")
        return head + _t(lang, " Je préfère le dire plutôt que d'inventer une réponse.\n\nSi ta phrase correspond à "
                               "l'une de mes commandes, choisis-la ci-dessous : j'apprendrai ces mots.",
                         " I'd rather say so than make up an answer.\n\nIf your sentence matches one of my commands, "
                         "pick it below and I'll learn those words.")

    # ---------------------------------------------------------------- framing
    def intro(self, lang: str) -> str:
        return _t(lang,
                  "Bonjour. Je suis **GENESIS**, un système cognitif expérimental. Je ne suis **pas** un modèle de "
                  "langage : je comprends le français et l'anglais de base (une faculté innée, limitée à ce qui "
                  "parle de moi et à des phrases factuelles simples), et j'apprends les mots et les relations "
                  "que tu m'enseignes.\n\n"
                  "Tout ce que je sais du monde, je l'ai appris par expérience, et je peux montrer les épisodes qui "
                  "le justifient. Donne-moi une suite de nombres entiers, par exemple `3, 7, 11, 15, 19, 23`, et "
                  "j'essaierai d'en découvrir la règle.",
                  "Hello. I am **GENESIS**, an experimental cognitive system. I am **not** a language model: I "
                  "understand basic French and English (an innate faculty, limited to what is about me), and I learn "
                  "the words and simple factual relations you teach me.\n\n"
                  "Everything I know about the world I learned from experience, and I can show the episodes that "
                  "justify it. Give me a sequence of integers, e.g. `3, 7, 11, 15, 19, 23`, and I will try to "
                  "discover its rule.")

    def help(self, lang: str) -> str:
        return _t(lang,
                  "Parle-moi normalement en français ou en anglais. Ce que je sais faire :\n"
                  "- une **suite d'entiers** (≥ 4) → je cherche la règle en prédisant chaque terme\n"
                  "- « continue avec 30 34 » → je lis d'autres termes de la dernière suite\n"
                  "- « prédis les 10 suivants » → la suite selon ma règle\n"
                  "- « pourquoi ? » / « explique la 12 » → les épisodes qui justifient une croyance\n"
                  "- « qu'est-ce que tu sais ? » · « où en es-tu ? » · « quelles sont tes limites ? »\n"
                  "- « qu'est-ce que tu ignores ? » · « quels mots t'ai-je appris ? »\n"
                  "- « va explorer 100 fois » → j'expérimente seul dans un monde isolé aux règles cachées\n"
                  "- « c'est quoi une règle ? » → je définis mes propres notions\n"
                  "- « Nala est un chat; Les chats sont des animaux » → enseignements non vérifiés\n"
                  "- « Nala est-elle un animal ? » puis « pourquoi ? » → déduction et prémisses\n"
                  "- « que sais-tu de Nala ? » → rappel ; « correction : Nala n'est pas un chat » → correction\n"
                  "- `retire le fait 12` → rétracter un enseignement sans effacer son historique\n"
                  "- `atelier objets` → apprendre des mots à partir d'objets et de leurs effets observés\n"
                  "- `apprends phrase : prends dax sans mip => dax et pas mip` → enseigner une construction "
                  "avec des mots déjà appris dans l'atelier\n"
                  "- `complète : il était une fois` → texte généré par mon petit modèle de langage entraîné ici "
                  "(s'il a été approuvé par l'administrateur), toujours marqué non vérifié\n"
                  "- un mot que je ne connais pas → dis-moi quelle commande tu voulais : j'apprendrai ce mot",
                  "Talk to me normally in English or French. What I can do:\n"
                  "- a **sequence of integers** (≥ 4) → I look for its rule by predicting each term\n"
                  "- “continue with 30 34” → I read more terms of the last sequence\n"
                  "- “predict the next 10” → the sequence according to my rule\n"
                  "- “why?” / “explain belief 12” → the episodes that justify a belief\n"
                  "- “what do you know?” · “how far along are you?” · “what are your limitations?”\n"
                  "- “what don't you know?” · “which words did I teach you?”\n"
                  "- “go explore 100 times” → I experiment on my own in a sandboxed world with hidden rules\n"
                  "- “what is a rule?” → I define my own notions\n"
                  "- “Nala is a cat; All cats are animals” → unverified teachings\n"
                  "- “Is Nala an animal?” then “why?” → deduction and premises\n"
                  "- “what do you know about Nala?” → recall; “correction: Nala is not a cat” → correction\n"
                  "- `retract fact 12` → retract a teaching without erasing its history\n"
                  "- `object workshop` → learn words from objects and their observed effects\n"
                  "- `learn phrase: fetch dax without mip => dax and not mip` → teach a construction "
                  "using words already learned in the workshop\n"
                  "- `complete: once upon a time` → text generated by my small language model trained here "
                  "(if the administrator approved one), always labelled unverified\n"
                  "- a word I don't know → tell me which command you meant: I'll learn that word")

    def not_understood(self, lang: str) -> str:
        return _t(lang,
                  "Je ne comprends pas cette phrase. Je connais le français et l'anglais de base, mais seulement "
                  "pour parler de moi et de mes expériences, et ces mots-là me sont inconnus. Je préfère le dire "
                  "plutôt que de faire semblant.\n\n"
                  "**Apprends-les-moi :** réponds avec la commande que tu voulais dire (choisis-la ci-dessous ou "
                  "tape-la) et j'associerai tes mots à cette commande. La prochaine fois, je proposerai cette "
                  "interprétation, puis je l'utiliserai seul une fois validée par mes prédictions.\n\n"
                  + self.help(lang),
                  "I don't understand that sentence. I know basic English and French, but only to talk about myself "
                  "and my experiences, and these words are unknown to me. I'd rather say so than pretend.\n\n"
                  "**Teach me:** reply with the command you meant (pick it below or type it) and I'll associate "
                  "your words with it. Next time I'll propose that interpretation, and use it on my own once my "
                  "predictions have validated it.\n\n" + self.help(lang))
