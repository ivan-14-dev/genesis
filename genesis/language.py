"""Innate bilingual (French / English) language faculty.

This is **not** a language model and contains no pretrained weights. It is a small, hand-written,
fully inspectable grammar-and-lexicon that the designers give GENESIS at birth, in the same way the
innate manifest gives it arithmetic and logic primitives. It is declared explicitly
(``faculty_manifest``) and kept separate from acquired knowledge:

* It knows **words about its own interface** (knowing, learning, rules, sequences, stages, limits,
  curiosity, exploring, predicting, explaining, greetings, yes/no, pronouns...) and the grammatical cues
  of both languages (elisions, contractions, interrogatives, negation, verb inversion, light stemming).
* It knows **nothing about the world** and nothing about any environment: no environment variable,
  rule, number or fact appears in it (checked by ``audit``). What GENESIS says about the world still
  comes only from its own memory.
* A sentence about something outside this vocabulary is still *parsed* (language, question type,
  negation, topic words) so GENESIS can answer honestly "I understand you are asking about X, but I
  have no experience of X" instead of pretending.

Words the faculty does not know can still be taught by the user (``lexicon.py``): learned mappings are
acquired knowledge with evidence, the faculty is innate and versioned.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

VERSION = 1

# concept -> (French forms, English forms). Forms are lower-case, accent-folded.
LEXICON = {
    "KNOW": ("savoir sais sait savez savons savent su connaitre connais connait connaissez connaissons "
             "connaissance connaissances connu", "know knows knew known knowing knowledge"),
    "LEARN": ("apprendre apprends apprend appris apprenez apprentissage decouvrir decouvert decouverte "
              "decouvertes", "learn learns learned learnt learning discover discovered discovery"),
    "BELIEF": ("croyance croyances croire crois hypothese hypotheses", "belief beliefs believe hypothesis"),
    "RULE": ("regle regles loi lois motif motifs formule", "rule rules law laws pattern patterns formula"),
    "SEQUENCE": ("suite suites sequence sequences nombre nombres serie series entier entiers",
                 "sequence sequences series number numbers integer integers"),
    "CONTINUE": ("continue continuer continuons continuez encore ensuite suite", "continue more further keep"),
    "PREDICT": ("predire predis predit prediction predictions prevoir prevois deviner devine",
                "predict predicts prediction predictions forecast guess"),
    "NEXT": ("prochain prochains prochaine prochaines suivant suivants suivante suivantes apres",
             "next following after upcoming"),
    "EXPLAIN": ("expliquer explique expliques expliquez explication justifier justifie preuve preuves",
                "explain explanation justify evidence proof prove"),
    "EXPLORE": ("explorer explore exploration experimenter experimente experience experiences tester teste "
                "essayer essaie", "explore exploration experiment experiments test try"),
    "WORD": ("mot mots vocabulaire lexique", "word words vocabulary lexicon"),
    "LANGUAGE": ("langue langues langage francais francaise anglais anglaise", "language languages french english"),
    "SPEAK": ("parler parles parle parlez", "speak speaks talk"),
    "UNDERSTAND": ("comprendre comprends comprend compris comprenez", "understand understands understood"),
    "LIMIT": ("limite limites limitation limitations faiblesse faiblesses echec echecs defaut defauts",
              "limit limits limitation limitations weakness weaknesses failure failures flaw flaws"),
    "QUESTION": ("question questions curiosite curieux objectif objectifs lacune lacunes doute doutes "
                 "incertitude incertain ignorer ignores", "question questions curiosity curious goal goals "
                 "objective objectives gap gaps doubt doubts uncertainty uncertain unknown"),
    "STAGE": ("stade stades etat progres progression developpement niveau avancement statut",
              "stage stages status state progress development level"),
    "MEMORY": ("memoire souvenir souvenirs episode episodes", "memory memories remember episode episodes"),
    "HELP": ("aide aider aides commande commandes capacite capacites fonctionnalite fonctionnalites",
             "help commands command capabilities capability features usage"),
    "CAN": ("pouvoir peux peut pouvez capable", "can could able"),
    "DO": ("faire fais fait faites", "do does doing"),
    "GREET": ("bonjour salut coucou bonsoir", "hello hi hey greetings yo"),
    "BYE": ("revoir adieu ciao", "bye goodbye"),
    "THANKS": ("merci", "thanks thank thx"),
    "YES": ("oui ouais exact exactement correct voila", "yes yeah yep correct exactly right ok okay sure"),
    "NO": ("non nan faux incorrect", "no nope wrong incorrect"),
    "NAME": ("nom appelles appelle", "name called"),
    "MEANING": ("signifie signifier definition sens definis", "mean means meaning definition define"),
    "SELF": ("tu toi te ton ta tes vous votre vos genesis", "you your yours yourself genesis"),
    "USER": ("je moi me mon ma mes", "i me my myself"),
}

# Multi-word expressions (on normalized tokens) -> concept. Matched before single words.
IDIOMS = {
    "ca va": "HOWAREYOU", "comment vas tu": "HOWAREYOU", "comment allez vous": "HOWAREYOU",
    "quoi de neuf": "HOWAREYOU", "how are you": "HOWAREYOU", "how is it going": "HOWAREYOU",
    "what is up": "HOWAREYOU", "how do you do": "HOWAREYOU",
    "ou en es tu": "STAGE", "ou en etes vous": "STAGE", "where are you at": "STAGE",
    "veut dire": "MEANING", "au revoir": "BYE", "a bientot": "BYE", "see you": "BYE",
    "de accord": "YES",
    "mode de emploi": "HELP",
}

FUNCTION_FR = set("le la les un une des de du au aux et ou mais donc ni car que qui quoi ce cet cette ces se ne "
                  "pas plus jamais rien en y a dans sur sous pour par avec sans est sont suis es etre ai as avons "
                  "avez ont il elle ils elles on nous son sa ses notre leur leurs ca cela ceci tout tous toute "
                  "toutes tres bien aussi si alors comme quel quelle quels quelles comment pourquoi quand combien "
                  "lui y aucun aucune vais vas va allez sur peu beaucoup c est-ce".split())
FUNCTION_EN = set("the a an of to in on at for with by from and or but if so is are am was were be been being "
                  "does did have has had he she it its we they them their this that these those there here not "
                  "very also just about as than then what which who whom whose why how when where will would "
                  "shall should may might must any some all much many tell please like".split())
FUNCTION = FUNCTION_FR | FUNCTION_EN

EN_CONTRACTIONS = ((r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"n't\b", " not"), (r"'re\b", " are"),
                   (r"'m\b", " am"), (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
                   (r"\b(it|what|that|there|here|who|where|how|he|she)'s\b", r"\1 is"), (r"'s\b", ""))
FR_ELISIONS = {"l": "le", "d": "de", "j": "je", "qu": "que", "n": "ne", "s": "se", "c": "ce", "t": "te",
               "m": "me", "jusqu": "jusque", "lorsqu": "lorsque", "puisqu": "puisque"}
ELISION_RE = re.compile(r"\b(" + "|".join(sorted(FR_ELISIONS, key=len, reverse=True)) + r")'")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
INT_RE = re.compile(r"(?<![\w.\-])-?\d+(?![\w.])")
NEGATORS = {"pas", "jamais", "rien", "aucun", "aucune", "guere", "not", "never", "nothing", "none"}
EN_AUX = {"do", "does", "did", "can", "could", "are", "is", "am", "will", "would", "have", "has", "should"}
SUFFIXES = sorted("issons issez issent erons erez eront aient ions iez ons ez ent ais ait ees ee es er ir e s x "
                  "ing ied ies ed ly".split(), key=len, reverse=True)
NO_STEM = {"francais", "francaise", "anglais", "anglaise"}  # "franc-" would also catch "France"
GLOSSARY = ("RULE", "BELIEF", "MEMORY", "STAGE", "SEQUENCE", "PREDICT", "QUESTION", "WORD", "EXPLORE")


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def stem(word: str) -> str:
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word


def _close(a: str, b: str) -> bool:
    """Damerau–Levenshtein distance <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]]
                                  and a[diff[1]] == b[diff[0]])
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _build():
    forms, langs = {}, {}
    for concept, (fr, en) in LEXICON.items():
        for lang, words in (("fr", fr), ("en", en)):
            for w in words.split():
                forms.setdefault(w, [])
                if concept not in forms[w]:
                    forms[w].append(concept)
                langs.setdefault(w, set()).add(lang)
    stems = {}
    for w in sorted(forms):
        if w not in NO_STEM:
            stems.setdefault(stem(w), forms[w])
    return forms, langs, stems


FORMS, FORM_LANGS, STEMS = _build()


@dataclass
class Parse:
    text: str
    lang: str
    tokens: list            # normalized (folded) tokens
    surface: list           # same tokens as typed (accents kept)
    concepts: dict          # concept -> count
    qtype: Optional[str]    # what / why / how / who / where / when / howmuch / yesno / None
    negated: bool
    numbers: list
    topic: list = field(default_factory=list)       # content words the faculty does not know
    corrected: dict = field(default_factory=dict)   # typo -> known form

    def has(self, *concepts) -> bool:
        return any(c in self.concepts for c in concepts)

    @property
    def understood(self) -> float:
        content = [t for t in self.tokens if t not in FUNCTION]
        return 1.0 if not content else 1 - len(self.topic) / len(content)


def _normalize(text: str) -> str:
    t = text.lower().replace("’", "'").replace("`", "'")
    for pat, rep in EN_CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = ELISION_RE.sub(lambda m: FR_ELISIONS[m.group(1)] + " ", t)
    return t.replace("-t-", " ").replace("-", " ")


def lookup(word: str):
    """Concepts of a folded word: exact form, then stem, then one typo (any edit for words of 6+ letters, only a
    swap of two adjacent letters for shorter ones). Returns (concepts, form)."""
    if word in FORMS:
        return FORMS[word], word
    s = stem(word)
    if s in STEMS:
        return STEMS[s], word
    if len(word) >= 4:
        for w in sorted(FORMS):
            if w[0] == word[0] and len(w) >= 4 and (len(word) >= 6 and len(w) >= 6 or sorted(w) == sorted(word)) \
                    and _close(word, w):
                return FORMS[w], w
    return None, None


def detect_lang(tokens: list, raw: str, fallback: Optional[str] = None) -> str:
    fr = en = 0
    for t in tokens:
        in_fr = t in FUNCTION_FR or "fr" in FORM_LANGS.get(t, ())
        in_en = t in FUNCTION_EN or "en" in FORM_LANGS.get(t, ())
        fr += in_fr and not in_en
        en += in_en and not in_fr
    if re.search(r"[éèêàçùôîâœ]", raw.lower()):
        fr += 2
    if re.search(r"\b(qu|l|d|j|n|c)'", raw.lower()):
        fr += 1
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    return fallback if fallback in ("fr", "en") else "en"


def _interrogative_qui(tokens: list) -> bool:
    """French "qui" is interrogative at the start or next to être ("qui es-tu", "c'est qui"), else relative."""
    for i, t in enumerate(tokens):
        if t == "qui" and (i == 0 or tokens[i - 1] in ("es", "est", "etes", "suis")
                           or tokens[i + 1:i + 2] in (["es"], ["est"], ["etes"], ["sont"])):
            return True
    return False


def _qtype(tokens: list, raw: str) -> Optional[str]:
    head = tokens[:4]
    joined = " ".join(tokens)
    if "pourquoi" in tokens or "why" in head:
        return "why"
    if "combien" in tokens or re.search(r"\bhow (many|much)\b", joined):
        return "howmuch"
    if "who" in head or _interrogative_qui(tokens):
        return "who"
    if re.search(r"(^|\s)où(\s|$|\?)", raw.lower()) or "where" in head:
        return "where"
    if "quand" in head or "when" in head:
        return "when"
    if "comment" in head or "how" in head:
        return "how"
    if (head[:1] == ["que"] or "quoi" in tokens or "que est ce" in joined
            or any(t in ("quel", "quelle", "quels", "quelles", "what", "which") for t in head)):
        return "what"
    if raw.rstrip().endswith("?") or joined.startswith("est ce que") \
            or (head[:1] and head[0] in EN_AUX and head[1:2] and head[1] in ("you", "i", "we", "they", "it")) \
            or re.search(r"\w-(tu|vous|il|elle|on)\b", raw.lower()):
        return "yesno"
    return None


def parse(text: str, fallback_lang: Optional[str] = None) -> Parse:
    norm = _normalize(text)
    surface = [w for w in WORD_RE.findall(norm) if w != "t"]
    tokens = [fold(w) for w in surface]
    numbers = [int(m) for m in INT_RE.findall(text)]
    concepts, topic, corrected = {}, [], {}
    joined = " " + " ".join(tokens) + " "
    used = set()
    for idiom, concept in sorted(IDIOMS.items(), key=lambda kv: -len(kv[0])):
        if f" {idiom} " in joined:
            concepts[concept] = concepts.get(concept, 0) + 1
            n, start = len(idiom.split()), joined.index(f" {idiom} ")
            first = joined[:start].count(" ")
            used.update(range(first, first + n))
    for i, (t, s) in enumerate(zip(tokens, surface)):
        if i in used:
            continue
        cs, form = lookup(t) if t not in FUNCTION or t in FORMS else (None, None)
        if cs:
            for c in cs:
                concepts[c] = concepts.get(c, 0) + 1
            if form != t:
                corrected[t] = form
        elif t not in FUNCTION and len(t) > 1:
            topic.append(s)
    negated = bool(set(tokens) & NEGATORS) or ("ne" in tokens and "plus" in tokens)
    return Parse(text=text, lang=detect_lang(tokens, text, fallback_lang), tokens=tokens, surface=surface,
                 concepts=concepts, qtype=_qtype(tokens, text), negated=negated, numbers=numbers, topic=topic,
                 corrected=corrected)


def words_of(concept: str) -> set:
    fr, en = LEXICON[concept]
    return set(fr.split()) | set(en.split())


def faculty_manifest() -> dict:
    body = {"version": VERSION, "languages": ["fr", "en"],
            "kind": "hand-written lexicon + grammatical cues (designer-provided, no learned weights)",
            "concepts": sorted(set(LEXICON) | set(IDIOMS.values())),
            "n_forms": len(FORMS), "n_idioms": len(IDIOMS),
            "scope": "words about GENESIS's own interface and cognition; no world or environment knowledge",
            "pretrained_models": []}
    blob = json.dumps({"body": body, "lexicon": LEXICON, "idioms": IDIOMS, "function": sorted(FUNCTION)},
                      sort_keys=True)
    return {**body, "digest": hashlib.sha256(blob.encode()).hexdigest()}


# Words that would describe the hidden rules of a world (operations, rule families, number properties). The
# faculty must not contain them: naming "plus 4" or "fibonacci" for GENESIS would hand it world knowledge.
WORLD_TERMS = ("plus moins fois double triple moitie carre cube modulo reste pair impair premier somme produit "
               "difference ajouter multiplier diviser incrementer alterner periode periodique arithmetique "
               "geometrique quadratique fibonacci recurrence add plus minus times twice half square cube modulo "
               "remainder even odd prime sum product difference multiply divide increment alternate period "
               "periodic arithmetic geometric quadratic recurrence counter").split()


def audit(forbidden_terms=()) -> dict:
    """Checks that the faculty holds no world knowledge: no digits, no rule vocabulary, no environment terms."""
    vocab = set(FORMS) | {w for i in IDIOMS for w in i.split()}
    forbidden = {fold(t) for t in (*WORLD_TERMS, *forbidden_terms)} & vocab
    checks = {"no_digits": not any(any(c.isdigit() for c in w) for w in vocab),
              "no_environment_terms": not forbidden,
              "no_pretrained_models": faculty_manifest()["pretrained_models"] == []}
    return {"passed": all(checks.values()), "checks": checks, "forbidden_found": sorted(forbidden)}
