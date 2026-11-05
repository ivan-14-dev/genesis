import pytest

from genesis import language
from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.language import parse


def test_normalisation_elisions_contractions_and_inversion():
    p = parse("Qu'est-ce que tu sais ?")
    assert p.lang == "fr" and p.qtype == "what" and p.has("KNOW", "SELF")
    p = parse("What don't you know?")
    assert p.lang == "en" and p.negated and p.has("KNOW")
    assert parse("Sais-tu parler anglais ?").qtype == "yesno"
    assert parse("Don't explore").qtype is None          # imperative, not a question


def test_stemming_typos_and_relative_pronouns():
    assert parse("tu as découvert des règles").has("LEARN", "RULE")
    p = parse("connaisance")
    assert p.has("KNOW") and p.corrected == {"connaisance": "connaissance"}
    assert parse("What do you knwo?").has("KNOW")         # adjacent swap in a short word
    assert not parse("ça roule ?").has("RULE")             # no cross-language guess on short words
    assert parse("Who are you?").qtype == "who"
    assert parse("ce qui coince chez toi").qtype is None  # relative "qui" is not a question


def test_language_detection_and_topics():
    assert parse("bonjour, que sais-tu ?").lang == "fr"
    assert parse("hello, what do you know?").lang == "en"
    p = parse("Quelle est la capitale de l'Italie ?")
    assert p.qtype == "what" and p.topic == ["capitale", "italie"] and not p.has("LANGUAGE")


def test_faculty_is_declared_and_holds_no_world_knowledge(tmp_path):
    import inspect
    import re
    from genesis.environments import causal_world, sequence_world
    terms = set(re.findall(r'"family": f?"([a-z]+)', inspect.getsource(sequence_world)))
    terms |= set(re.findall(r'"([a-z]+)": lambda', inspect.getsource(causal_world)))   # rule kinds
    terms |= {"fibonacci"}
    assert {"arithmetic", "geometric", "add", "copy"} <= terms
    rep = language.audit(terms)
    assert rep["passed"], rep
    m = language.faculty_manifest()
    assert m["languages"] == ["fr", "en"] and m["pretrained_models"] == []
    agent = Genesis(str(tmp_path / "a.db"))
    assert agent.ltm.get_meta("innate_language")["digest"] == m["digest"]
    assert agent.ltm.get_meta("stage0_audit")["checks"]["language_faculty_has_no_world_knowledge"]
    assert agent.ltm.get_meta("innate_manifest")["domain_knowledge"] == []
    agent.close()


def test_upgrade_of_faculty_on_old_memory_is_logged_not_silent(tmp_path):
    db = str(tmp_path / "old.db")
    a = Genesis(db)
    a.ltm.set_meta("innate_language", None)              # memory born before the faculty existed
    a.ltm.commit()
    a.close()
    a = Genesis(db)
    hist = a.ltm.get_meta("innate_language_history")
    assert hist and hist[-1]["from"] is None and hist[-1]["session"] == a.session
    assert a.ltm.get_meta("innate_language")["installed_session"] == a.session
    a.close()


@pytest.fixture
def say(tmp_path):
    agent = Genesis(str(tmp_path / "c.db"))
    eng = ChatEngine(agent)
    cid = eng.new_conversation()["id"]
    yield lambda t: eng.send(cid, t)["reply"]
    agent.close()


def test_chat_understands_paraphrases_in_both_languages(say):
    assert say("Salut !")["meta"]["intent"] == "intro"
    assert say("Qu'as-tu appris jusqu'ici ?")["meta"]["intent"] == "knowledge"
    assert say("How far along is your development?")["meta"]["intent"] == "stage"
    assert say("Parles-tu anglais ?")["meta"]["intent"] == "languages"
    r = say("Qu'est-ce qu'une croyance ?")
    assert r["meta"] == {**r["meta"], "intent": "define", "term": "BELIEF"}
    assert say("n'explore pas")["meta"]["intent"] == "declined"


def test_out_of_scope_questions_are_answered_honestly(say):
    r = say("Quelle est la capitale de l'Italie ?")
    assert r["meta"]["intent"] == "unknown-topic" and "je n'en sais rien" in r["text"]
    r = say("merci")                                        # moving on is not teaching
    assert r["meta"]["intent"] == "thanks" and not r["meta"].get("learned_word")


def test_language_benchmarks():
    from genesis.experiments import LANGUAGE_HELDOUT, run_language_understanding
    for bench in (None, LANGUAGE_HELDOUT):
        r = run_language_understanding(bench)
        assert r["accuracy"] >= 0.9 and r["lang_detection_accuracy"] >= 0.95
        assert r["out_of_scope_answered_as_if_known"] == 0
