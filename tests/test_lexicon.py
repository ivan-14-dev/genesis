import pytest

from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.lexicon import Lexicon, tokens


@pytest.fixture
def chat(tmp_path):
    agent = Genesis(str(tmp_path / "lex.db"))
    eng = ChatEngine(agent)
    cid = eng.new_conversation("fr")["id"]

    def say(text):
        return eng.send(cid, text, "fr")["reply"]

    yield eng, say
    agent.close()


def test_starts_with_no_vocabulary_and_does_not_guess(chat):
    eng, say = chat
    assert eng.lexicon.size() == 0
    r = say("zorglub blip")
    assert r["meta"]["intent"] == "not-understood"
    assert "Apprends" in r["text"] and r["meta"]["choices"]


def test_taught_words_are_proposed_then_validated_by_forward_predictions(chat):
    eng, say = chat
    say("zorglub blip")
    r = say("stade")                                     # teaching
    assert r["meta"]["intent"] == "stage" and r["meta"]["learned_word"]
    for _ in range(2):                                   # unvalidated: asks, never acts silently
        r = say("zorglub blip")
        assert r["meta"]["intent"] == "clarify" and r["meta"]["proposed"] == "stage"
        assert say("oui")["meta"]["intent"] == "stage"
    r = say("zorglub blip")                              # validated: acts, and says so
    assert r["meta"]["intent"] == "stage" and r["meta"]["interpreted"]
    assert "Réponds `non`" in r["text"]
    item = [k for k in eng.ltm.knowledge(kind="lexical") if k.key == "lexicon/zorglub"][0]
    assert item.status == "validated" and item.provenance == "user-taught" and item.forward_ok == 2
    ev = eng.ltm.explain(item.id)["evidence"]
    assert ev["induced_from"]["total"] == 1 and ev["supports"]["total"] == 2


def test_generalises_to_unseen_sentences_sharing_taught_words(chat):
    eng, say = chat
    say("balance tes trouvailles")
    say("que sais-tu ?")
    say("zorglub blip")
    say("stade")
    r = say("tes trouvailles stp")                       # never seen as a whole
    assert r["meta"]["intent"] == "clarify" and r["meta"]["proposed"] == "knowledge"
    say("non")
    say("stade")                                         # correction: reject + teach
    st = eng.lexicon.stats["trouvailles"]
    assert st["knowledge"]["c"] == 1 and st["knowledge"]["bad"] == 1 and st["stage"]["s"] == 1


def test_function_words_are_diluted_across_intents():
    class Mem:  # minimal in-memory stand-in for the knowledge store
        def __init__(self):
            self.n = 0
        def knowledge(self, **kw):
            return []
        def record_episode(self, *a, **k):
            self.n += 1
            return self.n
        def upsert_knowledge(self, *a, **k):
            self.n += 1
            return self.n
        def add_evidence(self, *a):
            pass
        def commit(self):
            pass
    lx = Lexicon(Mem(), ("stage", "knowledge", "help"), lambda: 1)
    lx.learn("dis moi tes souvenirs", "knowledge", "taught")
    lx.learn("dis moi ton stade", "stage", "taught")
    assert lx.interpret("souvenirs").intent == "knowledge"
    assert lx.interpret("ton stade").intent == "stage"
    assert lx.p("dis", "knowledge") < lx.p("souvenirs", "knowledge")
    assert lx.interpret("bonjour inconnu") is None


def test_wrong_interpretation_is_refuted_and_stops_acting(chat):
    eng, say = chat
    say("ça roule ?")
    say("stade")
    for _ in range(2):
        say("ça roule ?")
        say("oui")
    assert say("ça roule ?")["meta"].get("interpreted")
    for _ in range(3):                                   # user keeps saying the act was wrong
        r = say("ça roule ?")
        if r["meta"].get("interpreted") or r["meta"]["intent"] == "clarify":
            say("non")
            say("aide")
    r = say("ça roule ?")
    assert not (r["meta"].get("interpreted") and r["meta"]["intent"] == "stage")
    assert any(k.status == "refuted" for k in eng.ltm.knowledge(kind="lexical") if k.expr["intent"] == "stage")


def test_yes_no_are_never_learned_as_words(chat):
    eng, say = chat
    say("bidule truc")
    assert say("non")["meta"]["intent"] == "teach-request"
    say("aide")
    assert "non" not in eng.lexicon.stats and "oui" not in eng.lexicon.stats
    assert say("oui")["meta"]["intent"] == "yes-no"


def test_vocabulary_persists_across_sessions(tmp_path):
    db = str(tmp_path / "p.db")
    a = Genesis(db)
    e = ChatEngine(a)
    cid = e.new_conversation("fr")["id"]
    e.send(cid, "quid du palier", "fr")
    e.send(cid, "stade", "fr")
    a.close()
    a = Genesis(db)
    e = ChatEngine(a)
    assert e.lexicon.size() == 3
    cid = e.new_conversation("fr")["id"]
    assert e.send(cid, "quid du palier", "fr")["reply"]["meta"]["proposed"] == "stage"
    assert e.send(cid, "vocabulaire", "fr")["reply"]["meta"]["n"] == 3
    a.close()


def test_tokens_fold_accents_and_bound_input():
    assert tokens("Où ÊTES-vous où") == ["ou", "etes", "vous"]
    assert len(tokens(" ".join(f"w{'a' * i}" for i in range(50)))) == 12


def test_word_learning_experiment_generalises_without_wrong_actions():
    from genesis.experiments import run_word_learning
    r = run_word_learning(0)
    assert r["heldout_accuracy"] >= 0.8 > 3 * r["chance_accuracy"]
    assert r["heldout_act_precision"] == 1.0 and r["acted_wrong_during_training"] == 0
    assert r["control_would_act"] == 0
    first, last = r["per_pass"][0], r["per_pass"][-1]
    assert first.get("acted-correct", 0) == 0 and last.get("acted-correct", 0) >= 0.8 * r["train_sentences"]
