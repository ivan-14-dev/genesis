import json
import random

import pytest

from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.environments import ObjectWorld, sandboxed
from genesis.grounded_language import GroundedLexicon, parse_terms
from genesis.grounding import GroundedExplorer


@pytest.fixture
def workshop(tmp_path):
    a = Genesis(str(tmp_path / "words.db"))
    w = ObjectWorld(0)
    g = GroundedExplorer(a, sandboxed(w, 1000))
    g.run(80)
    yield a, w, g, GroundedLexicon(g)
    a.close()


def teach(lex, world, word, action, n=24):
    objects = list(lex.g.objects)
    random.Random(7).shuffle(objects)
    results = [lex.teach(word, obj, world._true_next(obj["features"], action) == 1) for obj in objects[:n]]
    return results


def test_meanings_are_learned_from_effects_not_preloaded(workshop):
    a, w, g, lex = workshop
    assert a.ltm.knowledge(kind="grounded-word") == []
    assert lex.predict("dax", g.objects[0])["value"] is None
    results = teach(lex, w, "dax", "a1")
    assert not results[0]["validated"]
    assert results[-1]["validated"]
    assert results[-1]["candidates"] == [("effect", "a1", 1)]
    for obj in w._heldout(777):
        assert lex.predict("dax", obj)["value"] == (w._true_next(obj["features"], "a1") == 1)
    ex = a.ltm.explain(results[-1]["knowledge_id"])
    assert ex["evidence"]["supports"]["total"] >= 3
    sources = {e["provenance"] for e in ex["evidence"]["induced_from"]["examples"]}
    assert "user-grounded-teaching" in sources or "experiment" in sources
    # Query the complete evidence set: both annotations and real experimental outcomes are present.
    provenance = {r[0] for r in a.ltm.db.execute(
        "SELECT DISTINCT e.provenance FROM evidence v JOIN episodes e ON e.id=v.episode_id "
        "WHERE v.knowledge_id=?", (results[-1]["knowledge_id"],))}
    assert {"experiment", "user-grounded-teaching"} <= provenance
    for ep in ex["evidence"]["supports"]["examples"]:
        p = json.loads(ep["predictions"])["dax"]
        assert p["novel_input"] and p["predicted"] == p["actual"]


def test_new_compositions_and_unknown_negation(workshop):
    _, w, g, lex = workshop
    teach(lex, w, "dax", "a1")
    results = teach(lex, w, "mip", "a0")
    assert len(results[-1]["candidates"]) == 2  # Sensory and action meanings remain observationally equivalent.
    heldout = w._heldout(888)
    for phrase in ("dax and mip", "dax et pas mip", "not dax and mip"):
        terms = parse_terms(phrase)
        expected = [o["id"] for o in heldout if all(
            (w._true_next(o["features"], "a1" if word == "dax" else "a0") == 1) != negative
            for word, negative in terms)]
        assert [o["id"] for o in lex.select(heldout, terms)] == expected
    assert lex.match(g.objects[0], [("inconnu", True)]) is None
    assert lex.select(heldout, [("inconnu", True)]) == []
    assert parse_terms("dax or mip") is None
    assert parse_terms("dax and mip and dax and mip and dax") is None


def test_repetition_does_not_validate(workshop):
    _, _, g, lex = workshop
    for _ in range(10):
        r = lex.teach("dax", g.objects[0], True)
    assert not r["validated"] and r["forward_ok"] == 0


def test_unknown_effects_abstain_and_queries_do_not_train(workshop):
    a, w, g, lex = workshop
    teach(lex, w, "dax", "a1")
    obj = w._heldout(777)[0]
    obj["features"][g.models["a1"].description()["features"][0]] = 2
    before = json.dumps([g.state, lex.state], sort_keys=True), a.ltm.count("episodes")
    assert lex.predict("dax", obj)["value"] is None
    assert lex.select([obj], [("dax", True)]) == []
    assert before == (json.dumps([g.state, lex.state], sort_keys=True), a.ltm.count("episodes"))
    for k in a.ltm.knowledge(key=f"{g.key}/a1", status="validated"):
        a.ltm.set_status(k.id, "candidate", "test: model validation unavailable")
    assert lex.predict("dax", w._heldout(777)[0])["value"] is None
    assert lex.predict("dax", g.objects[0])["value"] is not None  # Actual observations remain usable.


def test_label_limits_and_provenance_are_checked_before_mutation(workshop, monkeypatch):
    from genesis import grounded_language as language
    a, _, g, lex = workshop
    monkeypatch.setattr(language, "MAX_WORDS", 1)
    monkeypatch.setattr(language, "MAX_LABELS", 2)
    lex.teach("dax", g.objects[0], True)
    before = a.ltm.count("episodes")
    for word, label, source in (("mip", True, "user-grounded-teaching"),
                                ("dax", True, "experiment"), ("dax", 1, "user-grounded-teaching")):
        with pytest.raises(ValueError):
            lex.teach(word, g.objects[0], label, source=source)
    assert a.ltm.count("episodes") == before
    lex.teach("dax", g.objects[0], True)
    with pytest.raises(ValueError, match="memory budget"):
        lex.teach("dax", g.objects[0], True)
    assert len(lex.state["examples"]) == 2


def test_conflicting_annotation_and_explicit_correction(workshop):
    a, w, g, lex = workshop
    teach(lex, w, "dax", "a1")
    obj = g.objects[0]
    correct = w._true_next(obj["features"], "a1") == 1
    lex.teach("dax", obj, correct)
    before = len(lex.state["examples"])
    conflict = lex.teach("dax", obj, not correct)
    assert conflict["status"] == "inconsistent"
    assert lex.predict("dax", obj)["value"] is None
    repaired = lex.teach("dax", obj, correct, correction=True)
    assert repaired["value"] == correct
    assert len(lex.state["examples"]) == before + 2
    assert a.ltm.episode(conflict["episode_id"])
    assert not repaired["validated"]  # Corrections do not manufacture new forward checks.


def test_model_revision_requires_new_grounding(workshop):
    a, w, g, lex = workshop
    teach(lex, w, "dax", "a1")
    w._change("a1")
    g.experience(g.objects[0], "a1")
    assert lex.predict("dax", g.objects[0])["status"] == "unknown"
    assert not a.ltm.knowledge(kind="grounded-word", status="validated")
    assert lex.state["examples"]  # Original language experiences survive.


def test_dialogue_and_reload(tmp_path):
    db = str(tmp_path / "dialogue.db")
    a = Genesis(db)
    factory = lambda seed: sandboxed(ObjectWorld(seed), 1000)
    e = ChatEngine(a, object_factory=factory)
    cid = e.new_conversation("fr")["id"]
    say = lambda text: e.send(cid, text)["reply"]
    assert say("atelier objets")["meta"]["intent"] == "object-workshop"
    assert say("explore objets 80")["meta"]["steps"] == 80
    g, _ = e.objects.workshop(0)
    world = ObjectWorld(0)  # Test oracle only.
    objects = list(g.objects)
    random.Random(7).shuffle(objects)
    for obj in objects[:24]:
        say(f"observe {obj['id']}")
        is_dax = world._true_next(obj["features"], "a1") == 1
        verb = "est" if is_dax else "n'est pas"
        assert say(f"cet objet {verb} dax")["meta"]["intent"] == "object-word-teach"
    before = say("montre un objet dax")["meta"]["objects"]
    assert before
    assert say("pourquoi le mot dax ?")["meta"]["knowledge_ids"]
    a.close()
    a = Genesis(db)
    try:
        e = ChatEngine(a, object_factory=factory)
        vocabulary = say("vocabulaire")
        assert vocabulary["meta"]["grounded"] == 1
        assert "dax" in vocabulary["text"]
        assert say("show an object dax")["meta"]["objects"] == before
        assert say("Is this object dax?")["meta"]["value"] is True
        assert say("this object is not dax?")["meta"]["intent"] != "object-word-teach"
        assert say("observe nonexistent")["meta"]["intent"] == "refused"
        assert say("explore objets 9999")["meta"]["intent"] == "refused"
        other = e.new_conversation("en")["id"]
        assert e.send(other, "Is this object dax?")["reply"]["meta"]["intent"] == "object-workshop"
        e.send(other, "object workshop")
        assert e.send(other, "this object is dax")["reply"]["meta"]["intent"] == "refused"
        assert say("Is this object dax?")["meta"]["value"] is True
    finally:
        a.close()


def test_workshop_not_configured_is_explicit(tmp_path):
    a = Genesis(str(tmp_path / "none.db"))
    try:
        e = ChatEngine(a)
        cid = e.new_conversation()["id"]
        assert e.send(cid, "object workshop")["reply"]["meta"]["intent"] == "refused"
    finally:
        a.close()


def test_grounded_word_benchmark():
    from genesis.experiments import run_grounded_word_learning
    r = run_grounded_word_learning(3)
    assert r["initial_word_coverage"] == 0
    assert r["word_accuracy"] >= 0.95 and r["word_coverage"] == 1
    assert r["exact_compositional_references"] == r["composition_queries"] == 4
    assert r["validated_words"] == 2
    assert r["object_id_lookup_coverage"] == r["unknown_negative_references"] == 0
    assert r["reload_retention"] == r["post_correction_accuracy"] == 1
    assert r["conflict_abstention"] and r["correction_revalidated"] and r["old_annotation_preserved"]
    assert r["world_experiments"] == r["world_experiments_after_language"] == 80
