import json
import random

import pytest

from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.constructions import ConstructionLearner, canonical
from genesis.environments import ObjectWorld, sandboxed
from genesis.grounded_language import GroundedLexicon
from genesis.grounding import GroundedExplorer


@pytest.fixture
def workshop(tmp_path):
    a = Genesis(str(tmp_path / "syntax.db"))
    g = GroundedExplorer(a, sandboxed(ObjectWorld(0), 1000))
    g.run(80)
    lex = GroundedLexicon(g)
    objects = list(g.objects)
    random.Random(7).shuffle(objects)
    for i, word in enumerate(("dax", "mip", "zup", "tov", "bek", "nif")):
        for obj in objects[:24]:
            lex.teach(word, obj, obj["features"][f"s{i % 2}"] == i % 2)
    yield a, g, lex, ConstructionLearner(lex)
    a.close()


def train(grammar):
    results = []
    for a, b in (("dax", "mip"), ("mip", "dax"), ("zup", "tov"), ("tov", "zup")):
        results.append(grammar.teach(f"prends {a} sans {b}", [(a, False), (b, True)]))
    return results


def test_empty_birth_transfer_evidence_and_no_query_learning(workshop):
    a, g, lex, grammar = workshop
    assert grammar.parse("prends dax sans mip") is None
    results = train(grammar)
    assert [r["validated"] for r in results] == [False, False, False, True]
    before = json.dumps(grammar.state, sort_keys=True), a.ltm.count("episodes")
    parsed = grammar.parse("prends bek sans nif")
    assert parsed["validated"] and parsed["terms"] == canonical([("bek", False), ("nif", True)])
    assert lex.select(g.objects, parsed["terms"]) == lex.select(g.objects, [("bek", False), ("nif", True)])
    assert before == (json.dumps(grammar.state, sort_keys=True), a.ltm.count("episodes"))
    assert grammar.parse("prends inconnu sans nif") is None
    assert grammar.parse("prends bek avec nif") is None
    evidence = a.ltm.explain(results[-1]["knowledge_id"])["evidence"]["supports"]
    assert evidence["total"] == 3
    for episode in evidence["examples"]:
        pred, outcome = json.loads(episode["predictions"]), json.loads(episode["outcome"])
        assert pred["novel_binding"] and pred["terms"] == outcome["terms"]


def test_repetition_and_unvaried_slots_do_not_validate(workshop):
    _, _, _, grammar = workshop
    for _ in range(6):
        r = grammar.teach("prends dax sans mip", [("dax", False), ("mip", True)])
    assert r["forward_ok"] == 0 and not r["validated"]
    for word in ("zup", "tov", "bek", "nif"):
        r = grammar.teach(f"prends {word} sans mip", [(word, False), ("mip", True)])
    assert r["forward_ok"] == 4 and not r["validated"]


def test_conflict_correction_requires_new_checks(workshop):
    a, _, _, grammar = workshop
    train(grammar)
    wrong = grammar.teach("prends dax sans mip", [("dax", True), ("mip", False)])
    assert grammar.parse("prends bek sans nif")["terms"] is None
    repaired = grammar.teach("prends dax sans mip", [("dax", False), ("mip", True)], correction=True)
    assert not repaired["validated"] and repaired["forward_ok"] == 0
    assert a.ltm.episode(wrong["episode_id"])
    for left, right in (("bek", "nif"), ("nif", "bek"), ("dax", "nif")):
        r = grammar.teach(f"prends {left} sans {right}", [(left, False), (right, True)])
    assert r["validated"]


def test_surface_negation_is_learned_not_predefined(workshop):
    _, _, _, grammar = workshop
    for left, right in (("dax", "mip"), ("mip", "dax"), ("zup", "tov"), ("tov", "zup")):
        grammar.teach(f"{right} sans {left} choisis", [(left, False), (right, True)])
    assert grammar.parse("nif sans bek choisis")["terms"] == canonical([("bek", False), ("nif", True)])


def test_overlapping_constructions_abstain(workshop):
    _, _, _, grammar = workshop
    train(grammar)
    grammar.teach("prends dax sans mip", [("mip", False)])
    result = grammar.parse("prends dax sans mip")
    assert result["terms"] is None and not result["validated"]
    assert grammar.parse("prends bek sans nif")["validated"]


def test_bounds_and_invalid_demonstrations_do_not_mutate(workshop, monkeypatch):
    from genesis import constructions
    a, _, _, grammar = workshop
    before = a.ltm.count("episodes")
    for text, terms in (("prends unknown", [("unknown", False)]),
                        ("prends dax dax", [("dax", False)]), ("dax mip", [("dax", False), ("mip", True)]),
                        ("prends dax", [("dax", 1)])):
        with pytest.raises(ValueError):
            grammar.teach(text, terms)
    with pytest.raises(ValueError, match="provenance"):
        grammar.teach("prends dax", [("dax", False)], source="experiment")
    assert a.ltm.count("episodes") == before
    monkeypatch.setattr(constructions, "MAX_EXAMPLES", 1)
    grammar.teach("prends dax", [("dax", False)])
    with pytest.raises(ValueError, match="budget"):
        grammar.teach("prends mip", [("mip", False)])
    assert len(grammar.state["examples"]) == 1


def test_chat_candidates_transfer_and_reload(workshop):
    a, g, lex, _ = workshop
    factory = lambda seed: sandboxed(ObjectWorld(seed), 1000)
    engine = ChatEngine(a, object_factory=factory)
    cid = engine.new_conversation("fr")["id"]
    say = lambda text: engine.send(cid, text)["reply"]
    say("atelier objets")
    for i, (left, right) in enumerate((("dax", "mip"), ("mip", "dax"), ("zup", "tov"), ("tov", "zup"))):
        r = say(f"apprends phrase : prends {left} sans {right} => {left} et pas {right}")
        assert r["meta"]["intent"] == "construction-teach"
        if i == 0:
            assert say("prends bek sans nif")["meta"]["intent"] == "construction-uncertain"
    r = say("prends bek sans nif")
    assert r["meta"]["intent"] == "object-select" and r["meta"]["construction"]["validated"]
    assert r["meta"]["objects"] == [o["id"] for o in lex.select(g.objects, [("bek", False), ("nif", True)])]
    assert say("vocabulaire")["meta"]["constructions"] == 1
    restored = ConstructionLearner(GroundedLexicon(g))
    assert restored.parse("prends bek sans nif")["validated"]
    # A second connection verifies durable state rather than a shared in-memory object.
    b = Genesis(a.ltm.path)
    try:
        engine = ChatEngine(b, object_factory=factory)
        assert say("prends bek sans nif")["meta"]["objects"] == r["meta"]["objects"]
        assert say("learn phrase: fetch dax without mip => dax and not mip")["meta"]["intent"] == "construction-teach"
    finally:
        b.close()


def test_construction_benchmark():
    from genesis.experiments import run_construction_learning
    r = run_construction_learning(3)
    assert r["initial_interpretations"] == 0
    assert r["parse_accuracy"] == r["validated_coverage"] == r["reload_retention"] == 1
    assert r["exact_object_selections"] == r["heldout_sentences"] == 36
    assert r["sentence_lookup_coverage"] == 0 and r["unknown_frame_abstention"]
    assert r["episodes_before_reload"] == r["world_experiments"] + r["word_annotations"] + r["construction_annotations"]
