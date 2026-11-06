import json

import pytest

from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.semantic import Fact, parse_fact, parse_query


@pytest.fixture
def chat(tmp_path):
    a = Genesis(str(tmp_path / "semantic.db"))
    e = ChatEngine(a)
    cid = e.new_conversation("fr")["id"]
    yield a, e, cid
    a.close()


def say(chat, text):
    _, e, cid = chat
    return e.send(cid, text)["reply"]


def test_transfer_from_taught_premises_not_preloaded(chat):
    a, e, cid = chat
    assert say(chat, "Nala est-elle un animal ?")["meta"]["status"] == "unknown"
    assert a.ltm.knowledge(kind="semantic") == []
    say(chat, "Nala est un chat")
    say(chat, "Les chats sont des animaux")
    r = say(chat, "Nala est-elle un animal ?")
    assert r["meta"]["status"] == "supported" and r["meta"]["inferred"]
    ids = r["meta"]["knowledge_ids"]
    assert len(ids) == 2
    for kid in ids:
        ex = a.ltm.explain(kid)
        assert ex["item"]["provenance"] == "user-taught"
        assert ex["item"]["status"] == "taught"
        assert ex["evidence"]["supports"]["total"] == 1
    assert a.ltm.episode(r["meta"]["episode_id"])["provenance"] == "inference"
    # Transfer to an entity not present when the inclusion was taught.
    say(chat, "Milo est un chat")
    assert say(chat, "Milo est-il un animal ?")["meta"]["status"] == "supported"
    assert say(chat, "Nala est-elle un reptile ?")["meta"]["status"] == "unknown"


def test_unfamiliar_symbols_and_no_reverse_inference(chat):
    say(chat, "Zog is a dax; All daxs are mips")
    assert say(chat, "Is Zog a mip?")["meta"]["status"] == "supported"
    say(chat, "Vek is a mip")
    assert say(chat, "Is Vek a dax?")["meta"]["status"] == "unknown"
    say(chat, "Mip is a florp")
    assert say(chat, "Is Vek a florp?")["meta"]["status"] == "unknown"


def test_conflict_correction_and_recomputed_conclusions(chat):
    a, _, _ = chat
    say(chat, "Nala est un chat; Les chats sont des animaux")
    assert say(chat, "Nala est un animal ?")["meta"]["status"] == "supported"
    before = a.ltm.count("episodes")
    r = say(chat, "Nala n'est pas un chat")
    assert r["meta"]["conflict"]
    k = a.ltm.knowledge(kind="semantic")[-1]
    evidence = a.ltm.explain(k.id)
    assert evidence["evidence"]["contradicts"]["total"] == 1
    predicted = json.loads(evidence["evidence"]["supports"]["examples"][0]["predictions"])["0"]
    assert predicted["predicted"] is False
    assert predicted["confidence"] is None
    assert say(chat, "Nala est un chat ?")["meta"]["status"] == "conflict"
    assert say(chat, "Nala est un animal ?")["meta"]["status"] == "unknown"
    say(chat, "correction : Nala n'est pas un chat")
    assert say(chat, "Nala est un chat ?")["meta"]["status"] == "opposed"
    assert say(chat, "Nala est un animal ?")["meta"]["status"] == "unknown"
    old = next(k for k in a.ltm.knowledge(kind="semantic") if not k.expr["negative"]
               and k.expr["relation"] == "instance")
    ex = a.ltm.explain(old.id)
    assert old.status == "refuted"
    assert ex["evidence"]["supports"]["total"] == 1
    assert ex["evidence"]["refuted_by"]["total"] == 1
    assert a.ltm.count("episodes") > before
    say(chat, "correction : Nala est un chat")
    assert say(chat, "Nala est un animal ?")["meta"]["status"] == "supported"


def test_derived_conflict_blocks_downstream_inference(chat):
    say(chat, "Nala est un chat; Les chats sont des animaux; Les animaux sont des organismes")
    say(chat, "Nala n'est pas un animal")
    assert say(chat, "Nala est un animal ?")["meta"]["status"] == "conflict"
    assert say(chat, "Nala est un organisme ?")["meta"]["status"] == "unknown"


def test_context_and_persistence(tmp_path):
    db = str(tmp_path / "memory.db")
    with_agent = Genesis(db)
    e = ChatEngine(with_agent)
    cid = e.new_conversation("fr")["id"]
    e.send(cid, "Nala est un chat")
    e.send(cid, "Les chats sont des animaux")
    with_agent.close()
    a = Genesis(db)
    try:
        e = ChatEngine(a)
        assert e.send(cid, "Elle est un animal ?")["reply"]["meta"]["status"] == "supported"
        other = e.new_conversation("fr")["id"]
        assert e.send(other, "Elle est un animal ?")["reply"]["meta"]["intent"] == "semantic-clarify"
        assert e.send(other, "Nala est un animal ?")["reply"]["meta"]["status"] == "supported"
    finally:
        a.close()


def test_ambiguous_context_and_scoped_negation(chat):
    say(chat, "Nala est un chat; Milo n'est pas un chat")
    assert say(chat, "Elle est un chat ?")["meta"]["intent"] == "semantic-clarify"
    assert say(chat, "Nala est un chat ?")["meta"]["status"] == "supported"
    assert say(chat, "Milo est un chat ?")["meta"]["status"] == "opposed"


def test_location_is_not_transitive(chat):
    say(chat, "Nala est dans Paris; Paris est dans France")
    assert say(chat, "Nala est dans Paris ?")["meta"]["status"] == "supported"
    assert say(chat, "Nala est dans France ?")["meta"]["status"] == "unknown"


def test_cycles_and_resource_limit(chat):
    a, e, cid = chat
    say(chat, "Zog is a dax; All daxs are mips; All mips are daxs")
    assert say(chat, "Is Zog a mip?")["meta"]["status"] == "supported"
    assert say(chat, "Is Zog a florp?")["meta"]["status"] == "unknown"
    e.semantic.memory.MAX_FACTS = len(e.semantic.memory.items())
    before = a.ltm.count("episodes")
    assert say(chat, "Vek is a florp")["meta"]["intent"] == "refused"
    assert a.ltm.count("episodes") == before


def test_no_teaching_from_questions_or_unsupported_clauses(chat):
    a, _, _ = chat
    say(chat, "Nala est un chat ?")
    say(chat, "Si Nala est un chat")
    assert say(chat, "Nala est un chat; peut-être")["meta"]["intent"] == "semantic-clarify"
    assert say(chat, "Nala est un chat; explorer 100")["meta"]["intent"] == "semantic-clarify"
    say(chat, "Is Nala a cat?")
    assert a.ltm.knowledge(kind="semantic") == []
    assert parse_query("Est-ce que Nala est un chat ?") == Fact("nala", "instance", "chat")
    assert parse_fact("Nala est un chat") == Fact("nala", "instance", "chat")


def test_repetition_does_not_inflate_confidence_and_consolidation_preserves_facts(chat):
    a, _, _ = chat
    say(chat, "Nala est un chat")
    say(chat, "Nala est un chat")
    k = a.ltm.knowledge(kind="semantic")[0]
    assert k.confidence == 0.5 and k.status == "taught"
    assert k.n_support == 2
    from genesis.consolidation import Consolidation
    Consolidation().run(a)
    assert say(chat, "Nala est un chat ?")["meta"]["status"] == "supported"


def test_recall_explanation_local_context_and_retraction(chat):
    a, _, _ = chat
    r = say(chat, "Nala est un chat. Elle est dans Paris.")
    assert r["meta"]["intent"] == "semantic-teach"
    r = say(chat, "Les chats sont des animaux")
    rule_id = r["meta"]["knowledge_ids"][0]
    r = say(chat, "Elle est un animal ?")
    assert r["meta"]["status"] == "supported"
    explanation = say(chat, "pourquoi ?")
    assert explanation["meta"]["intent"] == "semantic-explain"
    assert "Les chats sont des animaux" in explanation["text"]
    assert "Nala est un chat" in explanation["text"]
    assert "instance chat" in say(chat, "que sais-tu de Nala ?")["text"]
    assert say(chat, f"retire le fait {rule_id}")["meta"]["intent"] == "semantic-retract"
    assert say(chat, "Elle est un animal ?")["meta"]["status"] == "unknown"
    assert say(chat, "Elle est dans Paris ?")["meta"]["status"] == "supported"
    assert a.ltm.explain(rule_id)["evidence"]["refuted_by"]["total"] == 1
    assert say(chat, "retire le fait 999")["meta"]["intent"] == "refused"


@pytest.mark.parametrize("seed", [0, 1])
def test_reproducible_semantic_transfer_benchmark(seed):
    from genesis.experiments import run_semantic_learning
    r = run_semantic_learning(seed)
    assert r["initial_knowledge"] == 0
    assert r["baseline_unknown"] == r["heldout_entities"] == 6
    assert r["transfer_accuracy"] == 1
    assert r["proofs_with_three_premises"] == 6
    assert r["negative_control_false_positives"] == 0
    assert r["retention_after_unrelated_update"] == r["retention_after_reload"] == 1
    assert r["conflict_detected"] and r["stale_conclusion_removed"]
    assert r["unaffected_retention"] == 1
