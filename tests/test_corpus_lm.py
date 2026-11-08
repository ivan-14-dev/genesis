import json
import random
import sqlite3

import pytest

from genesis import corpus, crypto, governance
from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.lm import registry
from genesis.memory import LongTermMemory

SEED = bytes(range(32))
PUB = crypto.public_key(SEED)
S = ["le chat", "la souris", "le chien", "l'oiseau", "le renard"]
V = ["mange", "regarde", "suit", "voit"]
O = ["le fromage", "la graine", "le poisson", "la pomme", "le pain"]


def _doc(rng, n=25):
    return " ".join(f"{rng.choice(S)} {rng.choice(V)} {rng.choice(O)}." for _ in range(n))


@pytest.fixture
def ltm(tmp_path):
    m = LongTermMemory(str(tmp_path / "c.db"))
    governance.pin_admin_key(m, PUB)
    yield m
    m.close()


def _admin(ltm):
    return governance.Guard(governance.load_policy(ltm), ltm, "alice", "admin")


def _fill(ltm, n=30, approve=True):
    rng, g = random.Random(7), _admin(ltm)
    ids = []
    for i in range(n):
        ids.append(corpus.submit(ltm, g, f"d{i}", _doc(rng))["id"])
        if approve:
            corpus.review(ltm, SEED, ids[-1], True, "ivan")
    return ids


# ------------------------------------------------------------------ corpus
def test_redaction_of_personal_data():
    text = ("mail jean.dupont@exemple.fr, tel +33 6 12 34 56 78 ou 06.12.34.56.78, iban FR76 3000 6000 0112 3456 "
            "7890 189, carte 4111 1111 1111 1111, faux 1234 5678 9012 3456, ip 192.168.1.20, version 1.2.3")
    out, counts = corpus.redact(text)
    assert "jean.dupont" not in out and "<EMAIL>" in out
    assert "4111" not in out and "<CARD>" in out
    assert "1234 5678 9012 3456" in out  # fails the Luhn check: not a card number
    assert "192.168" not in out and "version 1.2.3" in out
    assert "FR76" not in out and counts["IBAN"] == 1
    assert counts["PHONE"] == 2 and counts["EMAIL"] == 1 and counts["CARD"] == 1 and counts["IP"] == 1


def test_submit_normalizes_dedups_and_splits_deterministically(ltm):
    g = _admin(ltm)
    r = corpus.submit(ltm, g, "a", "  Cafe\u0301\x00 au lait  ")
    row = ltm.db.execute("SELECT text, status, split FROM corpus_docs WHERE id=?", (r["id"],)).fetchone()
    assert row["text"] == "Café au lait" and row["status"] == "pending"
    assert row["split"] == corpus.split_of(r["sha256"]) == r["split"]
    with pytest.raises(ValueError, match="already"):
        corpus.submit(ltm, g, "b", "Café au lait")
    with pytest.raises(ValueError, match="empty"):
        corpus.submit(ltm, g, "c", " \x01 ")
    splits = {corpus.split_of(f"{i:08x}" + "0" * 56) for i in range(0, 2 ** 32, 2 ** 25)}
    assert splits == {"train", "valid", "test"}


def test_submit_respects_policy(ltm):
    doc = governance.validate_policy({"serial": 1, "blocked_terms": ["secret défense"],
                                      "limits": {"max_corpus_doc_chars": 50}})
    governance.install_policy(ltm, governance.sign_policy(SEED, doc))
    policy = governance.load_policy(ltm)
    with pytest.raises(governance.PolicyDenied):
        corpus.submit(ltm, governance.Guard(policy, ltm, "bob", "user"), "x", "bonjour")
    teacher = governance.Guard(policy, ltm, "tom", "teacher")
    with pytest.raises(governance.PolicyDenied, match="blocked"):
        corpus.submit(ltm, teacher, "x", "un document SECRET-DEFENSE ? non: Secret Défense.")
    with pytest.raises(ValueError, match="longer"):
        corpus.submit(ltm, teacher, "x", "y" * 51)
    assert ltm.count("corpus_docs") == 0
    assert any(e["decision"] == "deny" for e in ltm.audit_entries())


def test_only_signed_untampered_documents_reach_training(ltm):
    ids = _fill(ltm, 6, approve=False)
    assert corpus.approved_texts(ltm, "train") == []
    with pytest.raises(PermissionError):
        corpus.review(ltm, bytes(32), ids[0], True)  # not the pinned admin key
    ltm.db.execute("UPDATE corpus_docs SET status='approved'")  # forged without a signature
    assert all(not corpus.approved_texts(ltm, s) for s in ("train", "valid", "test"))
    for i in ids:
        corpus.review(ltm, SEED, i, True, "ivan")
    total = sum(len(corpus.approved_texts(ltm, s)) for s in ("train", "valid", "test"))
    assert total == 6 and corpus.verify(ltm)["ok"]
    ltm.db.execute("UPDATE corpus_docs SET text = text || ' poison' WHERE id=?", (ids[0],))
    ltm.db.execute("UPDATE corpus_docs SET sha256 = 'x' WHERE id=?", (ids[1],))
    assert set(corpus.verify(ltm)["tampered_documents"]) == {ids[0], ids[1]}
    total = sum(len(corpus.approved_texts(ltm, s)) for s in ("train", "valid", "test"))
    assert total == 4


def test_reading_the_test_split_is_audited(ltm):
    _fill(ltm, 12)
    before = len(ltm.audit_entries())
    corpus.approved_texts(ltm, "train")
    assert len(ltm.audit_entries()) == before
    corpus.approved_texts(ltm, "test", "carol")
    last = ltm.audit_entries()[-1]
    assert last["action"] == "corpus.read-test" and last["actor"] == "carol"


# ------------------------------------------------------------- registry
def test_ngram_training_provenance_and_baseline(ltm):
    _fill(ltm)
    res = registry.train(ltm, _admin(ltm), "ngram", order=3, vocab_size=320)
    v = res["metrics"]["valid"]
    assert v["perplexity"] < v["unigram_perplexity"] / 3
    assert v["bits_per_byte"] < v["unigram_bits_per_byte"]
    p = res["provenance"]
    assert p["pretrained"] is None and p["external_data"] is False and p["trained_by"] == "alice"
    assert len(p["corpus"]) == res["metrics"]["train"]["docs"] and len(p["code"]) == 64
    again = registry.train(ltm, _admin(ltm), "ngram", order=3, vocab_size=320)
    assert again["weights_sha256"] == res["weights_sha256"]  # deterministic
    assert [m["status"] for m in registry.models(ltm)] == ["trained", "trained"]


def test_training_refuses_bad_config_and_unauthorized(ltm):
    _fill(ltm, 4)
    with pytest.raises(ValueError):
        registry.train(ltm, _admin(ltm), "ngram", order=9)
    with pytest.raises(ValueError):
        registry.train(ltm, _admin(ltm), "gpt", dim=30, n_heads=4)
    with pytest.raises(ValueError, match="unknown options"):
        registry.train(ltm, _admin(ltm), "ngram", dim=4)
    with pytest.raises(governance.PolicyDenied):
        registry.train(ltm, governance.Guard(governance.load_policy(ltm), ltm, "t", "teacher"), "ngram")


def test_approval_activation_and_integrity(ltm):
    _fill(ltm)
    mid = registry.train(ltm, _admin(ltm), "ngram", order=3, vocab_size=320)["id"]
    assert registry.load_active(ltm) is None
    with pytest.raises(PermissionError):
        registry.approve(ltm, bytes(32), mid)
    registry.approve(ltm, SEED, mid, "ivan")
    model = registry.load_active(ltm)
    a = model.generate("le chat", 20, seed=1)
    assert a == model.generate("le chat", 20, seed=1) and a["tokens"] > 0
    assert registry.verify(ltm)["ok"]

    weights = ltm.db.execute("SELECT weights FROM lm_models WHERE id=?", (mid,)).fetchone()[0]
    ltm.db.execute("UPDATE lm_models SET weights=? WHERE id=?", (weights[:-1] + bytes([weights[-1] ^ 1]), mid))
    registry._CACHE.clear()
    with pytest.raises(registry.LMIntegrityError):
        registry.load_active(ltm)
    assert not registry.verify(ltm)["ok"]
    ltm.db.execute("UPDATE lm_models SET weights=? WHERE id=?", (weights, mid))

    # an approval cannot be moved to another model
    other = registry.train(ltm, _admin(ltm), "ngram", order=2, vocab_size=300)["id"]
    approval = ltm.db.execute("SELECT approval FROM lm_models WHERE id=?", (mid,)).fetchone()[0]
    ltm.db.execute("UPDATE lm_models SET status='active', approval=? WHERE id=?", (approval, other))
    ltm.set_meta("active_lm", other)
    with pytest.raises(registry.LMIntegrityError, match="signature"):
        registry.load_active(ltm)


def test_approval_requires_beating_the_baseline(ltm):
    _fill(ltm)
    mid = registry.train(ltm, _admin(ltm), "ngram", order=3, vocab_size=320)["id"]
    metrics = json.loads(ltm.db.execute("SELECT metrics FROM lm_models WHERE id=?", (mid,)).fetchone()[0])
    metrics["valid"]["perplexity"] = metrics["valid"]["unigram_perplexity"] + 1
    ltm.db.execute("UPDATE lm_models SET metrics=? WHERE id=?", (json.dumps(metrics), mid))
    with pytest.raises(PermissionError, match="baseline"):
        registry.approve(ltm, SEED, mid)
    assert registry.approve(ltm, SEED, mid, force=True)["status"] == "active"
    assert registry.deactivate(ltm, SEED, "ivan") == mid and registry.load_active(ltm) is None


def test_evaluate_on_sealed_test_split(ltm):
    _fill(ltm)
    mid = registry.train(ltm, _admin(ltm), "ngram", order=3, vocab_size=320)["id"]
    res = registry.evaluate(ltm, mid, "ivan")
    assert res["perplexity"] < res["unigram_perplexity"]
    assert ltm.audit_entries()[-1]["action"] == "corpus.read-test"
    assert "test" in registry.models(ltm)[0]["metrics"]


def test_small_transformer_trains_here(ltm):
    _fill(ltm, 12)
    res = registry.train(ltm, _admin(ltm), "gpt", vocab_size=290, dim=16, n_layers=1, n_heads=2, block_size=16,
                         steps=30, batch_size=2, lr=1e-2)
    assert res["metrics"]["train"]["params"] < 20000
    assert res["provenance"]["pretrained"] is None
    registry.approve(ltm, SEED, res["id"], force=True)
    out = registry.load_active(ltm).generate("le chien", 10, seed=0)
    assert out["tokens"] <= 10 and isinstance(out["text"], str)


def test_blob_format_rejects_garbage_and_bombs():
    import zlib
    with pytest.raises(registry.LMIntegrityError):
        registry._unpack(zlib.compress(b"NOPE" + b"\0" * 100))
    bomb = zlib.compress(b"\0" * (registry.MAX_BLOB + 10), 9)
    with pytest.raises(registry.LMIntegrityError, match="too large"):
        registry._unpack(bomb)


# ------------------------------------------------------------------ chat
def test_chat_generation_is_gated_labelled_and_never_knowledge(tmp_path):
    db = str(tmp_path / "chat.db")
    agent = Genesis(db)
    try:
        governance.pin_admin_key(agent.ltm, PUB)
        eng = ChatEngine(agent)
        cid = eng.new_conversation("fr")["id"]
        r = eng.send(cid, "complète : le chat")["reply"]
        assert r["meta"]["intent"] == "lm-generate" and r["meta"]["active"] is False
        _fill(agent.ltm)
        mid = registry.train(agent.ltm, _admin(agent.ltm), "ngram", order=3, vocab_size=320)["id"]
        registry.approve(agent.ltm, SEED, mid, "ivan")
        before = agent.ltm.count("knowledge")
        r = eng.send(cid, "complète : le chat")["reply"]
        assert r["meta"]["intent"] == "lm-generate" and r["meta"]["model_id"] == mid
        assert "non vérifié" in r["text"] and r["text"].rstrip().split("\n")[-1].startswith("le chat")
        r2 = eng.send(eng.new_conversation("en")["id"], "generate: le chat")["reply"]
        assert r2["meta"]["seed"] == r["meta"]["seed"] and "unverified" in r2["text"]
        assert agent.ltm.count("knowledge") == before

        doc = governance.validate_policy({"serial": 1, "blocked_terms": ["fromage"],
                                          "roles": {**governance.DEFAULT_POLICY["roles"], "user": ["chat.read"]}})
        governance.install_policy(agent.ltm, governance.sign_policy(SEED, doc))
        eng.guard = governance.Guard(governance.load_policy(agent.ltm), agent.ltm)
        withheld = [eng.send(cid, f"complète : {s}")["reply"]["meta"] for s in S]
        assert any(m.get("withheld") for m in withheld)
        eng.guard = governance.Guard(governance.load_policy(agent.ltm), agent.ltm, "bob", "user")
        cid2 = eng.new_conversation("fr")["id"]
        r = eng.send(cid2, "complète : le chat")["reply"]
        assert r["meta"]["intent"] == "policy-denied" and r["meta"]["action"] == "lm.generate"
    finally:
        agent.close()


def test_chat_reports_tampered_model(tmp_path):
    agent = Genesis(str(tmp_path / "t.db"))
    try:
        governance.pin_admin_key(agent.ltm, PUB)
        _fill(agent.ltm)
        mid = registry.train(agent.ltm, _admin(agent.ltm), "ngram", order=2, vocab_size=300)["id"]
        registry.approve(agent.ltm, SEED, mid)
        agent.ltm.db.execute("UPDATE lm_models SET weights_sha256='0' WHERE id=?", (mid,))
        eng = ChatEngine(agent)
        r = eng.send(eng.new_conversation("en")["id"], "complete: the cat")["reply"]
        assert r["meta"]["error"] == "integrity"
    finally:
        agent.close()


def test_schema_tables_exist(tmp_path):
    LongTermMemory(str(tmp_path / "s.db")).close()
    names = {r[0] for r in sqlite3.connect(str(tmp_path / "s.db")).execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"corpus_docs", "lm_models", "audit_log", "audit_seals"} <= names
