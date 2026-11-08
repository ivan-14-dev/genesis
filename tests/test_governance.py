import json

import pytest

from genesis import crypto
from genesis import governance as G
from genesis.memory import LongTermMemory

ADMIN = bytes(range(32))
OTHER = bytes(range(1, 33))


def _ltm(tmp_path, name="m.db"):
    return LongTermMemory(str(tmp_path / name))


def _install(ltm, **doc):
    return G.install_policy(ltm, G.sign_policy(ADMIN, doc))


# ------------------------------------------------------------------ validation
def test_default_policy_preserves_open_local_use(tmp_path):
    ltm = _ltm(tmp_path)
    p = G.load_policy(ltm)
    assert not p.signed and p.identify(None) == ("local", "teacher")
    assert p.allowed("teacher", "semantic.teach") and p.allowed("teacher", "objects.teach")
    assert not p.allowed("teacher", "lm.approve") and not p.allowed("user", "semantic.teach")
    assert not p.allowed("admin", "made.up")  # default deny for unknown actions, even for '*'


@pytest.mark.parametrize("doc,msg", [
    ({"limits": {"max_message_chars": 10 ** 6}}, "frozen ceiling"),
    ({"limits": {"max_message_chars": 0}}, "positive"),
    ({"limits": {"unknown": 3}}, "unknown limit"),
    ({"roles": {"x": ["files.delete"]}, "default_role": "x"}, "permissions"),
    ({"roles": {"Bad Role": ["*"]}, "default_role": None}, "invalid role"),
    ({"domains": {"chat": False}}, "domains"),
    ({"users": [{"name": "bob", "role": "ghost", "token_sha256": "0" * 64}]}, "undefined role"),
    ({"users": [{"name": "local", "role": "user", "token_sha256": "0" * 64}]}, "user name"),
    ({"surprise": 1}, "unknown policy keys"),
    ({"serial": -1}, "serial"),
])
def test_invalid_policies_are_refused(doc, msg):
    with pytest.raises(ValueError, match=msg):
        G.validate_policy(doc)


def test_blocked_terms_are_folded_and_word_bounded():
    p = G.Policy(G.validate_policy({"blocked_terms": ["Épée"]}))
    assert p.blocked("une EPEE ici") == "epee"
    assert p.blocked("epees") is None


# ----------------------------------------------------------- signed installs
def test_signed_policy_install_rollback_and_tamper(tmp_path):
    ltm = _ltm(tmp_path)
    signed = G.sign_policy(ADMIN, {"serial": 1, "domains": {"objects": False}})
    with pytest.raises(PermissionError, match="pin"):
        G.install_policy(ltm, signed)
    G.pin_admin_key(ltm, crypto.public_key(ADMIN))
    with pytest.raises(PermissionError, match="already pinned"):
        G.pin_admin_key(ltm, crypto.public_key(OTHER))
    with pytest.raises(PermissionError, match="does not verify"):
        G.install_policy(ltm, G.sign_policy(OTHER, {"serial": 1}))
    p = G.install_policy(ltm, signed)
    assert p.signed and not p.allowed("teacher", "objects.explore")
    with pytest.raises(PermissionError, match="serial must increase"):
        _install(ltm, serial=1)
    _install(ltm, serial=2)
    assert G.load_policy(ltm).serial == 2
    assert [h["serial"] for h in ltm.get_meta("policy_history")] == [1, 2]

    # Editing the stored policy (e.g. granting a role more rights) breaks the signature.
    stored = ltm.get_meta("policy")
    stored["statement"]["roles"]["user"] = ["*"]
    ltm.set_meta("policy", stored)
    with pytest.raises(G.PolicyIntegrityError):
        G.load_policy(ltm)
    # So does swapping the pinned key for an attacker's.
    ltm.set_meta("policy", G.sign_policy(OTHER, {"serial": 9}))
    ltm.set_meta("admin_pubkey", crypto.public_key(OTHER).hex())
    with pytest.raises(G.PolicyIntegrityError, match="expected"):
        G.load_policy(ltm, expected_pub=crypto.public_key(ADMIN))


def test_tokens_roles_and_auth(tmp_path):
    ltm = _ltm(tmp_path)
    G.pin_admin_key(ltm, crypto.public_key(ADMIN))
    tok, h = G.new_token()
    p = _install(ltm, serial=1, require_auth=True, users=[{"name": "bob", "role": "user", "token_sha256": h}])
    assert p.identify(tok) == ("bob", "user")
    with pytest.raises(G.AuthError):
        p.identify(None)
    with pytest.raises(G.AuthError):
        p.identify(tok + "x")
    assert tok not in json.dumps(ltm.get_meta("policy"))  # only the hash is stored


# ----------------------------------------------------------------- guard + audit
def test_guard_denies_before_writing_and_audits(tmp_path):
    ltm = _ltm(tmp_path)
    G.pin_admin_key(ltm, crypto.public_key(ADMIN))
    p = _install(ltm, serial=1, domains={"semantic": False}, blocked_terms=["secret"],
                 limits={"max_message_chars": 20})
    g = G.Guard(p, ltm, "bob", "teacher")
    g.require("sequence.learn", "conv:1")
    with pytest.raises(G.PolicyDenied, match="domain disabled"):
        g.require("semantic.teach")
    with pytest.raises(G.PolicyDenied, match="not permitted"):
        g.require("lm.approve")
    with pytest.raises(G.PolicyDenied, match="unknown action"):
        g.require("shell.exec")
    with pytest.raises(G.PolicyDenied, match="blocked term"):
        g.check_text("le SECRET")
    with pytest.raises(G.PolicyDenied, match="too long"):
        g.check_text("x" * 21)
    g.require("chat.read")  # reads are not logged unless log_reads
    rows = ltm.audit_entries()
    assert [(r["action"], r["decision"]) for r in rows[-6:]] == [
        ("sequence.learn", "allow"), ("semantic.teach", "deny"), ("lm.approve", "deny"), ("shell.exec", "deny"),
        ("chat.read", "deny"), ("chat.read", "deny")]
    assert all(r["actor"] == "bob" for r in rows[-6:])
    assert G.verify_audit(ltm)["ok"]


def test_audit_chain_detects_edits_deletions_and_truncation(tmp_path):
    ltm = _ltm(tmp_path)
    G.pin_admin_key(ltm, crypto.public_key(ADMIN))
    for i in range(5):
        ltm.append_audit("sequence.learn", "allow", "bob", "user", i)
    sealed = G.seal_audit(ltm, ADMIN)
    assert sealed["entry_id"] == 6 and G.verify_audit(ltm)["ok"]
    with pytest.raises(PermissionError):
        G.seal_audit(ltm, OTHER)

    ltm.db.execute("UPDATE audit_log SET decision='deny' WHERE id=3")
    report = G.verify_audit(ltm)
    assert not report["ok"] and report["first_bad_entry"] == 3
    with pytest.raises(G.PolicyIntegrityError):
        G.seal_audit(ltm, ADMIN)

    ltm2 = _ltm(tmp_path, "b.db")
    G.pin_admin_key(ltm2, crypto.public_key(ADMIN))
    for i in range(5):
        ltm2.append_audit("x", "allow", "bob", "user", i)
    ltm2.db.execute("DELETE FROM audit_log WHERE id=2")
    assert G.verify_audit(ltm2)["first_bad_entry"] == 3

    ltm3 = _ltm(tmp_path, "c.db")
    G.pin_admin_key(ltm3, crypto.public_key(ADMIN))
    for i in range(5):
        ltm3.append_audit("x", "allow", "bob", "user", i)
    G.seal_audit(ltm3, ADMIN)
    ltm3.db.execute("DELETE FROM audit_log WHERE id>=5")  # silently dropping the latest entries
    report = G.verify_audit(ltm3)
    assert report["first_bad_entry"] is None and report["bad_seals"] == [1] and not report["ok"]


def test_signed_approvals_are_bound_to_one_memory(tmp_path):
    a, b = _ltm(tmp_path, "a.db"), _ltm(tmp_path, "b.db")
    for m in (a, b):
        G.pin_admin_key(m, crypto.public_key(ADMIN))
    signed = G.admin_approval(a, ADMIN, "lm-approval", {"model_id": 1}, "alice")
    assert G.check_approval(a, signed, "lm-approval", {"model_id": 1})
    assert not G.check_approval(a, signed, "lm-approval", {"model_id": 2})
    assert not G.check_approval(a, signed, "corpus-approval", {"model_id": 1})
    assert not G.check_approval(b, signed, "lm-approval", {"model_id": 1})  # replay on another memory
    with pytest.raises(PermissionError):
        G.admin_approval(a, OTHER, "lm-approval", {"model_id": 1})


def test_memory_file_is_private_and_migrates(tmp_path):
    import os
    import sqlite3
    import stat
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)  # a v1 memory created before governance existed
    db.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
                     "CREATE TABLE conversations (id INTEGER PRIMARY KEY, created_ts REAL NOT NULL, "
                     "updated_ts REAL NOT NULL, title TEXT, lang TEXT, summary TEXT);"
                     "INSERT INTO conversations(created_ts, updated_ts, title) VALUES (1, 1, 'old');")
    db.commit()
    db.close()
    ltm = LongTermMemory(path)
    assert ltm.get_meta("schema_version") == 2
    assert ltm.conversations(owner="local")[0]["title"] == "old"
    assert ltm.conversations(owner="bob") == [] and ltm.conversation(1, owner="bob") is None
    ltm.close()
    LongTermMemory(path).close()  # idempotent
    fresh = str(tmp_path / "new.db")
    LongTermMemory(fresh).close()
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o600
