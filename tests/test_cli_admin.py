import json
import os
import sqlite3
import stat

import pytest

from genesis.__main__ import main


def _run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    try:
        return code, json.loads(out.out), out.err
    except ValueError:
        return code, out.out, out.err


def test_admin_governance_and_language_model_cli(tmp_path, capsys):
    keys, db = str(tmp_path / "keys"), str(tmp_path / "m.db")
    k = ("--key-dir", keys)
    code, res, _ = _run(capsys, "admin", "keygen", *k)
    assert code == 0 and len(res["key_id"]) == 16
    assert stat.S_IMODE(os.stat(res["private_key"]).st_mode) == 0o600
    assert _run(capsys, "admin", "keygen", *k)[0] == 1  # never overwrites a key
    assert _run(capsys, "admin", "pin", "--db", db, *k)[0] == 0

    code, tok, _ = _run(capsys, "admin", "token", "--name", "alice", "--role", "user")
    policy = {"serial": 1, "name": "test", "users": [tok["policy_users_entry"]],
              "limits": {"max_generate_tokens": 30}}
    (tmp_path / "p.json").write_text(json.dumps(policy))
    code, shown, _ = _run(capsys, "admin", "policy-install", "--db", db, "--file", str(tmp_path / "p.json"), *k)
    assert code == 0 and shown["signed"] and shown["users"] == [{"name": "alice", "role": "user"}]
    assert _run(capsys, "admin", "policy-install", "--db", db, "--file", str(tmp_path / "p.json"), *k)[0] == 1

    lines = ["le chat mange le pain.", "la souris voit la pomme.", "le chien suit le chat."]
    files = []
    for i in range(20):
        f = tmp_path / f"d{i}.txt"
        f.write_text(" ".join(lines[(i + j) % 3] for j in range(12)) + f" fin {i}.")
        files.append(str(f))
    code, added, _ = _run(capsys, "corpus", "add", "--db", db, *files)
    assert code == 0 and all(a["status"] == "pending" for a in added)
    ids = [str(a["id"]) for a in added]
    assert _run(capsys, "corpus", "approve", "--db", db, "--by", "ivan", *ids)[0] == 1  # no key: refused
    capsys.readouterr()
    assert _run(capsys, "corpus", "approve", "--db", db, "--by", "ivan", *k, *ids)[0] == 0

    code, trained, _ = _run(capsys, "lm", "train", "--db", db, "--order", "3", "--vocab-size", "300")
    assert code == 0 and trained["provenance"]["pretrained"] is None
    assert _run(capsys, "lm", "approve", str(trained["id"]), "--db", db, "--by", "ivan", *k)[0] == 0
    code, gen, _ = _run(capsys, "lm", "generate", "le chat", "--db", db, "--max-tokens", "500")
    assert code == 0 and gen["unverified"] and gen["tokens"] <= 30  # capped by the policy

    assert _run(capsys, "audit", "seal", "--db", db, *k)[0] == 0
    code, rep, _ = _run(capsys, "audit", "verify", "--db", db, "--pub", os.path.join(keys, "admin.pub"))
    assert code == 0 and rep["ok"]

    con = sqlite3.connect(db)
    con.execute("UPDATE corpus_docs SET text = text || ' poison' WHERE id = 1")
    con.execute("UPDATE audit_log SET actor = 'mallory' WHERE id = 2")
    con.commit()
    con.close()
    code, rep, _ = _run(capsys, "audit", "verify", "--db", db)
    assert code == 2 and not rep["ok"]
    assert rep["corpus"]["tampered_documents"] == [1] and not rep["audit"]["ok"]


def test_init_generates_web_credentials_once_and_reset_rotates_them(tmp_path, capsys):
    from genesis import governance
    from genesis.memory import LongTermMemory
    home = str(tmp_path / "home")
    code, res, _ = _run(capsys, "init", "--home", home, "--json")
    pw = res["password"]
    assert code == 0 and res["created"] and res["user"] == "admin" and len(pw) == 23
    files = {n: os.path.join(home, n) for n in ("admin.key", "memory.db", "credentials.txt")}
    if os.name == "posix":
        assert stat.S_IMODE(os.stat(home).st_mode) == 0o700
        assert all(stat.S_IMODE(os.stat(f).st_mode) == 0o600 for f in files.values())
    assert pw in open(files["credentials.txt"]).read()
    ltm = LongTermMemory(files["memory.db"], read_only=True)
    policy = governance.load_policy(ltm, bytes.fromhex(open(os.path.join(home, "admin.pub")).read()))
    ltm.close()
    assert policy.signed and policy.doc["require_auth"] and policy.doc["default_role"] is None
    assert policy.authenticate("admin", pw) == ("admin", "admin")

    code, again, _ = _run(capsys, "init", "--home", home)  # idempotent: never prints or replaces credentials
    assert code == 0 and "already initialised" in again and pw not in again
    assert pw in open(files["credentials.txt"]).read()

    code, reset, _ = _run(capsys, "init", "--home", home, "--reset-password", "--json")
    assert code == 0 and reset["password"] != pw and reset["password"] in open(files["credentials.txt"]).read()
    ltm = LongTermMemory(files["memory.db"], read_only=True)
    policy = governance.load_policy(ltm)
    ltm.close()
    assert policy.serial == 2 and policy.authenticate("admin", reset["password"])
    with pytest.raises(governance.AuthError):
        policy.authenticate("admin", pw)
    assert _run(capsys, "start", "--home", str(tmp_path / "missing"))[0] == 1
