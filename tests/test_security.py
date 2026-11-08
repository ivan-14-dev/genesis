"""Admin policy inside the chat engine, and the HTTP hardening of the chat server and dashboard."""
import base64
import hashlib
import http.client
import itertools
import json
import sqlite3
import threading
import time
from http.server import HTTPServer

import pytest

from genesis import crypto, governance
from genesis import agent as agent_mod
from genesis import chat_server, dashboard
from genesis.agent import Genesis
from genesis.chat import ChatEngine
from genesis.environments import SequenceWorld, sandboxed
from genesis.memory import LongTermMemory

SEED = bytes(range(1, 33))
PUB = crypto.public_key(SEED)
ROLES = governance.DEFAULT_POLICY["roles"]


def _install(ltm, **doc):
    governance.pin_admin_key(ltm, PUB)
    serial = (governance.load_policy(ltm).serial if ltm.get_meta("policy") else 0) + 1
    governance.install_policy(ltm, governance.sign_policy(SEED, {"serial": serial, **doc}))
    return governance.load_policy(ltm)


@pytest.fixture
def agent(tmp_path):
    a = Genesis(str(tmp_path / "p.db"))
    yield a
    a.close()


def _engine(agent, actor="local", role=None, **doc):
    policy = _install(agent.ltm, **doc) if doc else governance.load_policy(agent.ltm)
    return ChatEngine(agent, lambda s: sandboxed(SequenceWorld(s), 1000),
                      guard=governance.Guard(policy, agent.ltm, actor, role or policy.doc["default_role"]))


# ------------------------------------------------------------ chat engine
def test_role_permissions_are_enforced_and_audited(agent):
    eng = _engine(agent, "bob", "user", users=[])
    cid = eng.new_conversation("en")["id"]
    r = eng.send(cid, "Nala is a cat")["reply"]
    assert r["meta"]["intent"] == "policy-denied" and r["meta"]["action"] == "semantic.teach"
    assert agent.ltm.count("knowledge") == 0
    assert eng.send(cid, "explore 50")["reply"]["meta"]["intent"] == "policy-denied"
    assert eng.send(cid, "3 6 9 12 15 18")["reply"]["meta"]["intent"] == "sequence"  # users may learn sequences
    denials = [e for e in agent.ltm.audit_entries() if e["decision"] == "deny"]
    assert {e["action"] for e in denials} >= {"semantic.teach", "world.explore"}
    assert all(e["actor"] == "bob" for e in denials)
    assert governance.verify_audit(agent.ltm, PUB)["ok"]


def test_disabled_domain_refuses_even_teachers(agent):
    eng = _engine(agent, domains={"semantic": False})
    cid = eng.new_conversation("fr")["id"]
    r = eng.send(cid, "Nala est un chat")["reply"]
    assert r["meta"]["intent"] == "policy-denied" and "désactivé" in r["text"]
    assert agent.ltm.count("knowledge") == 0


def test_refused_message_content_is_not_stored(agent):
    eng = _engine(agent, blocked_terms=["mot de passe"])
    cid = eng.new_conversation("fr")["id"]
    r = eng.send(cid, "Mon MOT DE PASSE est hunter2")
    assert r["reply"]["meta"]["intent"] == "policy-denied"
    texts = [m["text"] for m in eng.messages(cid)]
    assert not any("hunter2" in t for t in texts) and "[message refusé par la politique]" in texts
    row = sqlite3.connect(agent.ltm.path).execute("SELECT COUNT(*) FROM chat_messages WHERE text LIKE '%hunter2%'")
    assert row.fetchone()[0] == 0


def test_message_length_limit(agent):
    eng = _engine(agent, limits={"max_message_chars": 20})
    cid = eng.new_conversation("en")["id"]
    assert eng.send(cid, "x" * 21)["reply"]["meta"]["reason"] == "message too long"


def test_conversations_are_owner_scoped(agent):
    alice = _engine(agent, "alice", "teacher")
    bob = ChatEngine(agent, guard=governance.Guard(governance.load_policy(agent.ltm), agent.ltm, "bob", "teacher"))
    cid = alice.new_conversation("en")["id"]
    alice.send(cid, "2 4 6 8 10 12")
    assert [c["id"] for c in alice.conversations()] == [cid]
    assert bob.conversations() == [] and bob.conversation(cid) is None
    with pytest.raises(KeyError):
        bob.send(cid, "hello")
    with pytest.raises(KeyError):
        bob.messages(cid)


def test_sequence_length_cap_bounds_cost(agent):
    eng = _engine(agent, limits={"max_sequence_terms": 10})
    cid = eng.new_conversation("en")["id"]
    assert eng.send(cid, "1 2 3 4 5 6 7 8")["reply"]["meta"]["intent"] == "sequence"
    r = eng.send(cid, "continue 9 10 11")["reply"]
    assert r["meta"] == {**r["meta"], "intent": "refused", "reason": "max_sequence_terms"}
    assert agent.ltm.count("episodes") == 8


def test_time_budget_stops_long_work(agent, monkeypatch):
    ticks = itertools.count(0, 0.7)

    class FakeTime:
        def __getattr__(self, name):
            return getattr(time, name)

        @staticmethod
        def perf_counter():
            return next(ticks)

    monkeypatch.setattr(agent_mod, "time", FakeTime())
    eng = _engine(agent, limits={"max_seconds_per_message": 1})
    cid = eng.new_conversation("en")["id"]
    r = eng.send(cid, " ".join(str(3 * i) for i in range(40)))["reply"]
    assert r["meta"].get("stopped") == "time budget"
    assert agent.ltm.count("episodes") < 40


# ------------------------------------------------------------------ HTTP
@pytest.fixture
def serve(tmp_path):
    started = []

    def start(policy=None, allow_hosts=(), expected_pub=None):
        ready, box = threading.Event(), {}

        def run():
            a = Genesis(str(tmp_path / "http.db"))
            if policy is not None:
                _install(a.ltm, **policy)
            engine = ChatEngine(a)
            srv = HTTPServer(("127.0.0.1", 0), chat_server.make_handler(engine, allow_hosts, expected_pub))
            box.update(srv=srv, engine=engine, port=srv.server_port, db=a.ltm.path)
            ready.set()
            try:
                srv.serve_forever(poll_interval=0.05)
            finally:
                srv.server_close()
                a.close()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        assert ready.wait(10)
        started.append((box, t))
        return box

    yield start
    for box, t in started:
        box["srv"].shutdown()
        t.join(10)


def _http(port, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        c.putheader("Host", host or f"127.0.0.1:{port}")
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        h = {"Content-Type": "application/json"} if data is not None else {}
        if data is not None:
            h["Content-Length"] = str(len(data))
        h.update(headers or {})
        for k, v in h.items():
            c.putheader(k, v)
        c.endheaders()
        if data and int(h.get("Content-Length", "0") or 0) > 0:
            c.send(data)
        r = c.getresponse()
        raw = r.read()
        try:
            payload = json.loads(raw) if raw and r.getheader("Content-Type", "").startswith("application/json") \
                else raw
        except ValueError:
            payload = raw
        return r.status, payload, r
    finally:
        c.close()


def test_dns_rebinding_host_check(serve):
    box = serve(allow_hosts=("genesis.lan",))
    p = box["port"]
    assert _http(p, "GET", "/", host="evil.example")[0] == 421
    assert _http(p, "GET", "/api/status", host=f"evil.example:{p}")[0] == 421
    assert _http(p, "POST", "/api/conversations", {}, host=f"attacker.test:{p}")[0] == 421
    for ok in (f"127.0.0.1:{p}", f"localhost:{p}", f"genesis.lan:{p}", f"LOCALHOST:{p}"):
        assert _http(p, "GET", "/api/status", host=ok)[0] == 200, ok
    assert _http(p, "GET", "/", host="localhost:1")[0] == 421  # wrong port


def test_malformed_content_length(serve):
    p = serve()["port"]
    for value, code in (("-1", 400), ("abc", 400), (str(10 ** 9), 413)):
        assert _http(p, "POST", "/api/conversations", b"", {"Content-Length": value})[0] == code, value


def test_cross_site_requests_refused(serve):
    p = serve()["port"]
    assert _http(p, "POST", "/api/conversations", {}, {"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert _http(p, "POST", "/api/conversations", {}, {"Sec-Fetch-Site": "same-site"})[0] == 403
    assert _http(p, "POST", "/api/conversations", {}, {"Origin": "http://evil.example"})[0] == 403
    assert _http(p, "POST", "/api/conversations", {}, {"Origin": f"http://127.0.0.1:{p}",
                                                        "Sec-Fetch-Site": "same-origin"})[0] == 200


def test_security_headers_and_csp(serve):
    p = serve()["port"]
    code, page, r = _http(p, "GET", "/")
    csp = r.getheader("Content-Security-Policy")
    assert code == 200 and "default-src 'none'" in csp and "frame-ancestors 'none'" in csp
    script = page.decode().split("<script>", 1)[1].split("</script>", 1)[0]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in csp and "unsafe" not in csp
    assert " on" + "click=" not in page.decode()  # no inline handlers: they would be blocked by the CSP
    assert r.getheader("X-Frame-Options") == "DENY" and r.getheader("X-Content-Type-Options") == "nosniff"
    assert _http(p, "GET", "/api/status")[2].getheader("Cache-Control") == "no-store"


def test_tokens_cookie_and_roles(serve):
    alice_tok, alice_hash = governance.new_token()
    bob_tok, bob_hash = governance.new_token()
    p = serve({"require_auth": True, "users": [{"name": "alice", "role": "teacher", "token_sha256": alice_hash},
                                               {"name": "bob", "role": "user", "token_sha256": bob_hash}]})["port"]
    assert _http(p, "GET", "/api/status")[0] == 401
    assert _http(p, "GET", "/api/status", headers={"Authorization": "Bearer wrong-token-123456"})[0] == 401
    assert _http(p, "GET", "/?token=" + "x" * 40)[0] == 401
    code, _, r = _http(p, "GET", "/?token=" + alice_tok)
    cookie = r.getheader("Set-Cookie")
    assert code == 303 and r.getheader("Location") == "/" and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    alice = {"Cookie": cookie.split(";")[0]}
    assert _http(p, "GET", "/api/whoami", headers=alice)[1]["user"] == "alice"
    bob = {"Authorization": "Bearer " + bob_tok}
    who = _http(p, "GET", "/api/whoami", headers=bob)[1]
    assert who["role"] == "user" and "semantic.teach" not in who["permissions"]

    cid = _http(p, "POST", "/api/conversations", {"lang": "en"}, alice)[1]["id"]
    assert _http(p, "GET", f"/api/conversations/{cid}", headers=bob)[0] == 404
    assert _http(p, "POST", f"/api/conversations/{cid}/messages", {"text": "hi"}, bob)[0] == 404
    assert [c["id"] for c in _http(p, "GET", "/api/conversations", headers=alice)[1]] == [cid]
    assert _http(p, "GET", "/api/conversations", headers=bob)[1] == []
    bcid = _http(p, "POST", "/api/conversations", {"lang": "en"}, bob)[1]["id"]
    reply = _http(p, "POST", f"/api/conversations/{bcid}/messages", {"text": "Nala is a cat"}, bob)[1]["reply"]
    assert reply["meta"]["intent"] == "policy-denied"


def test_rate_limit(serve):
    p = serve({"limits": {"requests_per_minute": 3}})["port"]
    codes = [_http(p, "GET", "/api/status")[0] for _ in range(5)]
    assert codes[:3] == [200] * 3 and codes[3:] == [429, 429]


def test_internal_errors_do_not_leak(serve, capsys):
    box = serve()

    def boom():
        raise RuntimeError("secret-internal-detail")
    box["engine"].conversations = boom
    code, body, _ = _http(box["port"], "GET", "/api/conversations")
    assert code == 500 and "ref" in body["error"] and "secret" not in json.dumps(body)
    assert "secret-internal-detail" in capsys.readouterr().err


def test_tampered_policy_stops_service(serve):
    box = serve({"name": "strict"})
    p = box["port"]
    assert _http(p, "GET", "/api/status")[0] == 200
    db = sqlite3.connect(box["db"])
    signed = json.loads(db.execute("SELECT value FROM meta WHERE key='policy'").fetchone()[0])
    signed["statement"]["roles"]["user"] = ["*"]
    db.execute("UPDATE meta SET value=? WHERE key='policy'", (json.dumps(signed),))
    db.commit()
    db.close()
    assert _http(p, "GET", "/api/status")[0] == 503


def test_expected_admin_key(serve):
    p = serve({"name": "x"}, expected_pub=crypto.public_key(bytes(32)))["port"]
    assert _http(p, "GET", "/api/status")[0] == 503


# -------------------------------------------------------------- dashboard
@pytest.fixture
def dash(tmp_path):
    db = str(tmp_path / "d.db")
    a = Genesis(db)
    a.close()
    srv = HTTPServer(("127.0.0.1", 0), dashboard.make_handler(db))
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    yield srv.server_port
    srv.shutdown()
    srv.server_close()
    t.join(10)


def test_dashboard_host_check_limits_and_read_only(dash):
    assert _http(dash, "GET", "/", host="rebound.example")[0] == 421
    for q in ("-1", "0", "999999999", "abc"):
        assert _http(dash, "GET", f"/api/episodes?limit={q}")[0] == 200, q
    assert dashboard._limit("-1") == 1 and dashboard._limit("10**9") == 50
    assert dashboard._limit(str(10 ** 9)) == dashboard.MAX_LIMIT
    assert _http(dash, "POST", "/api/episodes", {})[0] == 405
    code, page, r = _http(dash, "GET", "/")
    assert code == 200 and "script-src 'sha256-" in r.getheader("Content-Security-Policy")
    assert " on" + "click=" not in page.decode()


def test_memory_files_are_private(tmp_path):
    import os
    import stat
    m = LongTermMemory(str(tmp_path / "priv.db"))
    m.checkpoint(str(tmp_path / "copy.db"))
    m.close()
    for name in ("priv.db", "copy.db"):
        assert stat.S_IMODE(os.stat(tmp_path / name).st_mode) & 0o077 == 0, name
