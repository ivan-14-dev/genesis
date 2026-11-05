import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from genesis.agent import Genesis
from genesis.chat import ChatEngine, detect_lang
from genesis.chat_server import make_handler
from genesis.environments import SequenceWorld, sandboxed


@pytest.fixture
def engine(tmp_path):
    agent = Genesis(str(tmp_path / "chat.db"))
    yield ChatEngine(agent, lambda s: sandboxed(SequenceWorld(s), 1000))
    agent.close()


def _say(engine, cid, text, lang="en"):
    return engine.send(cid, text, lang)["reply"]


def test_sequence_is_learned_through_the_cognitive_cycle(engine):
    cid = engine.new_conversation("en")["id"]
    r = _say(engine, cid, "3, 7, 11, 15, 19, 23, 27, 31")
    assert r["meta"]["intent"] == "sequence"
    assert r["meta"]["status"] == "validated"
    assert "(x[t-1] + 4)" in r["text"]
    assert "35, 39, 43" in r["text"]
    rows = engine.ltm.db.execute("SELECT provenance FROM episodes").fetchall()
    assert rows and {p for (p,) in rows} == {"user-provided"}


def test_continuing_a_sequence_accumulates_evidence(engine):
    cid = engine.new_conversation("en")["id"]
    r = _say(engine, cid, "1 1 2 3 5 8")
    assert r["meta"]["status"] != "validated"  # too little evidence yet
    r = _say(engine, cid, "continue 13 21 34 55 89 144")
    assert r["meta"]["status"] == "validated"
    assert "x[t-1] + x[t-2]" in r["text"] or "x[t-2] + x[t-1]" in r["text"]


def test_noise_is_reported_unpredictable_not_invented(engine):
    cid = engine.new_conversation("en")["id"]
    r = _say(engine, cid, "5 9 1 7 3 8 2 6 4 0 5 1 9 3")
    assert r["meta"]["status"] == "unpredictable"


def test_explain_cites_supporting_episodes(engine):
    cid = engine.new_conversation("fr")["id"]
    r = _say(engine, cid, "2 4 8 16 32 64 128 256", "fr")
    kid = r["meta"]["knowledge_id"]
    for q in (f"explique {kid}", "pourquoi ?"):
        e = _say(engine, cid, q, "fr")
        assert e["meta"]["intent"] == "explain"
        assert f"#{kid}" in e["text"] and "épisode" in e["text"]
    assert "aucune connaissance #9999" in _say(engine, cid, "explique 9999", "fr")["text"]


def test_unknown_text_is_answered_honestly(engine):
    cid = engine.new_conversation("en")["id"]
    r = _say(engine, cid, "what is the capital of France")
    assert r["meta"]["intent"] == "unknown-topic"
    assert "capital" in r["text"] and "know nothing about it" in r["text"]
    r = _say(engine, cid, "blorp zingle")
    assert r["meta"]["intent"] == "not-understood"
    assert "unknown to me" in r["text"]


def test_limits_and_language_detection(engine):
    cid = engine.new_conversation("en")["id"]
    before = engine.ltm.count("episodes")
    assert "refused" in _say(engine, cid, "1 2 3 99999999999 5")["text"]
    assert "refused" in _say(engine, cid, " ".join(["1"] * 201))["text"]
    assert engine.ltm.count("episodes") == before
    assert detect_lang("bonjour, que sais-tu ?") == "fr"
    assert detect_lang("hello, what do you know?") == "en"
    assert detect_lang("1 2 3", "fr") == "fr"


def test_explore_and_self_knowledge(engine):
    cid = engine.new_conversation("en")["id"]
    r = _say(engine, cid, "explore 60")
    assert r["meta"]["intent"] == "explore"
    for q, intent in (("what do you know?", "knowledge"), ("stage", "stage"),
                      ("limitations", "limitations"), ("help", "help")):
        assert _say(engine, cid, q)["meta"]["intent"] == intent


def test_conversations_persist_across_sessions(tmp_path):
    db = str(tmp_path / "p.db")
    a = Genesis(db)
    e = ChatEngine(a)
    cid = e.new_conversation("en")["id"]
    e.send(cid, "4 9 14 19 24 29 34", "en")
    a.close()
    a = Genesis(db)
    e = ChatEngine(a)
    assert [c["id"] for c in e.conversations()] == [cid]
    msgs = e.messages(cid)
    assert [m["role"] for m in msgs] == ["user", "genesis"]
    assert e.conversations()[0]["title"].startswith("4 9 14")
    a.close()


# ------------------------------------------------------------------- HTTP API
@pytest.fixture
def server(tmp_path):
    ready, box = threading.Event(), {}

    def run():  # the SQLite connection must live on the serving thread
        agent = Genesis(str(tmp_path / "http.db"))
        srv = HTTPServer(("127.0.0.1", 0), make_handler(ChatEngine(agent)))
        box["srv"], box["url"] = srv, f"http://127.0.0.1:{srv.server_port}"
        ready.set()
        try:
            srv.serve_forever(poll_interval=0.05)
        finally:
            srv.server_close()
            agent.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    ready.wait(10)
    yield box["url"]
    box["srv"].shutdown()
    t.join(10)


def _req(url, data=None, headers=None):
    body = None if data is None else (data if isinstance(data, bytes) else json.dumps(data).encode())
    h = {"Content-Type": "application/json"} if data is not None else {}
    h.update(headers or {})
    try:
        with urllib.request.urlopen(urllib.request.Request(url, body, h), timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_api_round_trip_and_guards(server):
    code, page = _req(server + "/")
    assert code == 200 and b"GENESIS" in page
    code, body = _req(server + "/api/conversations", {"lang": "en"})
    cid = json.loads(body)["id"]
    code, body = _req(f"{server}/api/conversations/{cid}/messages", {"text": "10 20 30 40 50 60 70"})
    assert code == 200 and json.loads(body)["reply"]["meta"]["intent"] == "sequence"
    code, body = _req(f"{server}/api/conversations/{cid}")
    assert len(json.loads(body)["messages"]) == 2
    assert json.loads(_req(server + "/api/status")[1])["knowledge"] >= 1

    url = f"{server}/api/conversations/{cid}/messages"
    assert _req(url, b"text=hi", {"Content-Type": "application/x-www-form-urlencoded"})[0] == 415
    assert _req(url, {"text": "hi"}, {"Origin": "http://evil.example"})[0] == 403
    assert _req(url, {"text": ""})[0] == 400
    assert _req(url, {"text": "x" * 20000})[0] == 413
    assert _req(f"{server}/api/conversations/999/messages", {"text": "hi"})[0] == 404
