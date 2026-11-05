"""Memory systems: limited working memory and persistent long-term memory.

Long-term memory is a single SQLite file with five stores:

* episodic      ``episodes``      every experience, never deleted by consolidation
* semantic      ``knowledge``     acquired rules / concepts with status + confidence
* procedural    ``procedures``    learned strategy priors (which model shapes work)
* associative   ``associations``  weighted links between concepts (correlational/causal)
* metacognitive ``metacog``       learning performance records

``evidence`` links every knowledge item to the episodes that support or
contradict it; ``knowledge_history`` audits every status change.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass
from typing import Optional

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session INTEGER NOT NULL, env_id TEXT NOT NULL,
    context TEXT NOT NULL, provenance TEXT NOT NULL, action TEXT, observation TEXT, outcome TEXT,
    predictions TEXT, surprise REAL, reward REAL, salience REAL, archived INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_episodes_ctx ON episodes(context, id);
CREATE TABLE IF NOT EXISTS knowledge (
    id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, kind TEXT NOT NULL, statement TEXT NOT NULL,
    expr TEXT, confidence REAL NOT NULL, status TEXT NOT NULL, provenance TEXT NOT NULL, context TEXT,
    created_ts REAL NOT NULL, updated_ts REAL NOT NULL, n_support INTEGER DEFAULT 0, n_contra INTEGER DEFAULT 0,
    forward_ok INTEGER DEFAULT 0, forward_bad INTEGER DEFAULT 0, UNIQUE(key, statement));
CREATE TABLE IF NOT EXISTS knowledge_history (
    knowledge_id INTEGER NOT NULL, ts REAL NOT NULL, old_status TEXT, new_status TEXT NOT NULL, reason TEXT);
CREATE TABLE IF NOT EXISTS evidence (
    knowledge_id INTEGER NOT NULL, episode_id INTEGER NOT NULL, relation TEXT NOT NULL,
    PRIMARY KEY (knowledge_id, episode_id, relation));
CREATE TABLE IF NOT EXISTS procedures (
    shape TEXT PRIMARY KEY, successes INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
    updated_ts REAL);
CREATE TABLE IF NOT EXISTS associations (
    a TEXT NOT NULL, b TEXT NOT NULL, kind TEXT NOT NULL, weight REAL NOT NULL, n INTEGER NOT NULL,
    detail TEXT, updated_ts REAL, PRIMARY KEY (a, b));
CREATE TABLE IF NOT EXISTS metacog (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, session INTEGER, metric TEXT NOT NULL,
    value REAL, context TEXT);
CREATE TABLE IF NOT EXISTS learner_state (key TEXT PRIMARY KEY, env_id TEXT, state TEXT NOT NULL);
"""


@dataclass
class KnowledgeItem:
    id: int
    key: str
    kind: str
    statement: str
    expr: Optional[list]
    confidence: float
    status: str
    provenance: str
    context: Optional[str]
    n_support: int
    n_contra: int
    forward_ok: int
    forward_bad: int


class LongTermMemory:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    # -------------------------------------------------------------------- meta
    def get_meta(self, key: str):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def set_meta(self, key: str, value) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, json.dumps(value)))

    # ---------------------------------------------------------------- episodic
    def record_episode(self, session: int, env_id: str, context: str, provenance: str, action=None,
                       observation=None, outcome=None, predictions=None, surprise=None, reward=None,
                       salience=None) -> int:
        cur = self.db.execute(
            "INSERT INTO episodes(ts, session, env_id, context, provenance, action, observation, outcome, "
            "predictions, surprise, reward, salience) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), session, env_id, context, provenance, json.dumps(action), json.dumps(observation),
             json.dumps(outcome), json.dumps(predictions), surprise, reward, salience))
        return cur.lastrowid

    def recent_episodes(self, context: str, n: int) -> list:
        rows = self.db.execute("SELECT * FROM episodes WHERE context=? ORDER BY id DESC LIMIT ?",
                               (context, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def episode(self, episode_id: int) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        return dict(row) if row else None

    def count(self, table: str) -> int:
        assert table in ("episodes", "knowledge", "procedures", "associations", "metacog", "evidence",
                         "knowledge_history", "learner_state")
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---------------------------------------------------------------- semantic
    def upsert_knowledge(self, key: str, kind: str, statement: str, expr, confidence: float, status: str,
                         provenance: str, context: Optional[str] = None, n_support: int = 0, n_contra: int = 0,
                         forward_ok: int = 0, forward_bad: int = 0, reason: str = "") -> int:
        now = time.time()
        row = self.db.execute("SELECT id, status FROM knowledge WHERE key=? AND statement=?",
                              (key, statement)).fetchone()
        if row is None:
            cur = self.db.execute(
                "INSERT INTO knowledge(key, kind, statement, expr, confidence, status, provenance, context, "
                "created_ts, updated_ts, n_support, n_contra, forward_ok, forward_bad) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, kind, statement, json.dumps(expr), confidence, status, provenance, context, now, now,
                 n_support, n_contra, forward_ok, forward_bad))
            kid = cur.lastrowid
            self.db.execute("INSERT INTO knowledge_history VALUES (?,?,?,?,?)", (kid, now, None, status, reason))
            return kid
        kid = row["id"]
        self.db.execute(
            "UPDATE knowledge SET confidence=?, status=?, provenance=?, context=?, updated_ts=?, n_support=?, "
            "n_contra=?, forward_ok=?, forward_bad=? WHERE id=?",
            (confidence, status, provenance, context, now, n_support, n_contra, forward_ok, forward_bad, kid))
        if row["status"] != status:
            self.db.execute("INSERT INTO knowledge_history VALUES (?,?,?,?,?)",
                            (kid, now, row["status"], status, reason))
        return kid

    def set_status(self, kid: int, status: str, reason: str) -> None:
        row = self.db.execute("SELECT status FROM knowledge WHERE id=?", (kid,)).fetchone()
        if row and row["status"] != status:
            self.db.execute("UPDATE knowledge SET status=?, updated_ts=? WHERE id=?", (status, time.time(), kid))
            self.db.execute("INSERT INTO knowledge_history VALUES (?,?,?,?,?)",
                            (kid, time.time(), row["status"], status, reason))

    def knowledge(self, key: Optional[str] = None, status: Optional[str] = None, kind: Optional[str] = None) -> list:
        q, args = "SELECT * FROM knowledge WHERE 1=1", []
        for col, val in (("key", key), ("status", status), ("kind", kind)):
            if val is not None:
                q += f" AND {col}=?"
                args.append(val)
        rows = self.db.execute(q + " ORDER BY id", args).fetchall()
        return [KnowledgeItem(r["id"], r["key"], r["kind"], r["statement"], json.loads(r["expr"]) if r["expr"]
                              else None, r["confidence"], r["status"], r["provenance"], r["context"],
                              r["n_support"], r["n_contra"], r["forward_ok"], r["forward_bad"]) for r in rows]

    def add_evidence(self, kid: int, episode_ids: list, relation: str) -> None:
        self.db.executemany("INSERT OR IGNORE INTO evidence VALUES (?,?,?)",
                            [(kid, e, relation) for e in episode_ids if e is not None])

    def explain(self, kid: int, limit: int = 10) -> dict:
        """Why is this knowledge held? Returns the item, its history and evidence episodes."""
        item = self.db.execute("SELECT * FROM knowledge WHERE id=?", (kid,)).fetchone()
        if item is None:
            return {}
        hist = self.db.execute("SELECT * FROM knowledge_history WHERE knowledge_id=? ORDER BY ts", (kid,)).fetchall()
        out = {"item": dict(item), "history": [dict(h) for h in hist], "evidence": {}}
        for rel in ("induced_from", "supports", "contradicts", "refuted_by"):
            rows = self.db.execute(
                "SELECT e.* FROM evidence v JOIN episodes e ON e.id=v.episode_id WHERE v.knowledge_id=? "
                "AND v.relation=? ORDER BY e.id LIMIT ?", (kid, rel, limit)).fetchall()
            total = self.db.execute("SELECT COUNT(*) FROM evidence WHERE knowledge_id=? AND relation=?",
                                    (kid, rel)).fetchone()[0]
            out["evidence"][rel] = {"total": total, "examples": [dict(r) for r in rows]}
        return out

    # -------------------------------------------------------------- procedural
    def shape_successes(self) -> dict:
        return {r["shape"]: r["successes"] for r in self.db.execute("SELECT * FROM procedures")}

    def record_shape(self, shape: str, success: bool) -> None:
        self.db.execute(
            "INSERT INTO procedures(shape, successes, attempts, updated_ts) VALUES (?,?,1,?) "
            "ON CONFLICT(shape) DO UPDATE SET successes=successes+?, attempts=attempts+1, updated_ts=?",
            (shape, int(success), time.time(), int(success), time.time()))

    # ------------------------------------------------------------- associative
    def upsert_association(self, a: str, b: str, kind: str, weight: float, n: int, detail=None) -> None:
        self.db.execute("INSERT OR REPLACE INTO associations VALUES (?,?,?,?,?,?,?)",
                        (a, b, kind, weight, n, json.dumps(detail), time.time()))

    def associations(self, kind: Optional[str] = None) -> list:
        q = "SELECT * FROM associations" + (" WHERE kind=?" if kind else "") + " ORDER BY a, b"
        return [dict(r) for r in self.db.execute(q, (kind,) if kind else ())]

    # ----------------------------------------------------------- metacognitive
    def record_metric(self, session: int, metric: str, value: float, context=None) -> None:
        self.db.execute("INSERT INTO metacog(ts, session, metric, value, context) VALUES (?,?,?,?,?)",
                        (time.time(), session, metric, value, json.dumps(context)))

    def metrics(self, metric: Optional[str] = None) -> list:
        q = "SELECT * FROM metacog" + (" WHERE metric=?" if metric else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (metric,) if metric else ())]

    # --------------------------------------------------------- learner states
    def save_learner(self, key: str, env_id: str, state: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO learner_state VALUES (?,?,?)", (key, env_id, json.dumps(state)))

    def load_learners(self) -> list:
        return [(r["key"], r["env_id"], json.loads(r["state"])) for r in self.db.execute("SELECT * FROM learner_state")]

    # ------------------------------------------------------ checkpoint/rollback
    def checkpoint(self, dest: str) -> str:
        self.db.commit()
        target = sqlite3.connect(dest)
        with target:
            self.db.backup(target)
        target.close()
        return dest

    @staticmethod
    def restore(checkpoint_path: str, live_path: str) -> None:
        shutil.copyfile(checkpoint_path, live_path)

    def size_bytes(self) -> int:
        self.db.commit()
        return os.path.getsize(self.path) if os.path.exists(self.path) else 0


@dataclass
class WMItem:
    key: tuple
    content: object
    salience: float
    ts: int


class WorkingMemory:
    """Capacity-limited buffer. Lowest (salience x recency) item is evicted first."""

    def __init__(self, capacity: int = 9, decay: float = 0.9):
        self.capacity = capacity
        self.decay = decay
        self.items: dict = {}
        self.clock = 0
        self.evictions = 0

    def put(self, key: tuple, content, salience: float = 1.0) -> list:
        self.clock += 1
        self.items[key] = WMItem(key, content, salience, self.clock)
        evicted = []
        while len(self.items) > self.capacity:
            victim = min(self.items.values(), key=self._strength)
            del self.items[victim.key]
            evicted.append(victim)
            self.evictions += 1
        return evicted

    def _strength(self, it: WMItem) -> float:
        return it.salience * (self.decay ** (self.clock - it.ts))

    def get(self, key: tuple):
        it = self.items.get(key)
        return None if it is None else it.content

    def touch(self, key: tuple) -> None:
        if key in self.items:
            self.clock += 1
            self.items[key].ts = self.clock

    def keys(self) -> list:
        return list(self.items)
