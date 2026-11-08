"""Training corpus registry: what the in-house language model may learn from, decided by the admin.

Nothing is downloaded and nothing is pretrained: documents are submitted locally, then go through four gates
before any model can read them:

1. **Policy**: the submitter needs ``corpus.submit``; the document must fit ``max_corpus_doc_chars`` and contain
   no admin-blocked term.
2. **Hygiene**: Unicode NFC, control characters stripped, personal data redacted (e-mails, phone numbers,
   IBANs, card numbers passing the Luhn check, IP addresses), exact duplicates refused (SHA-256).
3. **Split**: each document goes to ``train`` / ``valid`` / ``test`` from its hash (deterministic, stable when
   other documents are added). ``test`` is sealed: only evaluation reads it, and every read is audited.
4. **Approval**: only documents carrying an admin signature over their exact hash are ever given to training;
   flipping a status in the database is not enough.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from typing import Optional

from . import governance
from .memory import LongTermMemory

SPLITS = (("train", 80), ("valid", 90), ("test", 100))
MAX_NAME = 200
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PII = (
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    ("CARD", re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+\d{1,3}[ .-]?\d|0\d)(?:[ .-]?\d{2}){4}(?!\w)"
                         r"|(?<![\w(])\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)")),
    ("IP", re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])")),
)


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def redact(text: str) -> tuple:
    """Replaces personal data with typed placeholders; returns (text, {kind: count})."""
    counts = {}
    for kind, rx in PII:
        def sub(m, kind=kind):
            if kind == "CARD" and not _luhn(re.sub(r"\D", "", m.group(0))):
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return f"<{kind}>"
        text = rx.sub(sub, text)
    return text, counts


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return CONTROL.sub("", text).strip()


def split_of(sha256: str) -> str:
    bucket = int(sha256[:8], 16) % 100
    return next(name for name, upper in SPLITS if bucket < upper)


def submit(ltm: LongTermMemory, guard: governance.Guard, name: str, text: str, source: str = "local",
           license: str = "unspecified") -> dict:
    if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME:
        raise ValueError(f"a document needs a name of at most {MAX_NAME} characters")
    if not isinstance(text, str):
        raise ValueError("document text must be a string")
    text = normalize(text)
    if not text:
        raise ValueError("empty document")
    limit = guard.limit("max_corpus_doc_chars")
    if len(text) > limit:
        raise ValueError(f"document longer than {limit} characters (admin limit)")
    term = guard.policy.blocked(text)
    if term:
        guard.require("corpus.submit", name, {"refused": "blocked term"})
        raise governance.PolicyDenied("corpus.submit", "blocked term")
    text, redactions = redact(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if ltm.db.execute("SELECT 1 FROM corpus_docs WHERE sha256=?", (digest,)).fetchone():
        raise ValueError("this document is already in the corpus")
    split = split_of(digest)
    guard.require("corpus.submit", name, {"sha256": digest, "chars": len(text), "split": split,
                                          "redactions": redactions})
    now = time.time()
    did = ltm.db.execute(
        "INSERT INTO corpus_docs(ts, name, source, license, sha256, text, n_chars, submitted_by, status, split, "
        "redactions, approval, updated_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now, name.strip(), str(source)[:200], str(license)[:100], digest, text, len(text), guard.actor, "pending",
         split, json.dumps(redactions), None, now)).lastrowid
    ltm.commit()
    return {"id": did, "sha256": digest, "split": split, "chars": len(text), "redactions": redactions,
            "status": "pending"}


def review(ltm: LongTermMemory, seed: bytes, doc_id: int, approve: bool, by: str = "") -> dict:
    """Admin decision on a document, signed over its exact content hash."""
    row = ltm.db.execute("SELECT id, sha256, status FROM corpus_docs WHERE id=?", (doc_id,)).fetchone()
    if row is None:
        raise KeyError(doc_id)
    decision = "approved" if approve else "rejected"
    signed = governance.admin_approval(ltm, seed, "corpus-approval",
                                       {"doc_id": doc_id, "sha256": row["sha256"], "decision": decision}, by)
    ltm.db.execute("UPDATE corpus_docs SET status=?, approval=?, updated_ts=? WHERE id=?",
                   (decision, json.dumps(signed), time.time(), doc_id))
    ltm.commit()
    return {"id": doc_id, "status": decision, "sha256": row["sha256"]}


def documents(ltm: LongTermMemory, status: Optional[str] = None) -> list:
    q = "SELECT id, ts, name, source, license, sha256, n_chars, submitted_by, status, split, redactions FROM corpus_docs"
    rows = ltm.db.execute(q + (" WHERE status=?" if status else "") + " ORDER BY id", (status,) if status else ())
    return [{**dict(r), "redactions": json.loads(r["redactions"] or "{}")} for r in rows]


def _verified(ltm: LongTermMemory, row) -> bool:
    if row["status"] != "approved" or not row["approval"]:
        return False
    if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["sha256"]:
        return False
    return governance.check_approval(ltm, json.loads(row["approval"]), "corpus-approval",
                                     {"doc_id": row["id"], "sha256": row["sha256"], "decision": "approved"})


def approved_texts(ltm: LongTermMemory, split: str, reader: str = "system") -> list:
    """(id, sha256, text) of the admin-approved documents of one split whose signature and hash still verify.
    Reading ``test`` is logged: it must only serve final evaluation, never training or tuning."""
    if split not in dict(SPLITS):
        raise ValueError(f"unknown split {split!r}")
    rows = ltm.db.execute("SELECT * FROM corpus_docs WHERE split=? AND status='approved' ORDER BY id", (split,))
    out = [(r["id"], r["sha256"], r["text"]) for r in rows if _verified(ltm, r)]
    if split == "test":
        ltm.append_audit("corpus.read-test", "allow", reader, None, None, {"docs": [d for d, _, _ in out]})
        ltm.commit()
    return out


def verify(ltm: LongTermMemory) -> dict:
    """Approved documents whose content or signature no longer match (tampering)."""
    bad = [r["id"] for r in ltm.db.execute("SELECT * FROM corpus_docs WHERE status='approved'")
           if not _verified(ltm, r)]
    return {"ok": not bad, "tampered_documents": bad}
