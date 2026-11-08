"""Admin governance: signed policy, roles, default-deny guard, hash-chained audit and signed approvals.

Three layers decide what GENESIS may learn and do:

1. **Frozen core** — ``innate.SafetyPolicy`` and ``HARD_LIMITS`` below are fixed in code and recorded at
   birth. An admin policy can only *tighten* them.
2. **Admin policy** — a JSON document signed with the admin's Ed25519 key (``crypto.py``). Only the *public*
   key is pinned in the agent's memory, so no code path of the agent can write or alter its own rules; a
   tampered or unsigned replacement makes loading fail. Policies carry an increasing ``serial`` (no rollback).
3. **Guard** — every action is checked against the policy (default deny: unknown actions are refused), and
   decisions on writes and all refusals are appended to the hash-chained ``audit_log``.

The policy decides *who* may make GENESIS learn *what* (domains), within which limits, and which terms are
refused. It cannot make GENESIS believe anything: knowledge still has to pass GENESIS's own validation.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from . import crypto
from .language import fold
from .memory import GENESIS_HASH, LongTermMemory, audit_hash

ACTIONS = {
    "chat.read": "ask questions; read knowledge, explanations and own conversations",
    "sequence.learn": "submit number sequences (stored as user-provided episodes)",
    "semantic.teach": "teach, correct or retract relational facts",
    "lexicon.teach": "teach which command a sentence means",
    "objects.explore": "open the object workshop and run experiments on objects",
    "objects.teach": "label objects and teach constructions",
    "world.explore": "let GENESIS explore sandboxed worlds on its own",
    "lm.generate": "generate text with an approved in-house language model",
    "corpus.submit": "submit a document to the training corpus (pending admin approval)",
    "corpus.approve": "approve or reject corpus documents",
    "lm.train": "train an in-house language model on approved documents",
    "lm.approve": "approve and activate a trained language model",
    "audit.read": "read the audit log",
    "policy.install": "install a new signed policy",
    "version.approve": "deploy a self-improvement version",
}
READ_ACTIONS = ("chat.read", "audit.read")
CORE_DOMAINS = ("chat", "audit", "policy", "version")
DOMAINS = tuple(sorted({a.split(".")[0] for a in ACTIONS} - set(CORE_DOMAINS)))

# Frozen ceilings: an admin may lower these, never raise them.
HARD_LIMITS = {"max_message_chars": 2000, "max_terms_per_message": 200, "max_sequence_terms": 2000,
               "max_explore_steps": 300, "requests_per_minute": 600, "max_seconds_per_message": 120,
               "max_generate_tokens": 400, "max_corpus_doc_chars": 2_000_000}
DEFAULT_LIMITS = {"max_message_chars": 2000, "max_terms_per_message": 200, "max_sequence_terms": 400,
                  "max_explore_steps": 300, "requests_per_minute": 120, "max_seconds_per_message": 30,
                  "max_generate_tokens": 200, "max_corpus_doc_chars": 1_000_000}
DEFAULT_POLICY = {
    "format": 1,
    "serial": 0,
    "name": "built-in default (unsigned)",
    "default_role": "teacher",
    "require_auth": False,
    "roles": {
        "admin": ["*"],
        "teacher": ["chat.read", "sequence.learn", "semantic.teach", "lexicon.teach", "objects.*",
                    "world.explore", "lm.generate", "corpus.submit"],
        "user": ["chat.read", "sequence.learn", "lm.generate"],
        "auditor": ["chat.read", "audit.read"],
    },
    "domains": {d: True for d in DOMAINS},
    "users": [],
    "limits": dict(DEFAULT_LIMITS),
    "blocked_terms": [],
    "log_reads": False,
}
POLICY_KEYS = set(DEFAULT_POLICY)
NAME_RE = re.compile(r"[a-z][a-z0-9_.-]{0,31}")


class PolicyIntegrityError(RuntimeError):
    """The installed policy or admin key does not verify: GENESIS refuses to run under it."""


class PolicyDenied(PermissionError):
    def __init__(self, action: str, reason: str):
        super().__init__(f"{action}: {reason}")
        self.action, self.reason = action, reason


class AuthError(PermissionError):
    pass


def _pattern_ok(p: str) -> bool:
    return p == "*" or p in ACTIONS or (p.endswith(".*") and any(a.startswith(p[:-1]) for a in ACTIONS))


def _matches(pattern: str, action: str) -> bool:
    return pattern == "*" or pattern == action or (pattern.endswith(".*") and action.startswith(pattern[:-1]))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_policy(doc) -> dict:
    """Returns a normalized copy of a policy document, or raises ValueError with the first problem found."""
    if not isinstance(doc, dict):
        raise ValueError("policy must be a JSON object")
    unknown = set(doc) - POLICY_KEYS
    if unknown:
        raise ValueError(f"unknown policy keys: {sorted(unknown)}")
    out = copy.deepcopy(DEFAULT_POLICY)
    out.update(copy.deepcopy(doc))
    if out["format"] != 1:
        raise ValueError("unsupported policy format")
    if not isinstance(out["serial"], int) or isinstance(out["serial"], bool) or out["serial"] < 0:
        raise ValueError("serial must be a non-negative integer")
    if not isinstance(out["name"], str) or len(out["name"]) > 200:
        raise ValueError("name must be a short string")
    roles = out["roles"]
    if not isinstance(roles, dict) or not roles:
        raise ValueError("roles must be a non-empty object")
    for role, patterns in roles.items():
        if not NAME_RE.fullmatch(role):
            raise ValueError(f"invalid role name {role!r}")
        if not isinstance(patterns, list) or not all(isinstance(p, str) and _pattern_ok(p) for p in patterns):
            raise ValueError(f"role {role!r}: permissions must be known actions, 'domain.*' or '*'")
    if out["default_role"] is not None and out["default_role"] not in roles:
        raise ValueError("default_role must be a defined role or null")
    if not isinstance(out["require_auth"], bool) or not isinstance(out["log_reads"], bool):
        raise ValueError("require_auth and log_reads must be booleans")
    domains = out["domains"]
    if not isinstance(domains, dict) or not all(d in DOMAINS and isinstance(v, bool) for d, v in domains.items()):
        raise ValueError(f"domains must map some of {list(DOMAINS)} to true/false")
    out["domains"] = {**{d: True for d in DOMAINS}, **domains}
    users, names, hashes = out["users"], set(), set()
    if not isinstance(users, list):
        raise ValueError("users must be a list")
    for u in users:
        if not isinstance(u, dict) or set(u) != {"name", "role", "token_sha256"}:
            raise ValueError("each user needs exactly name, role and token_sha256")
        if not isinstance(u["name"], str) or not NAME_RE.fullmatch(u["name"]) or u["name"] in names \
                or u["name"] in ("local", "anonymous"):
            raise ValueError(f"invalid or duplicate user name {u.get('name')!r}")
        if u["role"] not in roles:
            raise ValueError(f"user {u['name']!r} has an undefined role")
        if not isinstance(u["token_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", u["token_sha256"]) \
                or u["token_sha256"] in hashes:
            raise ValueError(f"user {u['name']!r}: token_sha256 must be a unique sha256 hex digest")
        names.add(u["name"])
        hashes.add(u["token_sha256"])
    limits = out["limits"]
    if not isinstance(limits, dict):
        raise ValueError("limits must be an object")
    for k, v in limits.items():
        if k not in HARD_LIMITS:
            raise ValueError(f"unknown limit {k!r}")
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"limit {k!r} must be a positive integer")
        if v > HARD_LIMITS[k]:
            raise ValueError(f"limit {k!r}={v} exceeds the frozen ceiling {HARD_LIMITS[k]}")
    out["limits"] = {**DEFAULT_LIMITS, **limits}
    terms = out["blocked_terms"]
    if not isinstance(terms, list) or len(terms) > 1000 or not all(
            isinstance(t, str) and 0 < len(t.strip()) <= 100 for t in terms):
        raise ValueError("blocked_terms must be a list of at most 1000 non-empty short strings")
    out["blocked_terms"] = sorted({fold(t.strip()) for t in terms})
    return out


@dataclass
class Policy:
    doc: dict
    signed: bool = False
    key_id: Optional[str] = None

    @property
    def serial(self) -> int:
        return self.doc["serial"]

    def digest(self) -> str:
        return hashlib.sha256(crypto.canonical(self.doc)).hexdigest()

    def role_allows(self, role: Optional[str], action: str) -> bool:
        return role is not None and any(_matches(p, action) for p in self.doc["roles"].get(role, ()))

    def domain_enabled(self, action: str) -> bool:
        domain = action.split(".")[0]
        return domain in CORE_DOMAINS or self.doc["domains"].get(domain, False)

    def allowed(self, role: Optional[str], action: str) -> bool:
        return action in ACTIONS and self.role_allows(role, action) and self.domain_enabled(action)

    def limit(self, name: str) -> int:
        return self.doc["limits"][name]

    def blocked(self, text: str) -> Optional[str]:
        folded = fold(text)
        for term in self.doc["blocked_terms"]:
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", folded):
                return term
        return None

    def identify(self, token: Optional[str]) -> tuple:
        """Maps a bearer token to (user name, role). No token: the local default role, unless auth is required."""
        if not token:
            if self.doc["require_auth"] or self.doc["default_role"] is None:
                raise AuthError("authentication required")
            return "local", self.doc["default_role"]
        h = token_hash(token)
        for u in self.doc["users"]:
            if hmac.compare_digest(u["token_sha256"], h):
                return u["name"], u["role"]
        raise AuthError("unknown token")

    def summary(self) -> dict:
        return {"name": self.doc["name"], "serial": self.serial, "signed": self.signed, "key_id": self.key_id,
                "digest": self.digest(), "default_role": self.doc["default_role"],
                "require_auth": self.doc["require_auth"], "roles": self.doc["roles"],
                "domains": self.doc["domains"], "limits": self.doc["limits"],
                "users": [{"name": u["name"], "role": u["role"]} for u in self.doc["users"]],
                "blocked_terms": len(self.doc["blocked_terms"])}


# ------------------------------------------------------------- admin key + policy
def pinned_key(ltm: LongTermMemory) -> Optional[bytes]:
    h = ltm.get_meta("admin_pubkey")
    return bytes.fromhex(h) if h else None


def memory_id(ltm: LongTermMemory) -> str:
    """Random identity of this memory, bound into every signed statement so it cannot be replayed elsewhere."""
    mid = ltm.get_meta("memory_id")
    if mid is None:
        mid = secrets.token_hex(16)
        ltm.set_meta("memory_id", mid)
    return mid


def pin_admin_key(ltm: LongTermMemory, pub: bytes) -> str:
    current = pinned_key(ltm)
    if current is not None and current != pub:
        raise PermissionError("another admin key is already pinned in this memory")
    ltm.set_meta("admin_pubkey", pub.hex())
    memory_id(ltm)
    ltm.append_audit("admin.pin-key", "allow", "admin", "admin", crypto.key_id(pub))
    ltm.commit()
    return crypto.key_id(pub)


def _require_admin(ltm: LongTermMemory, seed: bytes) -> bytes:
    pub, pinned = crypto.public_key(seed), pinned_key(ltm)
    if pinned is None:
        raise PermissionError("no admin key pinned in this memory (run: genesis admin pin)")
    if pub != pinned:
        raise PermissionError("this key is not the admin key pinned in this memory")
    return pub


def sign_policy(seed: bytes, doc: dict) -> dict:
    return crypto.sign_statement(seed, "policy", validate_policy(doc))


def load_policy(ltm: LongTermMemory, expected_pub: Optional[bytes] = None) -> Policy:
    signed = ltm.get_meta("policy")
    pub = pinned_key(ltm)
    if expected_pub is not None and pub != expected_pub:
        raise PolicyIntegrityError("the admin key pinned in memory is not the expected one")
    if signed is None:
        return Policy(validate_policy({}), signed=False)
    if pub is None:
        raise PolicyIntegrityError("a policy is installed but no admin key is pinned")
    if not crypto.verify_statement(pub, signed, "policy"):
        raise PolicyIntegrityError("installed policy signature does not verify")
    try:
        return Policy(validate_policy(signed["statement"]), signed=True, key_id=signed["key_id"])
    except ValueError as exc:
        raise PolicyIntegrityError(f"installed policy is invalid: {exc}") from exc


def install_policy(ltm: LongTermMemory, signed: dict) -> Policy:
    pub = pinned_key(ltm)
    if pub is None:
        raise PermissionError("pin the admin public key first (genesis admin pin)")
    if not crypto.verify_statement(pub, signed, "policy"):
        raise PermissionError("policy signature does not verify with the pinned admin key")
    new = Policy(validate_policy(signed["statement"]), signed=True, key_id=signed["key_id"])
    current = load_policy(ltm)
    if current.signed and new.serial <= current.serial:
        raise PermissionError(f"policy serial must increase (installed: {current.serial}, new: {new.serial})")
    history = ltm.get_meta("policy_history") or []
    history.append({"serial": new.serial, "digest": new.digest(), "key_id": new.key_id, "ts": time.time()})
    ltm.set_meta("policy_history", history)
    ltm.set_meta("policy", signed)
    ltm.append_audit("policy.install", "allow", f"admin:{new.key_id}", "admin", new.serial,
                     {"digest": new.digest(), "name": new.doc["name"]})
    ltm.commit()
    return new


def new_token() -> tuple:
    """A fresh bearer token and the hash to put in the policy's ``users`` list (only the hash is stored)."""
    token = secrets.token_urlsafe(32)
    return token, token_hash(token)


# ------------------------------------------------------------------------ guard
class Guard:
    """Binds a policy to one identity. ``require`` raises ``PolicyDenied`` *before* any write happens."""

    def __init__(self, policy: Policy, ltm: Optional[LongTermMemory], actor: str = "local",
                 role: Optional[str] = None):
        self.policy, self.ltm, self.actor = policy, ltm, actor
        self.role = policy.doc["default_role"] if role is None else role

    def allowed(self, action: str) -> bool:
        return self.policy.allowed(self.role, action)

    def _log(self, action, decision, target=None, detail=None) -> None:
        if self.ltm is not None and not self.ltm.read_only:
            self.ltm.append_audit(action, decision, self.actor, self.role, target, detail)

    def require(self, action: str, target=None, detail: Optional[dict] = None) -> None:
        if not self.allowed(action):
            if action not in ACTIONS:
                reason = "unknown action (default deny)"
            elif not self.policy.domain_enabled(action):
                reason = "domain disabled by the admin"
            else:
                reason = f"role '{self.role}' is not permitted"
            self._log(action, "deny", target, {"reason": reason, **(detail or {})})
            raise PolicyDenied(action, reason)
        if action not in READ_ACTIONS or self.policy.doc["log_reads"]:
            self._log(action, "allow", target, detail)

    def check_text(self, text: str) -> None:
        if len(text) > self.limit("max_message_chars"):
            self._log("chat.read", "deny", None, {"reason": "message too long", "chars": len(text)})
            raise PolicyDenied("chat.read", "message too long")
        term = self.policy.blocked(text)
        if term:
            self._log("chat.read", "deny", None, {"reason": "blocked term", "term": term})
            raise PolicyDenied("chat.read", "blocked term")

    def limit(self, name: str) -> int:
        return self.policy.limit(name)


# --------------------------------------------------------------------- audit
def verify_audit(ltm: LongTermMemory, pub: Optional[bytes] = None) -> dict:
    """Recomputes the hash chain and checks every admin seal. Any edited, inserted, reordered or deleted entry
    breaks the chain; deleting the most recent entries is caught by the latest seal."""
    pub = pub or pinned_key(ltm)
    prev, expected, first_bad, hashes = GENESIS_HASH, 1, None, {0: GENESIS_HASH}
    for row in ltm.audit_entries():
        if row["id"] != expected or row["prev_hash"] != prev or audit_hash(row) != row["hash"]:
            first_bad = row["id"]
            break
        hashes[row["id"]] = prev = row["hash"]
        expected += 1
    bad_seals, n_seals = [], 0
    mid = ltm.get_meta("memory_id")
    for seal in ltm.db.execute("SELECT * FROM audit_seals ORDER BY id"):
        n_seals += 1
        signed = json.loads(seal["signed"])
        st = signed.get("statement") or {}
        ok = (pub is not None and crypto.verify_statement(pub, signed, "audit-seal")
              and st.get("memory_id") == mid and st.get("entry_id") == seal["entry_id"]
              and hashes.get(seal["entry_id"]) == st.get("head_hash") == seal["head_hash"])
        if not ok:
            bad_seals.append(seal["id"])
    return {"ok": first_bad is None and not bad_seals, "entries": len(hashes), "first_bad_entry": first_bad,
            "seals": n_seals, "bad_seals": bad_seals, "head": ltm.audit_head()[1] if first_bad is None else None}


def seal_audit(ltm: LongTermMemory, seed: bytes) -> dict:
    _require_admin(ltm, seed)
    report = verify_audit(ltm)
    if not report["ok"]:
        raise PolicyIntegrityError(f"refusing to seal a broken audit log: {report}")
    entry_id, head = ltm.audit_head()
    signed = crypto.sign_statement(seed, "audit-seal", {"entry_id": entry_id, "head_hash": head,
                                                        "memory_id": memory_id(ltm), "ts": time.time()})
    ltm.db.execute("INSERT INTO audit_seals(ts, entry_id, head_hash, signed) VALUES (?,?,?,?)",
                   (time.time(), entry_id, head, json.dumps(signed)))
    ltm.commit()
    return {"entry_id": entry_id, "head_hash": head, "key_id": signed["key_id"]}


# ------------------------------------------------------------- signed approvals
def admin_approval(ltm: LongTermMemory, seed: bytes, purpose: str, target: dict, by: str = "") -> dict:
    """An admin decision signed with the private key, bound to this memory and logged in the audit chain."""
    _require_admin(ltm, seed)
    statement = {**target, "by": by, "memory_id": memory_id(ltm), "ts": time.time()}
    signed = crypto.sign_statement(seed, purpose, statement)
    ltm.append_audit(purpose, "allow", f"admin:{signed['key_id']}" + (f" ({by})" if by else ""), "admin",
                     json.dumps(target, sort_keys=True), statement)
    return signed


def check_approval(ltm: LongTermMemory, signed, purpose: str, expected: dict) -> bool:
    pub = pinned_key(ltm)
    if pub is None or not crypto.verify_statement(pub, signed, purpose):
        return False
    st = signed["statement"]
    return st.get("memory_id") == ltm.get_meta("memory_id") and all(st.get(k) == v for k, v in expected.items())
