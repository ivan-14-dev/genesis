"""First-time setup of an installed GENESIS: ``genesis init`` and ``genesis start``.

``genesis init`` prepares a *home* folder (``--home``, ``$GENESIS_HOME`` or ``~/.genesis``, mode 0700):

    admin.key / admin.pub   Ed25519 admin key pair (the same files as ``genesis admin keygen``; never overwritten)
    memory.db               the agent's memory, born empty (0600)
    credentials.txt         the web user name and password (0600), also printed once by ``genesis init``

In the memory it pins the admin key and installs a policy signed with it, which requires authentication and
whose only user is an administrator with a password generated here (random, ~99 bits). Because the password
is never chosen by a human, the policy stores only its SHA-256, exactly like any other access token, and the
password also works as a bearer token (``/?token=<password>``). The web interface therefore asks for these
credentials from its very first opening. Running ``init`` again changes nothing; ``--reset-password`` signs a
new policy (serial + 1) with a new password.
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Optional

from . import crypto, governance
from .memory import LongTermMemory

ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o, 1/l/i: easy to read back from a terminal
DEFAULT_USER = "admin"
DEFAULT_PORT = 8766


def home_dir(home: Optional[str] = None) -> str:
    return os.path.abspath(home or os.environ.get("GENESIS_HOME") or os.path.join(os.path.expanduser("~"),
                                                                                    ".genesis"))


def paths(home: str) -> dict:
    return {"home": home, "key": os.path.join(home, "admin.key"), "pub": os.path.join(home, "admin.pub"),
            "db": os.path.join(home, "memory.db"), "credentials": os.path.join(home, "credentials.txt")}


def new_password() -> str:
    return "-".join("".join(secrets.choice(ALPHABET) for _ in range(5)) for _ in range(4))


def is_initialised(home: str) -> bool:
    p = paths(home)
    if not (os.path.exists(p["db"]) and os.path.exists(p["pub"])):
        return False
    ltm = LongTermMemory(p["db"], read_only=True)
    try:
        return ltm.get_meta("policy") is not None
    finally:
        ltm.close()


def _admin_seed(p: dict) -> bytes:
    if not os.path.exists(p["key"]):
        crypto.generate_keypair(p["key"], p["pub"])  # refuses if a lone admin.pub exists
    seed = crypto.load_private(p["key"])
    if not os.path.exists(p["pub"]):
        with open(p["pub"], "w") as f:
            f.write(crypto.public_key(seed).hex() + "\n")
    elif crypto.load_public(p["pub"]) != crypto.public_key(seed):
        raise PermissionError(f"{p['pub']} does not match {p['key']}")
    return seed


def _install(ltm: LongTermMemory, seed: bytes, user: str, password: str) -> governance.Policy:
    current = governance.load_policy(ltm)
    entry = {"name": user, "role": "admin", "token_sha256": governance.token_hash(password)}
    if current.signed:  # password reset: keep everything else, replace this user's token
        doc = dict(current.doc)
        old = [u for u in doc["users"] if u["name"] == user]
        entry["role"] = old[0]["role"] if old else "admin"
        doc["users"] = [u for u in doc["users"] if u["name"] != user] + [entry]
        doc["serial"] = current.serial + 1
    else:
        doc = {"serial": 1, "name": "created by genesis init", "require_auth": True, "default_role": None,
               "users": [entry]}
    return governance.install_policy(ltm, governance.sign_policy(seed, doc))


def _write_credentials(path: str, user: str, password: str, port: int) -> None:
    url = f"http://127.0.0.1:{port}/"
    text = (f"GENESIS web interface\n"
            f"url:       {url}\n"
            f"user:      {user}\n"
            f"password:  {password}\n"
            f"created:   {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n\n"
            f"Start it with: genesis start --open\n"
            f"New password:  genesis init --reset-password\n")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _result(p: dict, created: bool, user: Optional[str] = None, password: Optional[str] = None,
            port: int = DEFAULT_PORT, key_id: Optional[str] = None) -> dict:
    out = {"created": created, "home": p["home"], "db": p["db"], "admin_key": p["key"],
           "credentials_file": p["credentials"], "url": f"http://127.0.0.1:{port}/"}
    if password is not None:
        out.update(user=user, password=password, key_id=key_id)
    return out


def initialise(home: Optional[str] = None, user: str = DEFAULT_USER, port: int = DEFAULT_PORT) -> dict:
    """Creates the key, memory, policy and credentials once; afterwards returns ``created: False``."""
    if not governance.NAME_RE.fullmatch(user) or user in ("local", "anonymous"):
        raise ValueError(f"invalid user name {user!r} (lowercase letters, digits, '_', '.', '-')")
    p = paths(home_dir(home))
    os.makedirs(p["home"], mode=0o700, exist_ok=True)
    if is_initialised(p["home"]):
        return _result(p, False, port=port)
    seed = _admin_seed(p)
    from .agent import Genesis
    Genesis(p["db"]).close()  # birth (stage-0 audit of an empty memory) before any admin metadata
    ltm = LongTermMemory(p["db"])
    try:
        key_id = governance.pin_admin_key(ltm, crypto.public_key(seed))
        password = new_password()
        _install(ltm, seed, user, password)
    finally:
        ltm.close()
    _write_credentials(p["credentials"], user, password, port)
    return _result(p, True, user, password, port, key_id)


def reset_password(home: Optional[str] = None, user: str = DEFAULT_USER, port: int = DEFAULT_PORT) -> dict:
    p = paths(home_dir(home))
    if not is_initialised(p["home"]):
        raise FileNotFoundError(f"GENESIS is not initialised in {p['home']} (run: genesis init)")
    seed = crypto.load_private(p["key"])
    ltm = LongTermMemory(p["db"])
    try:
        password = new_password()
        policy = _install(ltm, seed, user, password)
    finally:
        ltm.close()
    _write_credentials(p["credentials"], user, password, port)
    return _result(p, True, user, password, port, policy.key_id)


def describe(res: dict) -> str:
    if not res["created"]:
        return (f"GENESIS is already initialised in {res['home']}.\n"
                f"Web credentials: {res['credentials_file']} (new password: genesis init --reset-password)\n"
                f"Start: genesis start --open  ->  {res['url']}")
    return (f"GENESIS is initialised in {res['home']}\n"
            f"  memory:    {res['db']}\n"
            f"  admin key: {res['admin_key']}  (keep it private: it signs every rule)\n\n"
            f"Web interface credentials (generated; saved in {res['credentials_file']}):\n"
            f"  user:      {res['user']}\n"
            f"  password:  {res['password']}\n\n"
            f"Start: genesis start --open  ->  {res['url']}")
