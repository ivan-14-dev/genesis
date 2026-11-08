"""Ed25519 signatures (RFC 8032), written here from the specification, plus admin key files.

The admin signs policies, approvals and audit seals with a private key that stays outside the agent's
memory; the agent and its servers only hold the *public* key, so no code path of the agent can forge a rule
or an approval. Verification follows RFC 8032 §5.1.7 (cofactorless check, canonical S, strict decoding).

This implementation is correct but not side-channel hardened (big-integer arithmetic is not constant
time): signing is meant to run in the admin's own CLI process, never in a network-facing server.
"""
from __future__ import annotations

import hashlib
import json
import os

P = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493
D = -121665 * pow(121666, P - 2, P) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)


def _inv(x: int) -> int:
    return pow(x, P - 2, P)


def _recover_x(y: int, sign: int):
    if y >= P:
        return None
    u, v = (y * y - 1) % P, (D * y * y + 1) % P
    x = u * pow(v, 3, P) * pow(u * pow(v, 7, P), (P - 5) // 8, P) % P
    vx2 = v * x * x % P
    if vx2 == (-u) % P:
        x = x * SQRT_M1 % P
    elif vx2 != u:
        return None
    if x == 0 and sign:
        return None
    return P - x if x & 1 != sign else x


_BY = 4 * _inv(5) % P
_BX = _recover_x(_BY, 0)
BASE = (_BX, _BY, 1, _BX * _BY % P)
IDENTITY = (0, 1, 1, 0)


def _add(p1, p2):
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = 2 * D * t1 * t2 % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f % P, g * h % P, f * g % P, e * h % P


def _double(p1):
    x1, y1, z1, _ = p1
    a, b, c = x1 * x1 % P, y1 * y1 % P, 2 * z1 * z1 % P
    h = a + b
    e = h - (x1 + y1) * (x1 + y1)
    g = a - b
    f = c + g
    return e * f % P, g * h % P, f * g % P, e * h % P


def _mul(k: int, point):
    acc = IDENTITY
    for bit in bin(k)[2:]:
        acc = _double(acc)
        if bit == "1":
            acc = _add(acc, point)
    return acc


def _equal(p1, p2) -> bool:
    return (p1[0] * p2[2] - p2[0] * p1[2]) % P == 0 and (p1[1] * p2[2] - p2[1] * p1[2]) % P == 0


def _encode(point) -> bytes:
    x, y, z, _ = point
    zi = _inv(z)
    x, y = x * zi % P, y * zi % P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decode(data: bytes):
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % P)


def _h(*parts: bytes) -> int:
    return int.from_bytes(hashlib.sha512(b"".join(parts)).digest(), "little")


def _expand(seed: bytes):
    if len(seed) != 32:
        raise ValueError("Ed25519 private seed must be 32 bytes")
    h = hashlib.sha512(seed).digest()
    s = int.from_bytes(h[:32], "little")
    s &= (1 << 254) - 8
    s |= 1 << 254
    return s, h[32:]


def public_key(seed: bytes) -> bytes:
    return _encode(_mul(_expand(seed)[0], BASE))


def sign(seed: bytes, message: bytes) -> bytes:
    s, prefix = _expand(seed)
    pub = _encode(_mul(s, BASE))
    r = _h(prefix, message) % L
    big_r = _encode(_mul(r, BASE))
    k = _h(big_r, pub, message) % L
    return big_r + int.to_bytes((r + k * s) % L, 32, "little")


def verify(pub: bytes, message: bytes, signature: bytes) -> bool:
    if len(pub) != 32 or len(signature) != 64:
        return False
    a, r = _decode(pub), _decode(signature[:32])
    s = int.from_bytes(signature[32:], "little")
    if a is None or r is None or s >= L:
        return False
    k = _h(signature[:32], pub, message) % L
    return _equal(_mul(s, BASE), _add(r, _mul(k, a)))


# ------------------------------------------------------------ signed statements
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def key_id(pub: bytes) -> str:
    return hashlib.sha256(pub).hexdigest()[:16]


def _message(purpose: str, statement) -> bytes:
    # Domain separation: a signature for one purpose can never be replayed as another.
    return b"genesis-v1/" + purpose.encode() + b"\n" + canonical(statement)


def sign_statement(seed: bytes, purpose: str, statement) -> dict:
    pub = public_key(seed)
    return {"purpose": purpose, "statement": statement, "key_id": key_id(pub),
            "signature": sign(seed, _message(purpose, statement)).hex()}


def verify_statement(pub: bytes, signed, purpose: str) -> bool:
    if not isinstance(signed, dict) or signed.get("purpose") != purpose:
        return False
    try:
        sig = bytes.fromhex(signed.get("signature") or "")
    except ValueError:
        return False
    return signed.get("key_id") == key_id(pub) and verify(pub, _message(purpose, signed.get("statement")), sig)


# -------------------------------------------------------------------- key files
def _write_private(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(data)


def generate_keypair(private_path: str, public_path: str) -> str:
    """Creates a new admin key pair; refuses to overwrite existing files. Returns the key id."""
    if os.path.exists(private_path) or os.path.exists(public_path):
        raise FileExistsError("refusing to overwrite an existing key file")
    seed = os.urandom(32)
    pub = public_key(seed)
    _write_private(private_path, seed.hex() + "\n")
    with open(public_path, "w") as f:
        f.write(pub.hex() + "\n")
    return key_id(pub)


def load_private(path: str) -> bytes:
    mode = os.stat(path).st_mode
    if os.name == "posix" and mode & 0o077:
        raise PermissionError(f"private key {path} is readable by others; run: chmod 600 {path}")
    with open(path) as f:
        seed = bytes.fromhex(f.read().strip())
    _expand(seed)
    return seed


def load_public(path_or_hex: str) -> bytes:
    text = path_or_hex
    if os.path.exists(path_or_hex):
        with open(path_or_hex) as f:
            text = f.read()
    pub = bytes.fromhex(text.strip())
    if _decode(pub) is None:
        raise ValueError("invalid Ed25519 public key")
    return pub
