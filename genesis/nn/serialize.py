"""Safe, deterministic weight serialisation for :mod:`genesis.nn` models (no pickle).

Layout::

    b"GNNW1\\n" | uint32 LE header length | canonical JSON header | float64 LE payload

The header (``sort_keys=True``, compact separators) holds ``format_version``,
the model ``class``, its ``config`` and the ordered ``params`` as
``[name, shape]`` pairs; the payload is the parameters' values concatenated in
that order. Loading only ever parses JSON and raw floats: the class must be in a
fixed whitelist, the header must match exactly what that class would build from
the config (checked before allocating the model), the payload size must be
exact and all values finite; anything else raises ``ValueError``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from array import array

from .models import MLPClassifier, TinyGPT

MAGIC = b"GNNW1\n"
FORMAT_VERSION = 1
MAX_HEADER = 1 << 20
_CLASSES = {"MLPClassifier": MLPClassifier, "TinyGPT": TinyGPT}


def _header_bytes(header: dict) -> bytes:
    return json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def dumps(model) -> bytes:
    name = type(model).__name__
    if _CLASSES.get(name) is not type(model):
        raise ValueError(f"cannot serialise {name}: not a whitelisted model class")
    named = model.named_parameters()
    header = {"format_version": FORMAT_VERSION, "class": name, "config": dict(model.config),
              "params": [[n, list(p.shape)] for n, p in named]}
    hb = _header_bytes(header)
    parts = [MAGIC, struct.pack("<I", len(hb)), hb]
    for _, p in named:
        a = array("d", p.data)
        if sys.byteorder == "big":
            a.byteswap()
        parts.append(a.tobytes())
    return b"".join(parts)


def loads(data: bytes):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("weights must be bytes")
    data = bytes(data)
    base = len(MAGIC) + 4
    if len(data) < base or not data.startswith(MAGIC):
        raise ValueError("not a GENESIS nn weight file (bad magic)")
    (hlen,) = struct.unpack_from("<I", data, len(MAGIC))
    if not 2 <= hlen <= MAX_HEADER or base + hlen > len(data):
        raise ValueError(f"bad header length {hlen}")
    try:
        header = json.loads(data[base:base + hlen].decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"malformed header: {exc}") from None
    if not isinstance(header, dict) or set(header) != {"format_version", "class", "config", "params"}:
        raise ValueError("header must have exactly format_version, class, config, params")
    if header["format_version"] != FORMAT_VERSION or type(header["format_version"]) is not int:
        raise ValueError(f"unsupported format_version {header['format_version']!r}")
    cls = _CLASSES.get(header["class"]) if isinstance(header["class"], str) else None
    if cls is None:
        raise ValueError(f"unknown or disallowed model class {header['class']!r}")
    config = header["config"]
    if not isinstance(config, dict) or not all(isinstance(k, str) and type(v) is int for k, v in config.items()):
        raise ValueError("config must map names to ints")
    try:
        specs = cls.param_specs(**config)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid config: {exc}") from None
    expected = [[n, list(shape)] for n, shape, _ in specs]
    if header["params"] != expected:
        raise ValueError("parameter names/shapes do not match the model config")
    sizes = [math.prod(shape) for _, shape, _ in specs]
    if len(data) - base - hlen != 8 * sum(sizes):
        raise ValueError(f"payload is {len(data) - base - hlen} bytes, expected {8 * sum(sizes)}")
    model = cls(**config)
    off = base + hlen
    for (_, p), n in zip(model.named_parameters(), sizes):
        a = array("d")
        a.frombytes(data[off:off + 8 * n])
        if sys.byteorder == "big":
            a.byteswap()
        if not all(map(math.isfinite, a)):
            raise ValueError("non-finite weight values")
        p.data[:] = a
        off += 8 * n
    return model


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(model, path: str) -> str:
    """Atomically write ``dumps(model)`` to ``path``; returns its sha256 digest."""
    data = dumps(model)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".gnnw_", dir=d)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return digest(data)


def load(path: str, max_bytes: int = 1 << 30):
    with open(path, "rb") as fh:
        data = fh.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"weight file larger than {max_bytes} bytes")
    return loads(data)
