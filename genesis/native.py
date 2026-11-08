"""Secure build-and-load of the C kernels shipped in this repository.

Only sources that are part of GENESIS are compiled, with the system compiler (``cc``, then ``gcc``), into a
private cache directory (``GENESIS_CACHE_DIR`` or ``~/.cache/genesis``). Each library is named after the
SHA-256 of its source and flags, so a changed kernel is never confused with a stale build. Compilation goes to
a temporary file that is atomically renamed. A library is loaded only from a directory and file owned by the
current user and not writable by group/other, and the caller's ``setup`` hook must declare the ctypes
signatures and run a self-test before the library is used. Callers fall back to pure Python on any failure.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
from typing import Callable

BASE_CFLAGS = ("-O2", "-shared", "-fPIC", "-std=c99", "-ffp-contract=off")


def source_digest(source: str, cflags=BASE_CFLAGS) -> str | None:
    """SHA-256 over the source bytes and the compiler flags (None if the source is missing)."""
    try:
        with open(source, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return hashlib.sha256(data + b"\0" + " ".join(cflags).encode()).hexdigest()


def cache_dir() -> str:
    return os.environ.get("GENESIS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "genesis")


def check_private(path: str) -> None:
    """Native code is only loaded from a directory and file that no other user can write: otherwise anyone
    with write access to the cache could plant a library that runs inside GENESIS."""
    if os.name != "posix":
        return
    st = os.stat(path)
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise PermissionError(f"{path} must be owned by the current user and not group/other-writable")


def _compile(source: str, lib_path: str, cflags) -> str:
    errors = []
    for cc in ("cc", "gcc"):
        fd, tmp = tempfile.mkstemp(prefix=".gn_build_", suffix=".so", dir=os.path.dirname(lib_path))
        os.close(fd)
        try:
            res = subprocess.run([cc, *cflags, "-o", tmp, source], capture_output=True, timeout=120)
            if res.returncode == 0:
                os.chmod(tmp, 0o700)
                os.replace(tmp, lib_path)
                return cc
            errors.append(f"{cc}: exit {res.returncode}: {res.stderr.decode(errors='replace').strip()[:300]}")
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{cc}: {exc}")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise RuntimeError("could not compile C kernel: " + "; ".join(errors))


def load(source: str, name: str, setup: Callable[[ctypes.CDLL], object], cflags=BASE_CFLAGS) -> dict:
    """Compile (once, cached) and load ``source``; ``setup(lib)`` declares signatures, self-tests and returns
    the handle callers keep. Never raises: returns ``{"handle", "compiler", "library", "failure"}``."""
    info = {"handle": None, "compiler": None, "library": None, "failure": None}
    try:
        digest = source_digest(source, cflags)
        if digest is None:
            raise RuntimeError(f"kernel source not found: {source}")
        cache = cache_dir()
        os.makedirs(cache, mode=0o700, exist_ok=True)
        check_private(cache)
        lib_path = os.path.join(cache, f"{name}_{digest[:16]}.so")
        compiler = "cached"
        try:
            if not os.path.exists(lib_path):
                raise FileNotFoundError(lib_path)
            check_private(lib_path)
            handle = setup(ctypes.CDLL(lib_path))
        except (OSError, RuntimeError):  # missing, unusable or untrusted cached build: rebuild from source
            compiler = _compile(source, lib_path, cflags)
            check_private(lib_path)
            handle = setup(ctypes.CDLL(lib_path))
        info.update(handle=handle, compiler=compiler, library=lib_path)
    except Exception as exc:  # any failure means "C unavailable"; the caller reports it
        info["failure"] = f"{type(exc).__name__}: {exc}"
    return info
