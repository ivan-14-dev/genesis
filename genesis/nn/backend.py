"""Matrix-multiply backends for :mod:`genesis.nn`: pure Python and an in-repo C kernel.

Everything here is built from this repository: the C backend compiles only the
shipped ``_kernels.c`` with the system compiler (``cc``, then ``gcc``) into a
cache directory (``GENESIS_CACHE_DIR`` or ``~/.cache/genesis``), named after the
SHA-256 of the source so a changed kernel is never confused with a stale build.
Compilation goes to a temporary file that is atomically renamed, and the
library is loaded with :mod:`ctypes` and self-tested before use. If anything
fails, ``auto`` mode silently falls back to the pure-Python implementation and
records the reason in :func:`backend_info`.

Selection: env ``GENESIS_NN_BACKEND`` in {``auto`` (default), ``python``, ``c``}
or :func:`set_backend`. Both backends are deterministic; they may differ from
each other in the last bits because Python's ``sumprod`` rounds differently.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import operator
import os
import random
import subprocess
import tempfile
import time
from array import array

__all__ = ["matmul", "set_backend", "backend_name", "backend_info", "kernel_source_digest",
           "available_backends", "benchmark_matmul"]

KERNEL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_kernels.c")
_BACKENDS = ("auto", "python", "c")
_CFLAGS = ["-O2", "-shared", "-fPIC", "-std=c99", "-ffp-contract=off"]

_mul, _add = operator.mul, operator.add
_dot = getattr(math, "sumprod", None) or (lambda x, y: sum(map(_mul, x, y)))

_requested: str | None = None   # None: read GENESIS_NN_BACKEND on first use
_active: str | None = None
_impl = None
_c: dict = {"tried": False, "fn": None, "compiler": None, "library": None, "failure": None}


def _matmul_python(a, b, m, k, n, ta, tb, out, acc):
    rows = [a[i::m] for i in range(m)] if ta else [a[i * k:(i + 1) * k] for i in range(m)]
    cols = [b[j * k:(j + 1) * k] for j in range(n)] if tb else [b[j::n] for j in range(n)]
    dot = _dot
    res = array("d", [dot(r, c) for r in rows for c in cols])
    if out is None:
        return res
    out[:] = array("d", map(_add, out, res)) if acc else res
    return out


def _matmul_c(a, b, m, k, n, ta, tb, out, acc):
    if out is None:
        out = array("d", bytes(8 * m * n))
    if m == 0 or n == 0:
        return out
    if k == 0:
        if not acc:
            out[:] = array("d", bytes(8 * m * n))
        return out
    _c["fn"]((ctypes.c_double * len(a)).from_buffer(a), (ctypes.c_double * len(b)).from_buffer(b),
             (ctypes.c_double * len(out)).from_buffer(out), m, k, n, int(ta), int(tb), int(acc))
    return out


def kernel_source_digest() -> str | None:
    """SHA-256 hex digest of the shipped C kernel source (None if it is missing)."""
    try:
        with open(KERNEL_SOURCE, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _cache_dir() -> str:
    return os.environ.get("GENESIS_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "genesis")


def _check_private(path: str) -> None:
    """Native code is only loaded from a directory and file that no other user can write: otherwise anyone
    with write access to the cache could plant a library that runs inside GENESIS."""
    if os.name != "posix":
        return
    st = os.stat(path)
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise PermissionError(f"{path} must be owned by the current user and not group/other-writable")


def _compile(lib_path: str) -> str:
    errors = []
    for cc in ("cc", "gcc"):
        fd, tmp = tempfile.mkstemp(prefix=".gn_kernels_", suffix=".so", dir=os.path.dirname(lib_path))
        os.close(fd)
        try:
            res = subprocess.run([cc, *_CFLAGS, "-o", tmp, KERNEL_SOURCE], capture_output=True, timeout=60)
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


def _open_lib(lib_path: str):
    fn = ctypes.CDLL(lib_path).gn_matmul
    fn.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_int] * 6
    fn.restype = None
    _c["fn"] = fn
    got = _matmul_c(array("d", [1, 2, 3, 4]), array("d", [5, 6, 7, 8]), 2, 2, 2, False, False, None, False)
    if list(got) != [19.0, 22.0, 43.0, 50.0]:
        _c["fn"] = None
        raise RuntimeError(f"C kernel self-test failed: {list(got)}")


def _load_c() -> bool:
    """Compile (once, cached on disk) and load the C kernel; never raises."""
    if _c["tried"]:
        return _c["fn"] is not None
    _c["tried"] = True
    try:
        digest = kernel_source_digest()
        if digest is None:
            raise RuntimeError(f"kernel source not found: {KERNEL_SOURCE}")
        cache = _cache_dir()
        os.makedirs(cache, mode=0o700, exist_ok=True)
        _check_private(cache)
        lib_path = os.path.join(cache, f"gn_kernels_{digest[:16]}.so")
        compiler = "cached"
        try:
            if not os.path.exists(lib_path):
                raise FileNotFoundError(lib_path)
            _check_private(lib_path)
            _open_lib(lib_path)
        except (OSError, RuntimeError):  # missing, unusable or untrusted cached build: rebuild from source
            compiler = _compile(lib_path)
            _check_private(lib_path)
            _open_lib(lib_path)
        _c.update(compiler=compiler, library=lib_path, failure=None)
        return True
    except Exception as exc:  # any failure means "C unavailable", reported via backend_info()
        _c.update(fn=None, failure=f"{type(exc).__name__}: {exc}")
        return False


def _select(name: str) -> str:
    if name not in _BACKENDS:
        raise ValueError(f"unknown nn backend {name!r}; expected one of {_BACKENDS}")
    if name == "python":
        return "python"
    if _load_c():
        return "c"
    if name == "c":
        raise RuntimeError(f"C backend unavailable: {_c['failure']}")
    return "python"


def _resolve():
    global _active, _impl
    if _active is None:
        req = _requested or (os.environ.get("GENESIS_NN_BACKEND", "auto").strip().lower() or "auto")
        _active = _select(req)
        _impl = _matmul_c if _active == "c" else _matmul_python
    return _impl


def set_backend(name: str) -> str:
    """Select ``auto``/``python``/``c`` programmatically; returns the active backend name."""
    global _requested, _active, _impl
    name = str(name).strip().lower()
    active = _select(name)
    _requested, _active = name, active
    _impl = _matmul_c if active == "c" else _matmul_python
    return active


def backend_name() -> str:
    _resolve()
    return _active


def available_backends() -> list[str]:
    return ["python", "c"] if _load_c() else ["python"]


def backend_info() -> dict:
    _resolve()
    return {"name": _active, "requested": _requested or os.environ.get("GENESIS_NN_BACKEND", "auto"),
            "compiler": _c["compiler"], "library": _c["library"],
            "kernel_source_sha256": kernel_source_digest(), "failure": _c["failure"]}


def _as_buffer(x):
    return x if isinstance(x, array) and x.typecode == "d" else array("d", x)


def matmul(a, b, m: int, k: int, n: int, trans_a: bool = False, trans_b: bool = False,
           out=None, accumulate: bool = False):
    """C (m x n) = op(A) @ op(B) on flat row-major ``array('d')`` buffers (see module doc)."""
    a, b = _as_buffer(a), _as_buffer(b)
    if len(a) != m * k or len(b) != k * n:
        raise ValueError(f"matmul size mismatch: len(a)={len(a)}, len(b)={len(b)}, m={m}, k={k}, n={n}")
    if out is not None and (len(out) != m * n or out is a or out is b
                            or not (isinstance(out, array) and out.typecode == "d")):
        raise ValueError("out must be a distinct array('d') of length m*n")
    if accumulate and out is None:
        accumulate = False
    return (_impl or _resolve())(a, b, m, k, n, trans_a, trans_b, out, accumulate)


def benchmark_matmul(n: int = 128, repeats: int = 3) -> dict:
    """Best-of-``repeats`` GFLOP/s of an n x n x n product for each available backend."""
    rng = random.Random(0)
    a = array("d", [rng.random() for _ in range(n * n)])
    b = array("d", [rng.random() for _ in range(n * n)])
    out = array("d", bytes(8 * n * n))
    result = {}
    for name in available_backends():
        fn = _matmul_c if name == "c" else _matmul_python
        best = math.inf
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            fn(a, b, n, n, n, False, False, out, False)
            best = min(best, time.perf_counter() - t0)
        result[name] = 2.0 * n ** 3 / max(best, 1e-12) / 1e9
    return result
