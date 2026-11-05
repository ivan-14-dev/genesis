"""Innate (pre-wired) mechanisms of GENESIS.

Everything in this module exists *before* any experience. It is deliberately
domain-free: elementary arithmetic and logic primitives, a handful of small
integer constants, and the safety envelope. No world rules, no answers, no
facts about any environment live here.

The manifest produced by :func:`innate_manifest` is written to long-term memory
at birth so that innate mechanisms and acquired knowledge are always
distinguishable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

VALUE_LIMIT = 10**18


def _bounded(v: Optional[int]) -> Optional[int]:
    if v is None or v < -VALUE_LIMIT or v > VALUE_LIMIT:
        return None
    return v


def _add(a: int, b: int) -> Optional[int]:
    return _bounded(a + b)


def _sub(a: int, b: int) -> Optional[int]:
    return _bounded(a - b)


def _mul(a: int, b: int) -> Optional[int]:
    return _bounded(a * b)


def _mod(a: int, b: int) -> Optional[int]:
    return a % b if b > 0 else None


@dataclass(frozen=True)
class Primitive:
    name: str
    symbol: str
    arity: int
    fn: Callable[..., Optional[int]]
    commutative: bool
    category: str
    description: str


ARITHMETIC_PRIMITIVES: dict[str, Primitive] = {
    "add": Primitive("add", "+", 2, _add, True, "arithmetic", "integer addition"),
    "sub": Primitive("sub", "-", 2, _sub, False, "arithmetic", "integer subtraction"),
    "mul": Primitive("mul", "*", 2, _mul, True, "arithmetic", "integer multiplication"),
    "mod": Primitive("mod", "%", 2, _mod, False, "arithmetic", "remainder by a positive constant"),
}

# Logic primitives: equality test and conditional selection. They are used by
# the induction engine to build "if <var> == <c> then <e1> else <e2>".
LOGIC_PRIMITIVES = {
    "eq": "equality test between two integers -> {0, 1}",
    "if_eq": "select between two values depending on an equality test",
}

# Small integers available without experience (subitizing-like range).
INNATE_CONSTANTS: tuple[int, ...] = (0, 1, 2, 3)

INNATE_MECHANISMS = {
    "pattern_detection": "bottom-up enumeration of compositions of innate primitives, scored by Bayesian Occam prior",
    "prediction_error": "binary error + surprise (-log2 p) of each outcome under the predictive distribution",
    "reward": "scalar extrinsic reward (+1 goal reached) delivered by environments",
    "memory_management": "limited-capacity working memory, SQLite long-term memory",
    "novelty": "visit counts over perceptual signatures",
    "environment_representation": "observations are dicts of named integer variables; actions are opaque tokens",
}


@dataclass(frozen=True)
class SafetyPolicy:
    """Hard limits. The agent may read but never modify these.

    The policy hash is stored at birth; an agent refuses to run if the active
    policy no longer matches (no autonomous modification of safety constraints).
    """

    max_interactions_per_session: int = 5000
    max_candidates_per_induction: int = 60000
    max_seconds_per_induction: float = 5.0
    max_plan_depth: int = 6
    network_access: bool = False
    filesystem_access: str = "own memory database only"
    self_modification: str = "learned parameters and hypothesis priors only; never code or safety policy"
    external_actions_require_authorization: bool = True

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


DEFAULT_SAFETY = SafetyPolicy()


def innate_manifest() -> dict:
    manifest = {
        "arithmetic_primitives": {k: p.description for k, p in ARITHMETIC_PRIMITIVES.items()},
        "logic_primitives": dict(LOGIC_PRIMITIVES),
        "innate_constants": list(INNATE_CONSTANTS),
        "mechanisms": dict(INNATE_MECHANISMS),
        "domain_knowledge": [],
        "pretrained_models": [],
    }
    manifest["digest"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest
