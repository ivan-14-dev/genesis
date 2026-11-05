"""Sandbox boundary between GENESIS and any world.

The agent never receives a world object. It receives a :class:`Sandbox` that
exposes only whitelisted methods, enforces an interaction budget and logs every
call (experiment log). Hidden rules and evaluation oracles stay on the world
object, which only the experimental harness holds.
"""

from __future__ import annotations

import time
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class Sandbox:
    __slots__ = ("_Sandbox__world", "_Sandbox__allowed", "budget", "used", "log", "env_id")

    def __init__(self, world: Any, allowed: tuple, budget: int):
        self.__world = world
        self.__allowed = frozenset(allowed)
        self.budget = budget
        self.used = 0
        self.log: list = []
        self.env_id = world.env_id

    def call(self, method: str, *args, **kwargs):
        if method not in self.__allowed:
            raise PermissionError(f"method {method!r} is not exposed by the sandbox")
        if self.used >= self.budget:
            raise BudgetExceeded(f"interaction budget of {self.budget} exhausted")
        self.used += 1
        result = getattr(self.__world, method)(*args, **kwargs)
        self.log.append({"ts": time.time(), "method": method, "args": repr(args)[:200]})
        return result

    def remaining(self) -> int:
        return self.budget - self.used
