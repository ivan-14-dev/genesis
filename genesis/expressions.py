"""Symbolic expressions built from innate primitives, and their enumeration.

An expression is a candidate *model* of how one observed variable depends on
others. The agent never receives expressions from outside: it constructs them
by composing innate primitives and keeps the ones the data supports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .innate import ARITHMETIC_PRIMITIVES


@dataclass(frozen=True, slots=True)
class Expr:
    op: str  # "var" | "const" | primitive name | "if_eq"
    args: tuple = ()
    value: object = None
    size: int = 1

    def evaluate(self, env: dict) -> Optional[int]:
        if self.op == "var":
            return env.get(self.value)
        if self.op == "const":
            return self.value
        if self.op == "if_eq":
            q, c, e1, e2 = self.args
            qv = q.evaluate(env)
            if qv is None:
                return None
            return e1.evaluate(env) if qv == c.value else e2.evaluate(env)
        a = self.args[0].evaluate(env)
        b = self.args[1].evaluate(env)
        if a is None or b is None:
            return None
        return ARITHMETIC_PRIMITIVES[self.op].fn(a, b)

    def __str__(self) -> str:
        if self.op == "var":
            return str(self.value)
        if self.op == "const":
            return str(self.value)
        if self.op == "if_eq":
            q, c, e1, e2 = self.args
            return f"if {q} == {c} then {e1} else {e2}"
        return f"({self.args[0]} {ARITHMETIC_PRIMITIVES[self.op].symbol} {self.args[1]})"

    def shape(self) -> str:
        """Structure with variables/constants abstracted (used by procedural memory)."""
        if self.op == "var":
            return "V"
        if self.op == "const":
            return "C"
        return f"{self.op}({','.join(a.shape() for a in self.args)})"

    def variables(self) -> set:
        if self.op == "var":
            return {self.value}
        out: set = set()
        for a in self.args:
            out |= a.variables()
        return out

    def condition_variable(self) -> Optional[str]:
        return self.args[0].value if self.op == "if_eq" else None

    def to_json(self):
        if self.op in ("var", "const"):
            return [self.op, self.value]
        return [self.op, [a.to_json() for a in self.args]]

    @staticmethod
    def from_json(obj) -> "Expr":
        op, payload = obj
        if op == "var":
            return var(payload)
        if op == "const":
            return const(payload)
        return app(op, *[Expr.from_json(a) for a in payload])


def var(name: str) -> Expr:
    return Expr("var", (), name, 1)


def const(c: int) -> Expr:
    return Expr("const", (), c, 1)


def app(op: str, *args: Expr) -> Expr:
    return Expr(op, tuple(args), None, 1 + sum(a.size for a in args))


def if_eq(q: str, c: int, e1: Expr, e2: Expr) -> Expr:
    return app("if_eq", var(q), const(c), e1, e2)


@dataclass
class Enumeration:
    candidates: list  # list[(Expr, signature tuple)] ordered by size
    counts: dict  # size -> number of syntactically constructed expressions
    evaluated: int
    truncated: bool
    seconds: float = 0.0
    by_size: dict = field(default_factory=dict)


def _trivial(op: str, a: Expr, b: Expr) -> bool:
    if a.op == "const" and b.op == "const":
        return True
    if op in ("add", "sub") and b.op == "const" and b.value == 0:
        return True
    if op == "add" and a.op == "const" and a.value == 0:
        return True
    if op == "mul" and ((a.op == "const" and a.value in (0, 1)) or (b.op == "const" and b.value in (0, 1))):
        return True
    if op == "sub" and a == b:
        return True
    if op == "mod" and not (b.op == "const" and b.value > 1):
        return True
    return False


def enumerate_programs(
    inputs: list[dict],
    var_names: Iterable[str],
    constants: Iterable[int],
    max_size: int,
    max_candidates: int,
    max_seconds: float,
    ops: Iterable[str] = ("add", "sub", "mul", "mod"),
    keep_all_upto: int = 3,
) -> Enumeration:
    """Bottom-up enumeration with observational-equivalence pruning.

    Expressions of size <= ``keep_all_upto`` are all kept (so equivalent-on-data
    alternatives survive and curiosity can discriminate them). Larger ones are
    deduplicated by their output signature on ``inputs``.
    """
    t0 = time.perf_counter()
    n = len(inputs)
    counts: dict[int, int] = {}
    by_size: dict[int, list] = {}
    seen: set = set()
    candidates: list = []
    evaluated = 0
    truncated = False

    leaves = []
    for v in var_names:
        sig = tuple(inp.get(v) for inp in inputs)
        if None not in sig:
            leaves.append((var(v), sig))
    for c in sorted(set(constants)):
        leaves.append((const(c), (c,) * n))
    counts[1] = len(leaves)
    by_size[1] = leaves
    for e, sig in leaves:
        candidates.append((e, sig))
        seen.add(sig)

    prims = [ARITHMETIC_PRIMITIVES[o] for o in ops]
    for size in range(3, max_size + 1, 2):
        level = []
        constructed = 0
        for ls in range(1, size - 1, 2):
            rs = size - 1 - ls
            if ls not in by_size or rs not in by_size:
                continue
            left, right = by_size[ls], by_size[rs]
            for p in prims:
                fn = p.fn
                for i, (ea, sa) in enumerate(left):
                    for j, (eb, sb) in enumerate(right):
                        if p.commutative and (ls > rs or (ls == rs and j < i)):
                            continue
                        if _trivial(p.name, ea, eb):
                            continue
                        constructed += 1
                        evaluated += 1
                        sig = tuple(fn(x, y) for x, y in zip(sa, sb))
                        if None in sig:
                            continue
                        if size > keep_all_upto and sig in seen:
                            continue
                        seen.add(sig)
                        e = Expr(p.name, (ea, eb), None, size)
                        level.append((e, sig))
                        if len(candidates) + len(level) >= max_candidates or time.perf_counter() - t0 > max_seconds:
                            truncated = True
                            break
                    if truncated:
                        break
                if truncated:
                    break
            if truncated:
                break
        counts[size] = max(constructed, 1)
        by_size[size] = level
        candidates.extend(level)
        if truncated:
            break
    return Enumeration(candidates, counts, evaluated, truncated, time.perf_counter() - t0, by_size)
