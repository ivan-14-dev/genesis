from .base import BudgetExceeded, Sandbox
from .causal_world import CausalWorld
from .sequence_world import SequenceWorld
from .object_world import ObjectWorld


def sandboxed(world, budget: int) -> Sandbox:
    return Sandbox(world, world.AGENT_METHODS, budget)


__all__ = ["BudgetExceeded", "CausalWorld", "ObjectWorld", "Sandbox", "SequenceWorld", "sandboxed"]
