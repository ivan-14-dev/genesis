from .base import BudgetExceeded, Sandbox
from .causal_world import CausalWorld
from .sequence_world import SequenceWorld


def sandboxed(world, budget: int) -> Sandbox:
    return Sandbox(world, world.AGENT_METHODS, budget)


__all__ = ["BudgetExceeded", "CausalWorld", "Sandbox", "SequenceWorld", "sandboxed"]
