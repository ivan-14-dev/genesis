"""Stage 7 — controlled self-improvement and self-diagnosis.

GENESIS may tune *learned-strategy* parameters (the allowlist ``TUNABLE`` in
``agent.py``) but never its code, its search budgets or its safety policy.
Every change follows the same protocol:

1. **Propose** one-parameter-at-a-time variants of the active configuration.
2. **Evaluate in isolation**: each variant is a fresh agent with a temporary
   memory, living in freshly generated sandboxed worlds. The live memory is
   never touched. The score is *observable*: the agent predicts the next terms
   of every stream, then actually observes them through the sandbox.
3. **Reproducibility check**: the baseline is evaluated twice; any difference
   aborts the cycle (an evaluation that is not deterministic cannot justify a
   change).
4. **Holdout validation**: the best trial variant must also score strictly
   better than the baseline on unseen worlds, without increasing observed
   false validations.
5. **Human approval**: a validated variant is stored as ``pending-approval``.
   Only :func:`approve` (a human-invoked CLI command) deploys it; the previous
   version stays recorded for :func:`rollback`.

It also produces a **limitations report**: what the agent's own records show
it cannot do (open problems that exceeded twice their expected cost, streams
judged unpredictable, unmet developmental gates).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Optional

from .adapters import SequenceAdapter
from .agent import TUNABLE, Genesis, validate_config
from .innate import DEFAULT_SAFETY
from .memory import LongTermMemory
from .metacognition import Metacognition

# Step used to propose neighbours of the current value of each tunable parameter.
STEPS = {"sequence.validation_k": 1, "sequence.shape_bias": 0.5, "sequence.max_proposed_constants": 4,
         "lab.validation_k": 1, "lab.shape_bias": 0.5, "agent.default_resolution_cost": 4,
         "agent.review_interval": 5, "curiosity.novelty_weight": 0.5, "curiosity.review_weight": 0.2}
DEFAULT_SEARCH = ("sequence.validation_k", "sequence.shape_bias", "agent.default_resolution_cost",
                  "curiosity.novelty_weight")


def _defaults() -> dict:
    tmp = tempfile.mkdtemp(prefix="genesis-defaults-")
    try:
        g = Genesis(os.path.join(tmp, "m.db"), config={})
        out = {}
        for k in TUNABLE:
            scope, name = k.split(".")
            obj = g.configs[scope] if scope in g.configs else (g if scope == "agent" else g.curiosity)
            out[k] = getattr(obj, name)
        g.ltm.close()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def propose(config: dict, params=DEFAULT_SEARCH) -> list:
    base = {**_defaults(), **config}
    out = []
    for p in params:
        lo, hi = TUNABLE[p]
        for sign in (-1, 1):
            v = base[p] + sign * STEPS[p]
            v = type(base[p])(v) if isinstance(base[p], int) else round(v, 4)
            if lo <= v <= hi:
                out.append(({**config, p: v}, f"{p}: {base[p]} -> {v}"))
    return out


def evaluate(config: dict, world_factory, seeds, budget: int = 100, horizon: int = 5) -> dict:
    """Observable score of a configuration, measured in isolated temporary agents.

    ``world_factory(seed)`` must return a sandbox over a fresh sequence world.
    """
    validate_config(config)
    acc, false_val, cpu = [], 0, 0.0
    for seed in seeds:
        tmp = tempfile.mkdtemp(prefix="genesis-trial-")
        try:
            agent = Genesis(os.path.join(tmp, "m.db"), seed=seed, config=config)
            sb = world_factory(seed)
            t0 = time.process_time()
            agent.live(SequenceAdapter(agent, sb), budget, "curiosity")
            for sid in sb.call("stream_ids"):
                key = f"{sb.env_id}/{sid}"
                pred = agent.rollout_sequence(key, horizon)
                seen = [sb.call("next_term", sid) for _ in range(horizon)]
                ok = sum(a == b for a, b in zip(pred, seen)) / horizon
                acc.append(ok)
                lr = agent.learners.get(key)
                false_val += bool(lr and lr.status() == "validated" and ok < 1.0)
            cpu += time.process_time() - t0
            agent.ltm.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return {"score": round(sum(acc) / len(acc), 4), "observed_false_validations": false_val,
            "cpu_seconds": round(cpu, 3), "seeds": list(seeds), "budget": budget}


def improve(memory_path: str, world_factory, trial_seeds, holdout_seeds, params=DEFAULT_SEARCH,
            budget: int = 100, min_gain: float = 0.01) -> dict:
    """One self-improvement cycle. Never deploys: the best variant waits for human approval."""
    ltm = LongTermMemory(memory_path)
    try:
        if ltm.get_meta("safety_policy_digest") not in (None, DEFAULT_SAFETY.digest()):
            raise PermissionError("safety policy changed since birth; self-improvement refused")
        current = ltm.get_meta("active_config") or {}
        deployed = [v for v in ltm.versions() if v["status"] == "deployed"]
        parent = deployed[-1]["id"] if deployed else None
        base = evaluate(current, world_factory, trial_seeds, budget)
        again = evaluate(current, world_factory, trial_seeds, budget)
        reproducible = base["score"] == again["score"] and \
            base["observed_false_validations"] == again["observed_false_validations"]
        if not reproducible:
            return {"reproducible": False, "baseline": base, "repeat": again, "decision": "aborted"}
        trials = []
        for cfg, why in propose(current, params):
            m = evaluate(cfg, world_factory, trial_seeds, budget)
            trials.append((m["score"] - base["score"], cfg, why, m))
        trials.sort(key=lambda t: -t[0])
        result = {"reproducible": True, "baseline": base, "variants": [
            {"change": why, "trial": m, "gain": round(g, 4)} for g, _, why, m in trials]}
        best = trials[0] if trials and trials[0][0] >= min_gain else None
        for g, cfg, why, m in trials:
            if best is None or cfg is not best[1]:
                ltm.add_version(cfg, parent, why, m, None, "rejected-trial", DEFAULT_SAFETY.digest())
        if best is None:
            result["decision"] = "no variant improved the trial score"
            return result
        g, cfg, why, m = best
        hb = evaluate(current, world_factory, holdout_seeds, budget)
        hv = evaluate(cfg, world_factory, holdout_seeds, budget)
        # A trial gain must be confirmed on unseen worlds; "not worse" is not evidence of improvement.
        ok = hv["score"] > hb["score"] and \
            hv["observed_false_validations"] <= hb["observed_false_validations"]
        status = "pending-approval" if ok else "rejected-holdout"
        vid = ltm.add_version(cfg, parent, why, {"variant": m, "baseline": base},
                              {"variant": hv, "baseline": hb}, status, DEFAULT_SAFETY.digest())
        result.update(candidate={"id": vid, "change": why, "holdout_variant": hv, "holdout_baseline": hb},
                      decision=status)
        return result
    finally:
        ltm.close()


def approve(memory_path: str, version_id: int, approved_by: str, admin_seed: Optional[bytes] = None) -> dict:
    """Human-only deployment of a validated version. Once an admin key is pinned in the memory, the approval
    must be signed with it (``admin_seed``); either way the decision is appended to the audit chain."""
    from . import governance
    if not approved_by:
        raise PermissionError("approval requires a named human approver")
    ltm = LongTermMemory(memory_path)
    try:
        v = next((x for x in ltm.versions() if x["id"] == version_id), None)
        if v is None or v["status"] != "pending-approval":
            raise PermissionError(f"version {version_id} is not pending approval")
        if v["safety_digest"] != DEFAULT_SAFETY.digest():
            raise PermissionError("version was validated under a different safety policy")
        validate_config(v["config"])
        approval = None
        if governance.pinned_key(ltm) is not None:
            if admin_seed is None:
                raise PermissionError("an admin key is pinned in this memory: the approval must be signed (--key)")
            approval = governance.admin_approval(ltm, admin_seed, "version-approval",
                                                 {"version_id": version_id, "config": v["config"]}, approved_by)
        else:
            ltm.append_audit("version.approve", "allow", approved_by, "human", version_id, {"config": v["config"]})
        for x in ltm.versions():
            if x["status"] == "deployed":
                ltm.set_version_status(x["id"], "superseded")
        ltm.set_meta("previous_config", ltm.get_meta("active_config") or {})
        ltm.set_meta("active_config", v["config"])
        ltm.set_version_status(version_id, "deployed", approved_by)
        if approval is not None:
            ltm.db.execute("UPDATE model_versions SET approval=? WHERE id=?", (json.dumps(approval), version_id))
        return v["config"]
    finally:
        ltm.close()


def rollback(memory_path: str) -> dict:
    """Restore the configuration that was active before the last deployment."""
    ltm = LongTermMemory(memory_path)
    try:
        prev = ltm.get_meta("previous_config") or {}
        for x in ltm.versions():
            if x["status"] == "deployed":
                ltm.set_version_status(x["id"], "rolled-back")
        ltm.set_meta("active_config", prev)
        ltm.append_audit("version.rollback", "allow", "human", "human", None, {"restored_config": prev})
        return prev
    finally:
        ltm.close()


def limitations(agent: Genesis) -> dict:
    """What the agent's own records say it cannot (yet) do."""
    stuck, unpredictable = [], []
    for key, lr in sorted(agent.learners.items()):
        st = lr.status()
        kind = "lab" if lr.cfg.allow_conditionals else "sequence"
        n = len(lr.data)
        if st == "unpredictable":
            unpredictable.append(key)
        elif st != "validated" and n > 2 * agent.expected_resolution_cost(kind):
            stuck.append({"key": key, "observations": n, "expected_cost": agent.expected_resolution_cost(kind),
                          "best_guess": str(lr.best()[0].expr) if lr.best()[0] else None,
                          "diagnosis": "no expression within the search bound explains the data; either noise "
                                       "or a rule outside the hypothesis space (e.g. a constant never proposed)"})
    stage = Metacognition.current_stage(agent.assessments())
    blocked = stage["blocked_by"] if isinstance(stage["blocked_by"], list) else []
    return {"n_identified_from_records": len(stuck) + len(unpredictable) + len(blocked),
            "stuck_problems": stuck, "judged_unpredictable": unpredictable,
            "blocked_by": stage["blocked_by"], "stage": stage["stage"],
            "architectural_notes": [
                "hypothesis space is bounded: programs up to max_size, deeper ones only as deepening "
                "compositions (up to deepen_max_size), sparse linear combinations or through invented primitives",
                "constants are proposed from observed data only; an unseen modulus is found only for affine "
                "modular rules (gcd of residue differences)",
                "search budgets are safety-locked and cannot be raised by self-improvement",
            ]}
