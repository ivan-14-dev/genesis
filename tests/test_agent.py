import ast
import pathlib

import pytest

from genesis.agent import Genesis, IntegrityError
from genesis.environments import BudgetExceeded, CausalWorld, SequenceWorld, sandboxed
from genesis.experiments import (confound_resolved, evaluate_sequences, lab_accuracy, lab_eval_set,
                                 tabular_accuracy)
from genesis.innate import SafetyPolicy
from genesis.metacognition import Metacognition

PKG = pathlib.Path(__file__).resolve().parents[1] / "genesis"


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("lab") / "m.db")
    world = CausalWorld(0)
    sb = sandboxed(world, 5000)
    agent = Genesis(db, seed=0)
    held = lab_eval_set(world, 300, 991, 0, world.n_values - 1)
    acc0 = lab_accuracy(agent, world, held)
    agent.watch_lab(sb, 60)
    acc_passive = lab_accuracy(agent, world, held)
    agent.experiment_lab(sb, 200)
    yield {"agent": agent, "world": world, "sb": sb, "held": held, "db": db,
           "acc0": acc0, "acc_passive": acc_passive}
    agent.close()


def test_stage0_audit_and_manifest(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"))
    audit = agent.ltm.get_meta("stage0_audit")
    assert all(audit["checks"].values())
    assert agent.ltm.get_meta("innate_manifest")["domain_knowledge"] == []
    assert agent.ltm.knowledge() == []
    assert Metacognition.current_stage(agent.assessments())["stage"] == 1
    agent.close()


def test_safety_policy_cannot_change_after_birth(tmp_path):
    db = str(tmp_path / "m.db")
    Genesis(db).close()
    with pytest.raises(IntegrityError):
        Genesis(db, safety=SafetyPolicy(network_access=True))


def test_sandbox_blocks_hidden_methods_and_enforces_budget():
    sb = sandboxed(CausalWorld(0), 2)
    with pytest.raises(PermissionError):
        sb.call("_describe_rules")
    with pytest.raises(PermissionError):
        sb.call("_true_next", {}, "a0")
    sb.call("spec")
    sb.call("spec")
    with pytest.raises(BudgetExceeded):
        sb.call("spec")


def test_cognitive_modules_never_touch_world_internals():
    """Static check: agent-side code cannot read hidden rules or oracles."""
    harness = {"experiments.py", "demo.py", "__main__.py"}
    for path in PKG.glob("*.py"):
        if path.name in harness:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"_rules", "_true_next", "_describe_rules",
                                                                 "_future", "_family", "_confound",
                                                                 "_render", "_Sandbox__world"}:
                pytest.fail(f"{path.name}:{node.lineno} accesses {node.attr}")
            if isinstance(node, ast.ImportFrom) and node.module and "environments" in node.module:
                names = {a.name for a in node.names}
                assert names <= {"BudgetExceeded", "Sandbox"}, f"{path.name} imports {names}"


def test_sequence_rules_discovered_and_noise_rejected(tmp_path):
    world = SequenceWorld(0)
    agent = Genesis(str(tmp_path / "m.db"), seed=0)
    agent.explore_sequences(sandboxed(world, 1000), 600)
    ev = evaluate_sequences(agent, world)
    assert ev["rule_extrapolation_accuracy"] >= 0.8
    assert ev["false_validation_rate"] == 0.0
    noise = [r for r in ev["streams"] if r["family"] == "noise"]
    assert all(r["status"] != "validated" for r in noise)
    agent.close()


def test_lab_learning_curve(lab):
    assert lab["acc0"] == 0.0
    final = lab_accuracy(lab["agent"], lab["world"], lab["held"])
    assert final >= 0.95
    assert final > lab["acc_passive"]


def test_generalises_beyond_memorisation(lab):
    world = lab["world"]
    agent = lab["agent"]
    held = lab["held"]
    assert lab_accuracy(agent, world, held) > tabular_accuracy(agent, world, held) + 0.2


def test_confound_resolved_by_experiment(lab):
    conf = confound_resolved(lab["agent"], lab["world"])
    confounded = [c for c in conf.values() if c["confounded"]]
    assert confounded
    assert all(c["correct_cause"] and c["status"] == "validated" for c in confounded)


def test_knowledge_is_explainable_with_evidence(lab):
    agent = lab["agent"]
    validated = agent.ltm.knowledge(status="validated")
    assert validated
    ex = agent.ltm.explain(validated[0].id)
    assert ex["evidence"]["supports"]["total"] > 0
    refuted = agent.ltm.knowledge(status="refuted")
    if refuted:
        ex = agent.ltm.explain(refuted[0].id)
        assert ex["evidence"]["refuted_by"]["total"] > 0


def test_planning_uses_learned_model(lab):
    agent, sb = lab["agent"], lab["sb"]
    results = [agent.pursue_goal(sb, 500 + i, "plan") for i in range(10)]
    assert sum(r["success"] for r in results) >= 9


def test_persistence_and_no_forgetting(tmp_path):
    db = str(tmp_path / "m.db")
    world = CausalWorld(2)
    sb = sandboxed(world, 5000)
    agent = Genesis(db, seed=2)
    agent.watch_lab(sb, 60)
    agent.experiment_lab(sb, 200)
    held = lab_eval_set(world, 200, 7, 0, world.n_values - 1)
    before = lab_accuracy(agent, world, held)
    agent.close()

    agent = Genesis(db, seed=2)
    assert agent.session == 1
    assert lab_accuracy(agent, world, held) == before
    other = CausalWorld(77)
    sb2 = sandboxed(other, 5000)
    agent.watch_lab(sb2, 60)
    agent.experiment_lab(sb2, 200)
    assert lab_accuracy(agent, world, held) == before
    agent.close()


def test_checkpoint_and_rollback(tmp_path):
    db = str(tmp_path / "m.db")
    agent = Genesis(db, seed=0)
    ck = agent.ltm.checkpoint(str(tmp_path / "ck.db"))
    agent.explore_sequences(sandboxed(SequenceWorld(1), 200), 50)
    assert agent.ltm.count("episodes") > 0
    agent.ltm.close()
    agent.ltm.restore(ck, db)
    restored = Genesis(db, seed=0)
    assert restored.ltm.count("episodes") == 0
    restored.close()
