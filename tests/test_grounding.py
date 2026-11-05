import json

import pytest

from genesis.agent import Genesis
from genesis.environments import ObjectWorld, sandboxed
from genesis.grounding import GroundedExplorer, Projection
from genesis.innate import SafetyPolicy


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    a = Genesis(str(tmp_path_factory.mktemp("grounded") / "m.db"))
    world = ObjectWorld(0)
    g = GroundedExplorer(a, sandboxed(world, 1000))
    assert all(g.predict(obj["features"], act)["value"] is None for obj in g.objects for act in g.actions)
    g.run(80)
    yield a, world, g
    a.close()


def test_learns_relevant_channels_not_identifiers(trained):
    _, world, g = trained
    for action in g.actions:
        learned = g.models[action].description()
        assert set(learned["features"]) == set(world._rules[action]["channels"])
        assert len(learned["features"]) < len(g.features)
    for obj in world._heldout(777):
        for action in g.actions:
            assert g.predict(obj["features"], action)["value"] == world._true_next(obj["features"], action)


def test_abstains_on_unseen_relevant_values(trained):
    _, world, g = trained
    for action in g.actions:
        obj = world._heldout(888, 1)[0]
        obj["features"][world._rules[action]["channels"][0]] = 2
        assert g.predict(obj["features"], action)["value"] is None


def test_future_validation_and_controlled_contrasts(trained):
    a, _, g = trained
    concepts = a.ltm.knowledge(kind="grounded-concept", status="validated")
    assert len(concepts) == len(g.actions)
    for concept in concepts:
        assert concept.provenance == "induced from controlled object experiments"
        ex = a.ltm.explain(concept.id)
        assert ex["evidence"]["supports"]["total"] >= 4
        for ep in ex["evidence"]["supports"]["examples"]:
            prediction = next(iter(json.loads(ep["predictions"]).values()))
            assert prediction["novel_input"] and prediction["predicted"] == prediction["actual"]
            assert prediction["confidence"] is None
    for action in g.actions:
        for channel, (a_id, b_id) in g.models[action].contrasts().items():
            rows = {r["episode"]: r for r in g.rows(action)}
            x, y = rows[a_id], rows[b_id]
            assert x["effect"] != y["effect"]
            assert [f for f in g.features if x["features"][f] != y["features"][f]] == [channel]


def test_ambiguous_correlations_do_not_claim_identified_concepts():
    rows = [{"features": {"p": i, "q": i}, "effect": i, "episode": i + 1} for i in (0, 1)]
    model = Projection(["p", "q"], rows)
    assert model.description() is None
    assert len(model.models) == 2
    assert model.predict({"p": 0, "q": 1})["value"] is None
    rows.append({"features": {"p": 0, "q": 1}, "effect": 0, "episode": 3})
    model = Projection(["p", "q"], rows)
    assert model.description()["features"] == ["p"]
    assert model.predict({"p": 1, "q": 0})["value"] == 1


def test_goal_selection_uses_learned_effects(trained):
    _, world, g = trained
    for obj in world._heldout(999):
        for target in (0, 1):
            possible = [a for a in g.actions if world._true_next(obj["features"], a) == target]
            chosen = g.choose_action(obj["features"], target)
            assert (chosen in possible) if possible else chosen is None


def test_sandbox_and_shared_safety_limits(tmp_path):
    a = Genesis(str(tmp_path / "limited.db"), safety=SafetyPolicy(max_interactions_per_session=2))
    try:
        sb = sandboxed(ObjectWorld(0), 100)
        with pytest.raises(PermissionError):
            sb.call("_true_next", {}, "a0")
        g = GroundedExplorer(a, sb)
        assert g.run(10)["steps"] == 2
        assert g.run(1)["steps"] == 0
        assert a.interactions == 2
        assert a.ltm.count("episodes") == 2
        with pytest.raises(ValueError):
            g.run(-1)
    finally:
        a.close()


def test_sandbox_exhaustion_is_reported(tmp_path):
    a = Genesis(str(tmp_path / "budget.db"))
    try:
        g = GroundedExplorer(a, sandboxed(ObjectWorld(1), 2))
        r = g.run(8)
        assert r["steps"] == 1 and "budget" in r["stopped"]
    finally:
        a.close()


def test_repeated_inputs_cannot_validate_an_abstraction(tmp_path):
    a = Genesis(str(tmp_path / "repeat.db"))
    try:
        g = GroundedExplorer(a, sandboxed(ObjectWorld(1), 50))
        for _ in range(10):
            g.experience(g.objects[0], "a0")
        assert not a.ltm.knowledge(kind="grounded-concept", status="validated")
    finally:
        a.close()


def test_grounded_experiment_baselines_adaptation_and_persistence(tmp_path):
    from genesis.experiments import run_grounded_learning
    db = str(tmp_path / "benchmark.db")
    r = run_grounded_learning(7, db=db)
    assert r["initial_knowledge"] == 0 and r["initial"]["coverage"] == 0
    assert r["transfer"]["accuracy"] >= 0.9
    assert r["correct_feature_sets"] == 2
    assert r["validated_concepts"] == 2
    assert r["baselines"]["object_id_lookup"]["coverage"] == 0
    assert r["baselines"]["full_state_lookup"]["coverage"] == 0
    assert r["baselines"]["existing_symbolic"]["accuracy"] >= 0.9
    assert r["transfer"]["accuracy"] > r["baselines"]["majority"]["accuracy"]
    assert r["goals"]["success_rate"] > r["goals"]["random_expected_success"]
    assert r["reload_retention"] == r["other_world_retention"] == 1
    assert r["changed_rule_before"]["accuracy"] == 0
    assert r["changed_rule_after"]["accuracy"] >= 0.9
    assert r["unchanged_action_after_revision"]["accuracy"] == 1
    assert r["adaptation_run"]["revisions"] == 1
    a = Genesis(db)
    try:
        from genesis.dashboard import summary
        assert a.ltm.get_meta("grounded_evaluation") == r
        assert summary(a.ltm)["grounded_evaluation"] == r
        assert a.ltm.get_meta("innate_grounding")["domain_knowledge"] == []
        assert a.ltm.knowledge(kind="grounded-concept", status="refuted")
        assert a.ltm.count("episodes") == r["episodes"]
        rows = a.ltm.get_meta("grounding/object-world-7")["rows"]
        assert len(rows) == 160
        assert all(row["object_id"].startswith("specimen-") for row in rows)
        assert all(set(row["features"].values()) <= {0, 1} for row in rows)
    finally:
        a.close()
    with pytest.raises(FileExistsError):
        run_grounded_learning(7, db=db)


def test_grounded_cli_keeps_memory_and_refuses_overwrite(tmp_path, capsys):
    from genesis.__main__ import main
    db = str(tmp_path / "cli.db")
    assert main(["grounded", "--db", db, "--steps", "4"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["training_interactions"] == 4
    assert main(["grounded", "--db", db, "--steps", "4"]) == 1
    assert "refused" in capsys.readouterr().err


def test_grounded_behavior_is_independent_of_hash_seed():
    import os
    import subprocess
    import sys
    code = """
import json, tempfile
from pathlib import Path
from genesis.agent import Genesis
from genesis.environments import ObjectWorld, sandboxed
from genesis.grounding import GroundedExplorer
with tempfile.TemporaryDirectory() as td:
    agent = Genesis(str(Path(td) / 'state.db'))
    learner = GroundedExplorer(agent, sandboxed(ObjectWorld(3), 100))
    report = learner.run(40)
    print(json.dumps([report, [learner.models[a].description() for a in learner.actions]], sort_keys=True))
    agent.close()
"""
    results = [subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("0", "17")]
    assert results[0] == results[1]
