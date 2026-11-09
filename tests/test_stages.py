import hashlib

import pytest

from genesis import dashboard
from genesis import self_improvement as SI
from genesis.adapters import SEQ_VARS, LabAdapter, SequenceAdapter, sequence_rollout_probes
from genesis.agent import Genesis, validate_config
from genesis.environments import CausalWorld, SequenceWorld, sandboxed
from genesis.experiments import run_autonomy, run_robustness_benchmark
from genesis.induction import InductionConfig, RuleLearner
from genesis.innate import DEFAULT_SAFETY
from genesis.memory import LongTermMemory


def _feed(lr, terms):
    for t in range(3, len(terms)):
        lr.observe({"x[t-1]": terms[t - 1], "x[t-2]": terms[t - 2], "x[t-3]": terms[t - 3], "t": t}, terms[t])


# ------------------------------------------------------ robustness and depth
def test_robustness_benchmark_deepening_and_soundness():
    r = run_robustness_benchmark([100], ("noise_20", "deep", "deep_without_deepening"))
    assert r["deep"]["rule_streams_validated"] > r["deep_without_deepening"]["rule_streams_validated"]
    assert r["deep"]["deep_searches"] > 0 == r["deep_without_deepening"]["deep_searches"]
    assert r["noise_20"]["corrupted_observations"] > 0
    assert all(c["false_validation_rate_on_noise"] == 0 for c in r.values())
    assert all(c["validated_rule_extrapolation_accuracy"] >= 0.9 for c in r.values())


# ------------------------------------------------- procedural memory speed-up
def test_remembered_shapes_cut_search_cost():
    terms = [4 + 9 * t for t in range(14)]
    cold = RuleLearner("a", SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes)
    warm = RuleLearner("b", SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes,
                       shape_library=lambda: ["add(V,C)"])
    _feed(cold, terms)
    _feed(warm, terms)
    assert cold.status() == warm.status() == "validated"
    assert warm.library_searches > 0
    assert warm.expressions_evaluated * 5 < cold.expressions_evaluated


def test_unknown_shape_falls_back_to_full_search():
    terms = [2, 5]
    while len(terms) < 16:
        terms.append(terms[-1] + terms[-2])
    lr = RuleLearner("f", SEQ_VARS, InductionConfig(max_size=5), probe_fn=sequence_rollout_probes,
                     shape_library=lambda: ["mul(V,C)"])
    _feed(lr, terms)
    assert lr.status() == "validated" and lr.full_searches > 0


# ---------------------------------------------------- budget-aware attention
def test_focus_engages_only_when_budget_is_tight(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"))
    ad = SequenceAdapter(agent, sandboxed(SequenceWorld(3), 100))
    cands = ad.candidates()
    utils = [1.0] * len(cands)
    assert agent._focus("sequence", cands, utils, budget_left=10_000) is None
    assert agent._focus("sequence", cands, utils, budget_left=20) is not None


# ----------------------------------------------------------------- stage 6
def test_progress_allocation_avoids_noise_world():
    prog = run_autonomy(1, 150, "progress")
    rr = run_autonomy(1, 150, "round_robin")
    assert prog["noise_share"] < rr["noise_share"]
    assert prog["knowledge_score"] >= rr["knowledge_score"]
    assert prog["objectives"]["achieved"] > 0


def test_objectives_are_persisted_and_resolved(tmp_path):
    from genesis.autonomy import Explorer
    agent = Genesis(str(tmp_path / "m.db"), seed=2)
    ad = LabAdapter(agent, sandboxed(CausalWorld(2), 500))
    Explorer(agent, {"lab": ad}).run(60)
    objs = agent.ltm.objectives()
    assert objs and all(o["env_id"] == "causal-world-2" for o in objs)
    achieved = [o for o in objs if o["status"] == "achieved"]
    assert achieved and all(agent.learners[o["key"]].status() == "validated" for o in achieved)


# ----------------------------------------------------------------- stage 7
def test_config_allowlist_protects_safety():
    with pytest.raises(PermissionError):
        validate_config({"sequence.max_candidates": 10 ** 9})
    with pytest.raises(PermissionError):
        validate_config({"safety.max_interactions_per_session": 10 ** 9})
    with pytest.raises(ValueError):
        validate_config({"sequence.validation_k": 1})


def test_config_is_applied(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"), config={"sequence.validation_k": 3, "agent.default_resolution_cost": 20})
    assert agent.configs["sequence"].validation_k == 3
    assert agent.default_resolution_cost == 20
    assert agent.configs["sequence"].max_candidates == DEFAULT_SAFETY.max_candidates_per_induction


def test_evaluation_is_reproducible_and_isolated(tmp_path):
    fac = lambda s: sandboxed(SequenceWorld(s), 10 ** 6)  # noqa: E731
    a = SI.evaluate({}, fac, [7], budget=40)
    b = SI.evaluate({}, fac, [7], budget=40)
    assert a["score"] == b["score"] and a["observed_false_validations"] == b["observed_false_validations"]


def test_deployment_requires_human_approval_and_can_roll_back(tmp_path):
    db = str(tmp_path / "m.db")
    Genesis(db).close()
    ltm = LongTermMemory(db)
    rejected = ltm.add_version({"sequence.validation_k": 5}, None, "x", {}, None, "rejected-trial",
                               DEFAULT_SAFETY.digest())
    pending = ltm.add_version({"sequence.validation_k": 3}, None, "y", {}, {}, "pending-approval",
                              DEFAULT_SAFETY.digest())
    ltm.close()
    with pytest.raises(PermissionError):
        SI.approve(db, rejected, "alice")
    with pytest.raises(PermissionError):
        SI.approve(db, pending, "")
    g = Genesis(db)
    assert g.config == {}
    g.close()
    SI.approve(db, pending, "alice")
    g = Genesis(db)
    assert g.config == {"sequence.validation_k": 3} and g.configs["sequence"].validation_k == 3
    assert next(v for v in g.ltm.versions() if v["id"] == pending)["approved_by"] == "alice"
    g.close()
    SI.rollback(db)
    g = Genesis(db)
    assert g.config == {}
    g.close()


def test_limitations_come_from_records(tmp_path):
    agent = Genesis(str(tmp_path / "m.db"), seed=0)
    agent.explore_sequences(sandboxed(SequenceWorld(0), 400), 300)
    lim = SI.limitations(agent)
    assert lim["n_identified_from_records"] >= len(lim["judged_unpredictable"]) > 0
    assert all(agent.learners[k].status() == "unpredictable" for k in lim["judged_unpredictable"])


# --------------------------------------------------------------- dashboard
def test_dashboard_api_is_read_only(tmp_path):
    db = str(tmp_path / "m.db")
    agent = Genesis(db, seed=0)
    sb = sandboxed(CausalWorld(0), 500)
    agent.watch_lab(sb, 30)
    agent.experiment_lab(sb, 40)
    agent.close()
    digest = hashlib.sha256(open(db, "rb").read()).hexdigest()
    s = dashboard.api(db, "/api/summary", {})
    assert s["counts"]["episodes"] == 70 and "stage" in s
    k = dashboard.api(db, "/api/knowledge", {"status": "validated"})
    assert k and all(x["status"] == "validated" for x in k)
    ex = dashboard.api(db, f"/api/explain/{k[0]['id']}", {})
    assert ex["item"]["id"] == k[0]["id"] and "evidence" in ex
    perf = dashboard.api(db, "/api/performance", {})
    assert perf["accuracy_history"] and perf["accuracy_history"][-1]["accuracy"] > perf["accuracy_history"][0]["accuracy"]
    for name in ("episodes", "failures", "objectives", "versions", "procedures", "associations"):
        assert dashboard.api(db, f"/api/{name}", {}) is not None
    assert dashboard.api(db, "/api/nope", {}) is None
    ro = LongTermMemory(db, read_only=True)
    with pytest.raises(Exception):
        ro.set_meta("x", 1)
        ro.db.commit()
    ro.close()
    assert hashlib.sha256(open(db, "rb").read()).hexdigest() == digest


def test_behaviour_independent_of_hash_seed(tmp_path):
    """Evaluations must be reproducible across processes (string hashing is randomized per process)."""
    import os
    import subprocess
    import sys
    code = ("import sys; from genesis.environments import SequenceWorld, sandboxed; "
            "from genesis import self_improvement as SI; "
            "print(SI.evaluate({}, lambda s: sandboxed(SequenceWorld(s), 10**6), [9102], budget=100)['score'])")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = set()
    for h in ("0", "2"):
        env = {**os.environ, "PYTHONHASHSEED": h, "PYTHONPATH": root}
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                                check=True).stdout)
    assert len(outs) == 1
