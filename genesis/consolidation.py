"""Knowledge consolidation ("sleep").

Runs between bouts of activity. It never deletes episodes. It:

1. writes current beliefs (and serious alternatives) into semantic memory with
   provenance, confidence and the episodes that induced/support/contradict them;
2. replays stored experience against previously held beliefs and marks the ones
   that the record now contradicts as *refuted* (with the refuting episodes);
3. strengthens procedural memory with the shapes of newly validated rules;
4. derives candidate generalisations (e.g. "action a affects only p2") and only
   marks them validated when every supporting rule is validated;
5. recomputes associations, comparing observational vs interventional evidence
   to separate causal links from spurious correlations;
6. measures replay consistency of validated knowledge (forgetting check);
7. improves the library (genesis.library): promotes recurring sub-programs of
   validated rules to invented primitives, and trains the recognition model on
   dreams sampled from procedural memory plus replays of validated problems.
"""

from __future__ import annotations

import zlib
from collections import defaultdict

from . import library
from .adapters import SEQ_VARS, sequence_dream
from .expressions import Expr, var

STATUS_FROM_LEARNER = {"validated": "validated", "candidate": "candidate", "unknown": "hypothesis"}


def _provenance(lr) -> str:
    counts = defaultdict(int)
    for p in lr.provenance:
        counts[p] += 1
    return "induced from " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))


class Consolidation:
    def __init__(self, min_contrast: float = 0.5, max_spurious_contrast: float = 0.2):
        self.min_contrast = min_contrast
        self.max_spurious_contrast = max_spurious_contrast

    def run(self, agent) -> dict:
        ltm = agent.ltm
        summary = {"written": 0, "refuted": 0, "newly_validated": [], "generalizations": 0, "associations": 0}
        for key, lr in agent.learners.items():
            ltm.save_learner(key, agent.learner_env[key], {**lr.to_state(), "kind": agent.learner_kind[key]})
            self._beliefs(agent, key, lr, summary)
        summary["generalizations"] = self._frame_generalizations(agent)
        summary["associations"] = self._associations(agent)
        summary["replay_consistency"] = self._replay_consistency(agent)
        summary["library"] = self._sleep(agent)
        ltm.record_metric(agent.session, "replay_consistency", summary["replay_consistency"])
        ltm.record_metric(agent.session, "consolidation", summary["written"], summary)
        ltm.commit()
        return summary

    # ------------------------------------------------------------------ beliefs
    def _beliefs(self, agent, key, lr, summary) -> None:
        ltm = agent.ltm
        env_id = agent.learner_env[key]
        st = lr.status()
        prov = _provenance(lr)
        current = set()
        if st == "unpredictable":
            _, _, p_null = lr.best()
            stmt = f"no predictive rule found among compositions of innate primitives (max size {lr.cfg.max_size})"
            ltm.upsert_knowledge(key, "meta", stmt, None, p_null, "unpredictable", prov, env_id,
                                 n_support=len(lr.data), reason="null hypothesis dominates")
            current.add(stmt)
        else:
            post, _ = lr.posterior()
            classes = {}
            for h, p in post:
                classes.setdefault(h.probe_sig, [h, 0.0])[1] += p
            ranked = sorted(classes.values(), key=lambda t: -t[1])
            for rank, (h, mass) in enumerate(ranked[:3]):
                if rank > 0 and mass < 0.05:
                    break
                status = STATUS_FROM_LEARNER[st] if rank == 0 else ("candidate" if mass >= 0.3 else "hypothesis")
                stmt = str(h.expr)
                before = ltm.knowledge(key=key)
                prev = next((k.status for k in before if k.statement == stmt), None)
                n_bad = h.n_bad
                kid = ltm.upsert_knowledge(key, "rule", stmt, h.expr.to_json(), mass, status, prov, env_id,
                                           n_support=h.n_ok, n_contra=n_bad, forward_ok=h.fwd_ok,
                                           forward_bad=h.fwd_bad, reason=f"consolidation (rank {rank})")
                self._evidence(ltm, kid, h, lr)
                current.add(stmt)
                summary["written"] += 1
                if status == "validated" and prev != "validated":
                    ltm.record_shape(h.expr.shape(), True)
                    summary["newly_validated"].append(f"{key}: {stmt}")
        # Replay: beliefs no longer held are checked against the stored record.
        for item in ltm.knowledge(key=key):
            if item.statement in current or item.status in ("refuted", "dormant") or item.expr is None:
                continue
            e = Expr.from_json(item.expr)
            contra = [lr.episodes[i] for i, (inp, y) in enumerate(lr.data) if e.evaluate(inp, True) != y]
            if contra:
                ltm.add_evidence(item.id, contra[:5], "refuted_by")
                ltm.set_status(item.id, "refuted", f"contradicted by {len(contra)} stored episodes on replay")
                if item.status == "validated":
                    ltm.record_shape(e.shape(), False)
                summary["refuted"] += 1
            else:
                ltm.set_status(item.id, "dormant", "consistent with all evidence but displaced by simpler beliefs")

    @staticmethod
    def _sleep(agent) -> dict:
        """Abstraction and dreaming, recomputed from the current validated knowledge (see genesis.library)."""
        if not getattr(agent, "library_learning", False):
            return {}
        ltm = agent.ltm
        rules = [(k.key, Expr.from_json(k.expr)) for k in ltm.knowledge(status="validated", kind="rule")
                 if k.expr is not None]
        successes = ltm.shape_successes()
        shapes = [s for s, n in sorted(successes.items(), key=lambda t: (-t[1], t[0])) if n > 0]
        # Nothing new was validated since the last sleep: the library would be the same.
        digest = zlib.crc32(repr(([(k, str(e)) for k, e in rules], shapes)).encode())
        previous = ltm.get_meta("library")
        if previous and previous.get("digest") == digest:
            return {"unchanged": True}
        replays = []
        for key in sorted(agent.learners):
            lr = agent.learners[key]
            top = lr.best()[0]
            if top is not None and lr.status() == "validated":
                replays.append((top.expr.shape(), [y for _, y in lr.data]))
        lib = library.sleep(rules, shapes, replays, {"sequence": (SEQ_VARS, sequence_dream)},
                            seed=agent.seed * 7919, dreamt=(previous or {}).get("dreamt"))
        lib.update(session=agent.session, digest=digest)
        ltm.set_meta("library", lib)
        out = {"inventions": [d["statement"] for d in lib["inventions"]], "dreams": lib["dreams"],
               "replays": lib["replays"]}
        ltm.record_metric(agent.session, "library", len(lib["inventions"]), out)
        return out

    @staticmethod
    def _evidence(ltm, kid, h, lr) -> None:
        induced = [e for e in lr.episodes[: h.born_n]]
        sup, con = [], []
        for i in range(h.born_n, len(lr.data)):
            inp, y = lr.data[i]
            (sup if h.expr.evaluate(inp, True) == y else con).append(lr.episodes[i])
        ltm.add_evidence(kid, induced, "induced_from")
        ltm.add_evidence(kid, sup, "supports")
        ltm.add_evidence(kid, con, "contradicts")

    # --------------------------------------------------------- generalisations
    @staticmethod
    def _frame_generalizations(agent) -> int:
        groups = defaultdict(dict)
        for key, lr in agent.learners.items():
            if agent.learner_kind.get(key) == "lab":
                env_id, action, prop = key.rsplit("/", 2)
                groups[(env_id, action)][prop] = lr
        n = 0
        for (env_id, action), learners in groups.items():
            affected, all_valid = [], True
            for prop, lr in sorted(learners.items()):
                top, _, _ = lr.best()
                all_valid &= lr.status() == "validated"
                if top is None or top.expr != var(prop):
                    affected.append(prop)
            stmt = f"{action} can only change {{{', '.join(affected)}}}; all other properties are left unchanged"
            status = "validated" if all_valid else "candidate"
            agent.ltm.upsert_knowledge(f"{env_id}/{action}", "generalization", stmt, None,
                                       1.0 if all_valid else 0.5, status,
                                       f"consolidation: derived from {len(learners)} per-property rules", env_id,
                                       reason="frame generalisation")
            n += 1
        return n

    # ------------------------------------------------------------ associations
    def _associations(self, agent) -> int:
        """Contrast Δ = P(change | q=v) - P(change | q≠v), computed separately on
        passive observations and on the agent's own interventions. A link that is
        strong observationally but vanishes under intervention is spurious."""
        n_written = 0
        for key, lr in agent.learners.items():
            if agent.learner_kind.get(key) != "lab":
                continue
            env_id, action, prop = key.rsplit("/", 2)
            top, _, _ = lr.best()
            cond_var = top.expr.condition_variable() if top else None
            cond_val = top.expr.args[1].value if cond_var else None
            rows = {"observation": [], "experiment": []}
            for (inp, y), prov in zip(lr.data, lr.provenance):
                if prov in rows:
                    rows[prov].append((inp, int(y != inp[prop])))

            def contrast(prov, q, v):
                on = [c for inp, c in rows[prov] if inp[q] == v]
                off = [c for inp, c in rows[prov] if inp[q] != v]
                if len(on) < 3 or len(off) < 3:
                    return None, len(on)
                return sum(on) / len(on) - sum(off) / len(off), len(on)

            conds = {(q, v) for inp, _ in rows["observation"] for q, v in inp.items()}
            for q, v in sorted(conds):
                d_obs, n_obs = contrast("observation", q, v)
                if d_obs is None or d_obs < self.min_contrast:
                    continue
                d_int, n_int = contrast("experiment", q, v)
                if d_int is None:
                    kind = "correlational"
                elif d_int >= self.min_contrast:
                    kind = "causal"
                elif d_int < self.max_spurious_contrast:
                    kind = "spurious"
                else:
                    kind = "inconclusive"
                agent.ltm.upsert_association(
                    f"{env_id}:{q}=={v}", f"{action} changes {prop}", kind, d_obs, n_obs,
                    {"contrast_observational": round(d_obs, 3),
                     "contrast_interventional": None if d_int is None else round(d_int, 3),
                     "n_observational": n_obs, "n_interventional": n_int,
                     "matches_validated_rule_condition": (q, v) == (cond_var, cond_val)})
                n_written += 1
        return n_written

    @staticmethod
    def _replay_consistency(agent) -> float:
        ok = tot = 0
        for item in agent.ltm.knowledge(status="validated", kind="rule"):
            lr = agent.learners.get(item.key)
            if lr is None:
                continue
            e = Expr.from_json(item.expr)
            for inp, y in lr.data:
                ok += e.evaluate(inp, True) == y
                tot += 1
        return ok / tot if tot else 1.0
