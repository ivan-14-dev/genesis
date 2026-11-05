"""Reproducible experiments: benchmarks, baselines, controls and the demonstration.

The harness is the only code that touches hidden rules (``world._*`` methods),
and only to *score* the agent. The agent only ever sees a :class:`Sandbox`.
"""

from __future__ import annotations

import os
import random
import statistics
import tempfile
import time
from contextlib import contextmanager

from .adapters import LabAdapter, SequenceAdapter
from .agent import Genesis
from .autonomy import Explorer
from .environments import CausalWorld, SequenceWorld, sandboxed


@contextmanager
def _tmpdb():
    d = tempfile.mkdtemp(prefix="genesis-")
    try:
        yield os.path.join(d, "memory.db")
    finally:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        os.rmdir(d)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _first_validation(lr):
    return next((h["n"] for h in lr.status_history if h["status"] == "validated"), None)


# ===================================================================== stage 1
def evaluate_sequences(agent: Genesis, world: SequenceWorld, horizon: int = 10) -> dict:
    rows = []
    for sid in world.stream_ids():
        key = f"{world.env_id}/{sid}"
        lr = agent.learners.get(key)
        future = world._future(sid, horizon)
        pred = agent.rollout_sequence(key, horizon)
        window, _ = agent.recall_sequence(key)
        rep = [window[-1] if window else None] * horizon
        w2 = list(window)
        lin = []
        for _ in range(horizon):
            v = 2 * w2[-1] - w2[-2] if len(w2) >= 2 else None
            lin.append(v)
            w2.append(v if v is not None else 0)
        top, mass, _ = lr.best() if lr else (None, 0, 1)
        rows.append({
            "stream": sid, "family": world._family(sid), "is_noise": world._family(sid) == "noise",
            "status": lr.status() if lr else "unknown", "belief": str(top.expr) if top else None,
            "observations": len(lr.data) + 3 if lr else 0,
            "observations_to_validation": (_first_validation(lr) + 3) if lr and _first_validation(lr) else None,
            "extrapolation_accuracy": sum(a == b for a, b in zip(pred, future)) / horizon,
            "baseline_repeat_last": sum(a == b for a, b in zip(rep, future)) / horizon,
            "baseline_linear": sum(a == b for a, b in zip(lin, future)) / horizon,
        })
    rules = [r for r in rows if not r["is_noise"]]
    noise = [r for r in rows if r["is_noise"]]
    return {
        "streams": rows,
        "rule_streams_validated": sum(r["status"] == "validated" for r in rules) / len(rules),
        "rule_extrapolation_accuracy": _mean([r["extrapolation_accuracy"] for r in rules]),
        "validated_rule_extrapolation_accuracy": _mean([r["extrapolation_accuracy"] for r in rules
                                                        if r["status"] == "validated"]),
        "baseline_repeat_last": _mean([r["baseline_repeat_last"] for r in rules]),
        "baseline_linear": _mean([r["baseline_linear"] for r in rules]),
        "noise_flagged_unpredictable": _mean([r["status"] == "unpredictable" for r in noise]) if noise else None,
        "false_validation_rate": _mean([r["status"] == "validated" for r in noise]) if noise else None,
        "observations_to_validation": _mean([r["observations_to_validation"] for r in rules]),
        "observations_on_noise": sum(r["observations"] for r in noise),
    }


def run_sequence_benchmark(seed: int, budget: int = 400, strategy: str = "curiosity",
                           stop_when_idle: bool = True, db: str | None = None) -> dict:
    with _tmpdb() as tmp:
        agent = Genesis(db or tmp, seed=seed)
        world = SequenceWorld(seed)
        sb = sandboxed(world, budget + 1)
        t0 = time.process_time()
        rep = agent.explore_sequences(sb, budget, strategy, stop_when_idle=stop_when_idle)
        res = evaluate_sequences(agent, world)
        res.update(interactions=rep.steps, stopped_because=rep.stopped_because,
                   cpu_seconds=round(time.process_time() - t0, 3),
                   expressions_evaluated=sum(lr.expressions_evaluated for lr in agent.learners.values()),
                   calibration=agent.metacog.calibration(), memory_bytes=agent.ltm.size_bytes())
        agent.ltm.close()
        return res


# ================================================================== stage 2-3
def lab_eval_set(world: CausalWorld, n: int, seed: int, lo: int, hi: int, natural: bool = False) -> list:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = {p: rng.randint(lo, hi) for p in world.props}
        if natural:
            s[world._confound[1]] = s[world._confound[0]]
        a = rng.choice(world.actions)
        out.append((s, a, world._true_next(s, a)))
    return out


def lab_accuracy(agent: Genesis, world: CausalWorld, evalset: list, exclude: set | None = None) -> float:
    ok = n = 0
    for s, a, nxt in evalset:
        if exclude and (tuple(s.values()), a) in exclude:
            continue
        pred, _ = agent.predict_transition(world.env_id, world.props, s, a)
        ok += pred == nxt
        n += 1
    return ok / n if n else None


def tabular_accuracy(agent: Genesis, world: CausalWorld, evalset: list) -> float:
    """Memorisation baseline given exactly the agent's experiences."""
    table = {}
    for key, lr in agent.learners.items():
        if not key.startswith(world.env_id + "/"):
            continue
        _, a, p = key.rsplit("/", 2)
        for inp, y in lr.data:
            table.setdefault((tuple(inp[q] for q in world.props), a), dict(inp))[p] = y
    ok = 0
    for s, a, nxt in evalset:
        ok += table.get((tuple(s[q] for q in world.props), a), s) == nxt
    return ok / len(evalset)


def experienced(agent: Genesis, world: CausalWorld) -> set:
    seen = set()
    for key, lr in agent.learners.items():
        if key.startswith(world.env_id + "/"):
            a = key.rsplit("/", 2)[1]
            seen |= {(tuple(inp[q] for q in world.props), a) for inp, _ in lr.data}
    return seen


def confound_resolved(agent: Genesis, world: CausalWorld) -> dict:
    """Did the agent attribute each conditional rule to the true cause, not the correlate?"""
    out = {}
    for a, r in world._rules.items():
        if "cond" not in r:
            continue
        lr = agent.learners.get(f"{world.env_id}/{a}/{r['target']}")
        top, mass, _ = lr.best() if lr else (None, 0, 1)
        cv = top.expr.condition_variable() if top else None
        out[a] = {"true_condition": f"{r['cond'][0]}=={r['cond'][1]}",
                  "believed": str(top.expr) if top else None, "status": lr.status() if lr else "unknown",
                  "correct_cause": cv == r["cond"][0] and top.expr.args[1].value == r["cond"][1] if cv else False,
                  "confounded": r["cond"][0] == world._confound[0]}
    return out


def run_lab_benchmark(seed: int, watch: int = 60, budget: int = 120, strategy: str = "curiosity",
                      checkpoints=(0, 5, 10, 20, 40, 80, 120), db: str | None = None, keep_agent: bool = False):
    with _tmpdb() as tmp:
        path = db or tmp
        agent = Genesis(path, seed=seed)
        world = CausalWorld(seed)
        sb = sandboxed(world, watch + budget + 1000)
        held = lab_eval_set(world, 400, seed + 991, 0, world.n_values - 1)
        natural = lab_eval_set(world, 400, seed + 992, 0, world.n_values - 1, natural=True)
        transfer = lab_eval_set(world, 400, seed + 993, world.n_values, world.n_values + 8)
        t0 = time.process_time()
        before = lab_accuracy(agent, world, held)
        agent.watch_lab(sb, watch)
        after_watch = {"interventional_heldout": lab_accuracy(agent, world, held),
                       "natural_heldout": lab_accuracy(agent, world, natural),
                       "confound": confound_resolved(agent, world)}
        adapter = LabAdapter(agent, sb)
        curve, done = [], 0
        for cp in checkpoints:
            if cp > done:
                agent.live(adapter, cp - done, strategy, stop_when_idle=False)
                done = cp
            curve.append({"experiments": done, "heldout_unseen": lab_accuracy(agent, world, held,
                                                                              experienced(agent, world))})
        to_perfect = next((c["experiments"] for c in curve if c["heldout_unseen"] == 1.0), None)
        res = {
            "seed": seed, "strategy": strategy, "hidden_rules": world._describe_rules(),
            "accuracy_before_experience": before,
            "after_passive_watching": after_watch,
            "learning_curve": curve,
            "experiments_to_perfect_heldout": to_perfect,
            "final_heldout_unseen": curve[-1]["heldout_unseen"],
            "final_transfer_out_of_range": lab_accuracy(agent, world, transfer),
            "tabular_baseline_heldout": tabular_accuracy(agent, world, held),
            "tabular_baseline_transfer": tabular_accuracy(agent, world, transfer),
            "no_change_baseline_heldout": sum(s == n for s, _, n in held) / len(held),
            "confound_after_experiments": confound_resolved(agent, world),
            "associations": agent.ltm.associations(),
            "cpu_seconds": round(time.process_time() - t0, 3),
            "memory_bytes": agent.ltm.size_bytes(),
            "calibration": agent.metacog.calibration(),
        }
        if keep_agent:
            return res, agent, world, sb
        agent.ltm.close()
        return res


# ===================================================================== stage 6
def run_autonomy(seed: int, budget: int = 300, strategy: str = "progress", agent: Genesis | None = None) -> dict:
    """Four worlds, one budget: the agent decides where to spend it.

    Knowledge score (harness-only) = mean over the three learnable worlds of
    rule-stream extrapolation accuracy (sequences) or held-out transition
    accuracy (labs). The fourth world is pure noise: nothing can be learned.
    """
    with _tmpdb() as tmp:
        own = agent is None
        agent = agent or Genesis(tmp, seed=seed)
        seq, noise = SequenceWorld(seed), SequenceWorld(seed + 500, n_rule_streams=0, n_noise_streams=10)
        labs = [CausalWorld(seed), CausalWorld(seed + 700)]
        sbs = {w.env_id: sandboxed(w, budget + 50) for w in [seq, noise, *labs]}
        adapters = {"sequences": SequenceAdapter(agent, sbs[seq.env_id]),
                    "noise": SequenceAdapter(agent, sbs[noise.env_id]),
                    "lab-A": LabAdapter(agent, sbs[labs[0].env_id]),
                    "lab-B": LabAdapter(agent, sbs[labs[1].env_id])}
        t0 = time.process_time()
        explorer = Explorer(agent, adapters)
        res = explorer.run(budget, strategy)
        seq_eval = evaluate_sequences(agent, seq)
        lab_scores = [lab_accuracy(agent, w, lab_eval_set(w, 300, seed + 991 + i, 0, w.n_values - 1))
                      for i, w in enumerate(labs)]
        res.update(seed=seed, sequences=seq_eval["rule_extrapolation_accuracy"], labs=lab_scores,
                   knowledge_score=round((seq_eval["rule_extrapolation_accuracy"] + sum(lab_scores)) / 3, 4),
                   noise_share=round(res["allocation"]["noise"] / max(res["interactions"], 1), 4),
                   objectives={st: len(agent.ltm.objectives(st)) for st in ("open", "achieved", "abandoned")},
                   cpu_seconds=round(time.process_time() - t0, 3))
        if own:
            agent.ltm.close()
        return res


# ===================================================================== stage 5
def run_planning(agent: Genesis, sb, world: CausalWorld, n_tasks: int = 20) -> dict:
    planned = [agent.pursue_goal(sb, 1000 + i, "plan") for i in range(n_tasks)]
    rnd = [agent.pursue_goal(sb, 1000 + i, "random") for i in range(n_tasks)]
    return {"plan_success": _mean([r["success"] for r in planned]),
            "random_success": _mean([r["success"] for r in rnd]),
            "plan_mean_steps": _mean([r["steps"] for r in planned if r["success"]]),
            "random_mean_steps": _mean([r["steps"] for r in rnd if r["success"]]),
            "examples": planned[:3]}


# ===================================================================== stage 4
def run_memory_benchmark(seed: int) -> dict:
    with _tmpdb() as path:
        res, agent, world, _ = run_lab_benchmark(seed, db=path, keep_agent=True)
        held = lab_eval_set(world, 400, seed + 991, 0, world.n_values - 1)
        acc_before = lab_accuracy(agent, world, held)
        agent.close()
        reborn = Genesis(path, seed=seed)
        acc_reload = lab_accuracy(reborn, world, held)
        world_b = CausalWorld(seed + 5000)
        sb_b = sandboxed(world_b, 400)
        reborn.watch_lab(sb_b, 60)
        reborn.experiment_lab(sb_b, 120, stop_when_idle=False)
        acc_b = lab_accuracy(reborn, world_b, lab_eval_set(world_b, 400, seed + 994, 0, world_b.n_values - 1))
        acc_after_b = lab_accuracy(reborn, world, held)
        out = {"accuracy_world_a": acc_before, "accuracy_world_a_after_reload": acc_reload,
               "session_after_reload": reborn.session, "accuracy_world_b": acc_b,
               "accuracy_world_a_after_learning_b": acc_after_b,
               "forgetting": round(acc_reload - acc_after_b, 4),
               "replay_consistency": reborn.ltm.metrics("replay_consistency")[-1]["value"]}
        reborn.ltm.close()
        return out


# ================================================================ meta-learning
def run_procedural_transfer(seed: int, n_worlds: int = 6, procedural_prior: bool = True) -> dict:
    with _tmpdb() as path:
        agent = Genesis(path, seed=seed, procedural_prior=procedural_prior)
        per_world, evals, rule_evals, cpu, acc, lib = [], [], [], [], [], []
        for i in range(n_worlds):
            world = SequenceWorld(seed * 100 + i)
            keys = [f"{world.env_id}/{sid}" for sid in world.stream_ids()]
            t0 = time.process_time()
            agent.explore_sequences(sandboxed(world, 401), 400)
            cpu.append(round(time.process_time() - t0, 3))
            ev = evaluate_sequences(agent, world)
            per_world.append(ev["observations_to_validation"])
            acc.append(ev["rule_extrapolation_accuracy"])
            lrs = [agent.learners[k] for k in keys if k in agent.learners]
            rule_lrs = [agent.learners[f"{world.env_id}/{sid}"] for sid in world.stream_ids()
                        if world._family(sid) != "noise" and f"{world.env_id}/{sid}" in agent.learners]
            evals.append(sum(lr.expressions_evaluated for lr in lrs))
            rule_evals.append(round(sum(lr.expressions_evaluated for lr in rule_lrs) / max(len(rule_lrs), 1)))
            searches = sum(lr.library_searches + lr.full_searches for lr in lrs)
            lib.append(round(sum(lr.library_searches for lr in lrs) / searches, 4) if searches else 0.0)
        agent.ltm.close()
        late = slice(n_worlds // 2, None)
        return {"observations_to_validation_per_world": per_world,
                "expressions_evaluated_per_world": evals, "cpu_seconds_per_world": cpu,
                "search_cost_per_rule_stream_per_world": rule_evals,
                "accuracy_per_world": acc, "library_search_share_per_world": lib,
                "late_worlds_mean": _mean(per_world[late]),
                "late_worlds_expressions": _mean(evals[late]), "late_worlds_cpu": _mean(cpu[late]),
                "late_worlds_search_cost_per_rule_stream": _mean(rule_evals[late]),
                "late_worlds_accuracy": _mean(acc[late])}


# ============================================================ word learning
# Test data written by the designers for evaluation only. The agent never sees it as a list: a simulated
# user sends these sentences one at a time and answers GENESIS's questions truthfully. No sentence contains an
# innate command keyword, so every success must come from words learned during the run.
# Slang / jargon chosen to lie *outside* the innate language faculty, so that only learned words can help.
WORD_CORPUS = {
    "knowledge": (["balance tes trouvailles", "déballe ton butin", "crache tes pépites"],
                  ["tes trouvailles stp", "ton butin ?", "tes pépites du jour"]),
    "stage": (["ton palier", "t'as grimpé jusqu'où", "ta grimpette"],
              ["palier actuel ?", "grimpé jusqu'où", "ta grimpette du jour"]),
    "limitations": (["tes bugs", "tes galères", "ce qui coince chez toi"],
                    ["tes galères principales", "des bugs ?", "ça coince ?"]),
    "questions": (["ce qui t'intrigue", "tes énigmes", "ce qui te titille"],
                  ["tes énigmes actuelles", "ça t'intrigue ?", "titillé par quoi ?"]),
    "help": (["je suis paumé", "je capte rien", "comment ça marche"],
             ["un peu paumé", "je capte pas", "ça marche comment ?"]),
}
WORD_CONTROL = ["il pleut dehors", "quelle heure est-il", "j'aime le chocolat", "ferme la fenêtre",
                "le chat dort", "achète du pain"]
WORD_COMMANDS = {"knowledge": "que sais-tu ?", "stage": "stade", "limitations": "limites",
                 "questions": "questions", "help": "aide"}


def run_word_learning(seed: int, passes: int = 4) -> dict:
    """Interactive word learning with a truthful simulated user, then evaluation on unseen sentences."""
    from .chat import ChatEngine
    from .lexicon import ASK_POSTERIOR
    rng = random.Random(seed)
    with _tmpdb() as path:
        agent = Genesis(path, seed=seed)
        eng = ChatEngine(agent)
        cid = eng.new_conversation("fr")["id"]
        say = lambda t: eng.send(cid, t, "fr")["reply"]["meta"]
        train = [(i, u) for i, (tr, _) in WORD_CORPUS.items() for u in tr]
        log = []
        exchanges = 0
        for p in range(passes):
            order = train[:]
            rng.shuffle(order)
            for intent, u in order:
                m = say(u)
                exchanges += 1
                if m.get("interpreted"):
                    ok = m["intent"] == intent
                    log.append((p, "acted-correct" if ok else "acted-wrong"))
                    if not ok:
                        say("non")
                        say(WORD_COMMANDS[intent])
                        exchanges += 2
                elif m["intent"] == "clarify":
                    ok = m["proposed"] == intent
                    log.append((p, "asked-correct" if ok else "asked-wrong"))
                    say("oui" if ok else "non")
                    exchanges += 1
                    if not ok:
                        say(WORD_COMMANDS[intent])
                        exchanges += 1
                elif m["intent"] in ("not-understood", "unknown-topic"):
                    log.append((p, "not-understood"))
                    say(WORD_COMMANDS[intent])
                    exchanges += 1
                else:
                    log.append((p, "innate"))
        held = [(i, u) for i, (_, ho) in WORD_CORPUS.items() for u in ho]
        res = {"correct": 0, "would_act_correct": 0, "would_act_wrong": 0, "unknown": 0}
        for intent, u in held:
            g = eng.lexicon.interpret(u)
            if g is None:
                res["unknown"] += 1
                continue
            res["correct"] += g.intent == intent
            if g.act:
                res["would_act_correct" if g.intent == intent else "would_act_wrong"] += 1
        control = [eng.lexicon.interpret(u) for u in WORD_CONTROL]
        per_pass = {}
        for p, kind in log:
            per_pass.setdefault(p, {}).setdefault(kind, 0)
            per_pass[p][kind] += 1
        out = {"seed": seed, "train_sentences": len(train), "user_exchanges": exchanges,
               "per_pass": [per_pass.get(p, {}) for p in range(passes)],
               "vocabulary": eng.lexicon.size(),
               "validated_mappings": sum(1 for e in eng.lexicon.entries() if e["status"] == "validated"),
               "heldout_n": len(held), "heldout_accuracy": round(res["correct"] / len(held), 4),
               "heldout_unknown": res["unknown"],
               "heldout_act_precision": round(res["would_act_correct"] /
                                              max(res["would_act_correct"] + res["would_act_wrong"], 1), 4),
               "heldout_act_rate": round((res["would_act_correct"] + res["would_act_wrong"]) / len(held), 4),
               "control_n": len(control),
               "control_would_act": sum(1 for g in control if g and g.act),
               "control_would_ask": sum(1 for g in control if g and not g.act and g.posterior >= ASK_POSTERIOR),
               "chance_accuracy": round(1 / len(eng.lexicon.intents), 4),
               "acted_wrong_during_training": sum(1 for _, k in log if k == "acted-wrong")}
        agent.close()
        return out


# Designer-written test sentences for the innate FR/EN faculty (written after the lexicon, never used to tune
# it beyond fixing genuine parsing bugs). (sentence, expected intent, category, needs a prior sequence)
LANGUAGE_BENCH = [
    ("Salut GENESIS", "intro", "fr", False), ("Hey there!", "intro", "en", False),
    ("Tu es qui au juste ?", "intro", "fr", False), ("Who are you exactly?", "intro", "en", False),
    ("Qu'as-tu appris jusqu'ici ?", "knowledge", "fr", False), ("What have you learnt so far?", "knowledge", "en", False),
    ("Montre-moi tes connaissances", "knowledge", "fr", False), ("Which rules did you discover?", "knowledge", "en", False),
    ("Tu en es à quel stade ?", "stage", "fr", False), ("How far along is your development?", "stage", "en", False),
    ("Parle-moi de ta mémoire", "stage", "fr", False), ("Show me your progress", "stage", "en", False),
    ("Quels sont tes points faibles et tes limites ?", "limitations", "fr", False),
    ("What are your weaknesses?", "limitations", "en", False),
    ("Qu'est-ce que tu ne sais pas encore ?", "questions", "fr", False),
    ("What are you curious about?", "questions", "en", False), ("Quels sont tes objectifs ?", "questions", "fr", False),
    ("Quels mots t'ai-je appris ?", "vocabulary", "fr", False), ("Show me your vocabulary", "vocabulary", "en", False),
    ("Que peux-tu faire pour moi ?", "help", "fr", False), ("What can you do?", "help", "en", False),
    ("Je ne comprends pas comment t'utiliser", "help", "fr", False), ("I don't understand", "help", "en", False),
    ("Va explorer un peu", "explore", "fr", False), ("Go and run some experiments", "explore", "en", False),
    ("Parles-tu français ?", "languages", "fr", False), ("Do you speak English?", "languages", "en", False),
    ("Qu'est-ce qu'une croyance ?", "define", "fr", False), ("What is a stage?", "define", "en", False),
    ("Que signifie hypothèse ?", "define", "fr", False), ("What does sequence mean?", "define", "en", False),
    ("Comment vas-tu ?", "smalltalk", "fr", False), ("How are you today?", "smalltalk", "en", False),
    ("Merci beaucoup !", "thanks", "fr", False), ("Thank you so much", "thanks", "en", False),
    ("À bientôt", "bye", "fr", False), ("Goodbye!", "bye", "en", False),
    ("Prédis les 3 prochains", "predict", "fr", True), ("What comes next?", "predict", "en", True),
    ("Continue avec 27 31", "sequence", "fr", True), ("Pourquoi ?", "explain", "fr", True),
    ("Explain your rule", "explain", "en", True), ("Justifie ta réponse", "explain", "fr", True),
    ("N'explore surtout pas", "declined", "negation", False), ("Don't explore", "declined", "negation", False),
    ("Ne prédis rien", "declined", "negation", True), ("Please do not predict", "declined", "negation", True),
    ("Qu'est-ce que tu connais ?", "knowledge", "typo", False),
    ("Quelles sont tes limitatons ?", "limitations", "typo", False),
    ("What do you knwo?", "knowledge", "typo", False), ("Shwo me your vocabulary", "vocabulary", "typo", False),
    ("Expliqe ta règle", "explain", "typo", True), ("Whats your curiousity about?", "questions", "typo", False),
    ("Quelle est la capitale de l'Italie ?", "unknown-topic", "out-of-scope", False),
    ("Who won the football match yesterday?", "unknown-topic", "out-of-scope", False),
    ("Comment cuisiner des pâtes ?", "unknown-topic", "out-of-scope", False),
    ("What is the meaning of life?", "unknown-topic", "out-of-scope", False),
    ("Raconte-moi une blague", "not-understood", "out-of-scope", False),
    ("Write me a poem about the sea", "not-understood", "out-of-scope", False),
]

# Second set, written after all parser fixes and measured once without any further tuning (true held-out).
LANGUAGE_HELDOUT = [
    ("Coucou, tu t'appelles comment ?", "intro", "fr", False), ("Hi, what's your name?", "intro", "en", False),
    ("Dis-moi ce que tu as découvert", "knowledge", "fr", False), ("Tell me what you know", "knowledge", "en", False),
    ("Quelles hypothèses as-tu ?", "knowledge", "fr", False), ("Any beliefs yet?", "knowledge", "en", False),
    ("Quel est ton niveau actuel ?", "stage", "fr", False), ("What's your current status?", "stage", "en", False),
    ("Où sont tes échecs ?", "limitations", "fr", False), ("Tell me your limits", "limitations", "en", False),
    ("Sur quoi as-tu des doutes ?", "questions", "fr", False), ("What gaps do you have?", "questions", "en", False),
    ("Combien de mots connais-tu ?", "vocabulary", "fr", False), ("Which words do you know?", "vocabulary", "en", False),
    ("Aide-moi stp", "help", "fr", False), ("What are your commands?", "help", "en", False),
    ("Tu peux faire des expériences ?", "explore", "fr", False), ("Try exploring a bit", "explore", "en", False),
    ("Tu comprends l'anglais ?", "languages", "fr", False), ("Can you talk in French?", "languages", "en", False),
    ("C'est quoi une suite ?", "define", "fr", False), ("What is a rule?", "define", "en", False),
    ("Ça va bien ?", "smalltalk", "fr", False), ("How's it going?", "smalltalk", "en", False),
    ("Merci GENESIS", "thanks", "fr", False), ("Bye bye", "bye", "en", False),
    ("Quels sont les suivants ?", "predict", "fr", True), ("Predict five more terms", "predict", "en", True),
    ("Explique-moi pourquoi", "explain", "fr", True), ("Why do you believe that?", "explain", "en", True),
    ("N'essaie pas d'explorer", "declined", "negation", False), ("Never predict anything", "declined", "negation", True),
    ("Quelles sont tes connaisances ?", "knowledge", "typo", False), ("Explian your rule", "explain", "typo", True),
    ("What's the population of Japan?", "unknown-topic", "out-of-scope", False),
    ("Pourquoi le ciel est bleu ?", "unknown-topic", "out-of-scope", False),
    ("Quel temps fera-t-il demain ?", "unknown-topic", "out-of-scope", False),
    ("Traduis cette phrase en espagnol", "not-understood", "out-of-scope", False),
]


def run_language_understanding(bench=None) -> dict:
    """Accuracy of the innate FR/EN faculty on designer-written test sentences (no learning involved)."""
    bench = LANGUAGE_BENCH if bench is None else bench
    from .chat import ChatEngine
    with _tmpdb() as path:
        agent = Genesis(path, seed=0)
        eng = ChatEngine(agent)
        rows = []
        for text, gold, cat, needs_seq in bench:
            lang = "en" if cat == "en" else "fr" if cat == "fr" else None
            cid = eng.new_conversation(lang)["id"]
            if needs_seq:
                eng.send(cid, "3, 7, 11, 15, 19, 23")
            got = eng.send(cid, text)["reply"]["meta"]
            rows.append({"text": text, "gold": gold, "got": got["intent"], "category": cat,
                         "ok": got["intent"] == gold, "lang": got["lang"]})
        agent.close()
    by_cat = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], [0, 0])
        c[0] += r["ok"]
        c[1] += 1
    lang_rows = [r for r in rows if r["category"] in ("fr", "en")]
    unsafe = [r for r in rows if r["category"] == "out-of-scope" and r["got"] not in ("unknown-topic",
                                                                                     "not-understood")]
    return {"n": len(rows), "accuracy": round(sum(r["ok"] for r in rows) / len(rows), 4),
            "by_category": {k: {"correct": v[0], "n": v[1]} for k, v in by_cat.items()},
            "lang_detection_accuracy": round(sum(r["lang"] == r["category"] for r in lang_rows) / len(lang_rows), 4),
            "out_of_scope_answered_as_if_known": len(unsafe),
            "errors": [{k: r[k] for k in ("text", "gold", "got")} for r in rows if not r["ok"]]}


# ==================================================================== suite
def run_benchmark_suite(seeds=range(10), quick: bool = False) -> dict:
    seeds = list(seeds)
    t0 = time.time()
    seq = [run_sequence_benchmark(s) for s in seeds]
    budgets = (60, 100) if quick else (60, 100, 150)
    tight = {(b, strat): [run_sequence_benchmark(s, budget=b, strategy=strat, stop_when_idle=False) for s in seeds]
             for b in budgets for strat in ("curiosity", "round_robin", "random")}
    lab_c = [run_lab_benchmark(s, strategy="curiosity") for s in seeds]
    lab_r = [run_lab_benchmark(s, strategy="random") for s in seeds]
    plan = []
    for s in seeds:
        with _tmpdb() as path:
            _, agent, world, sb = run_lab_benchmark(s, db=path, keep_agent=True)
            plan.append(run_planning(agent, sb, world))
            agent.ltm.close()
    mem = [run_memory_benchmark(s) for s in seeds]
    proc_seeds = seeds[: 2 if quick else 5]
    proc_on = [run_procedural_transfer(s, procedural_prior=True) for s in proc_seeds]
    proc_off = [run_procedural_transfer(s, procedural_prior=False) for s in proc_seeds]
    auto_budgets = (150,) if quick else (100, 150, 300)
    auto = {(b, strat): [run_autonomy(s, b, strat) for s in seeds]
            for b in auto_budgets for strat in ("progress", "round_robin", "random")}
    from . import self_improvement as SI
    factory = lambda s: sandboxed(SequenceWorld(s), 10 ** 6)  # noqa: E731
    with _tmpdb() as path:
        Genesis(path).close()
        improve = SI.improve(path, factory, [9001, 9002] if quick else [9001, 9002, 9003],
                             [9101, 9102] if quick else [9101, 9102, 9103],
                             params=("sequence.validation_k",) if quick else SI.DEFAULT_SEARCH)

    def curve(runs):
        return [{"experiments": pts[0]["experiments"], "heldout_unseen": _mean([p["heldout_unseen"] for p in pts])}
                for pts in zip(*[r["learning_curve"] for r in runs])]

    confound = [v for r in lab_c for v in r["confound_after_experiments"].values() if v["confounded"]]
    confound_passive = [v for r in lab_c for v in r["after_passive_watching"]["confound"].values() if v["confounded"]]
    spurious = [a for r in lab_c for a in r["associations"] if a["kind"] == "spurious"]
    causal = [a for r in lab_c for a in r["associations"] if a["kind"] == "causal"]
    summary = {
        "seeds": seeds,
        "wall_seconds": round(time.time() - t0, 1),
        "stage1_patterns": {
            "rule_streams_validated": _mean([r["rule_streams_validated"] for r in seq]),
            "rule_extrapolation_accuracy": _mean([r["rule_extrapolation_accuracy"] for r in seq]),
            "validated_rule_extrapolation_accuracy": _mean([r["validated_rule_extrapolation_accuracy"] for r in seq]),
            "baseline_repeat_last": _mean([r["baseline_repeat_last"] for r in seq]),
            "baseline_linear_extrapolation": _mean([r["baseline_linear"] for r in seq]),
            "noise_flagged_unpredictable": _mean([r["noise_flagged_unpredictable"] for r in seq]),
            "false_validation_rate_on_noise": _mean([r["false_validation_rate"] for r in seq]),
            "observations_to_validation": _mean([r["observations_to_validation"] for r in seq]),
            "interactions_used": _mean([r["interactions"] for r in seq]),
            "cpu_seconds": _mean([r["cpu_seconds"] for r in seq]),
            "expressions_evaluated": _mean([r["expressions_evaluated"] for r in seq]),
            "calibration_ece": _mean([r["calibration"]["ece"] for r in seq]),
            "memory_bytes": _mean([r["memory_bytes"] for r in seq]),
            "failures": [(s, r2["stream"], r2["family"], r2["status"], r2["belief"]) for s, r in zip(seeds, seq)
                         for r2 in r["streams"] if not r2["is_noise"] and r2["status"] != "validated"],
        },
        "attention_budget_sweep": {
            f"budget_{b}/{strat}": {"rule_streams_validated": _mean([r["rule_streams_validated"] for r in runs]),
                    "observations_on_noise": _mean([r["observations_on_noise"] for r in runs])}
            for (b, strat), runs in tight.items()},
        "stage2_3_causal": {
            "accuracy_before_experience": _mean([r["accuracy_before_experience"] for r in lab_c]),
            "after_watching_natural_distribution": _mean(
                [r["after_passive_watching"]["natural_heldout"] for r in lab_c]),
            "after_watching_interventional_distribution": _mean(
                [r["after_passive_watching"]["interventional_heldout"] for r in lab_c]),
            "confounded_rules_correct_after_watching": _mean([v["correct_cause"] and v["status"] == "validated"
                                                              for v in confound_passive]),
            "confounded_rules_correct_after_experiments": _mean([v["correct_cause"] for v in confound]),
            "curve_curiosity": curve(lab_c),
            "curve_random": curve(lab_r),
            "experiments_to_perfect_curiosity": _mean([r["experiments_to_perfect_heldout"] for r in lab_c]),
            "experiments_to_perfect_random": _mean([r["experiments_to_perfect_heldout"] for r in lab_r]),
            "runs_reaching_perfect_curiosity": _mean([r["experiments_to_perfect_heldout"] is not None for r in lab_c]),
            "runs_reaching_perfect_random": _mean([r["experiments_to_perfect_heldout"] is not None for r in lab_r]),
            "final_heldout_unseen": _mean([r["final_heldout_unseen"] for r in lab_c]),
            "transfer_out_of_range": _mean([r["final_transfer_out_of_range"] for r in lab_c]),
            "tabular_baseline_heldout": _mean([r["tabular_baseline_heldout"] for r in lab_c]),
            "tabular_baseline_transfer": _mean([r["tabular_baseline_transfer"] for r in lab_c]),
            "no_change_baseline": _mean([r["no_change_baseline_heldout"] for r in lab_c]),
            "associations_causal": len(causal),
            "associations_spurious": len(spurious),
            "spurious_associations_matching_a_rule": sum(a["kind"] == "spurious" and '"matches_validated_rule_condition": true' in (a["detail"] or "") for a in spurious),
            "cpu_seconds": _mean([r["cpu_seconds"] for r in lab_c]),
            "calibration_ece": _mean([r["calibration"]["ece"] for r in lab_c]),
        },
        "stage4_memory": {
            "accuracy_world_a": _mean([m["accuracy_world_a"] for m in mem]),
            "retention_after_reload": _mean([m["accuracy_world_a_after_reload"] / m["accuracy_world_a"]
                                             if m["accuracy_world_a"] else None for m in mem]),
            "accuracy_world_b": _mean([m["accuracy_world_b"] for m in mem]),
            "forgetting_after_new_world": _mean([m["forgetting"] for m in mem]),
            "replay_consistency": _mean([m["replay_consistency"] for m in mem]),
        },
        "stage5_planning": {
            "plan_success": _mean([p["plan_success"] for p in plan]),
            "random_policy_success": _mean([p["random_success"] for p in plan]),
            "plan_mean_steps": _mean([p["plan_mean_steps"] for p in plan]),
            "random_policy_mean_steps_when_successful": _mean([p["random_mean_steps"] for p in plan]),
        },
        "procedural_meta_learning": {
            "with_prior": [p["observations_to_validation_per_world"] for p in proc_on],
            "without_prior": [p["observations_to_validation_per_world"] for p in proc_off],
            "late_worlds_with_prior": _mean([p["late_worlds_mean"] for p in proc_on]),
            "late_worlds_without_prior": _mean([p["late_worlds_mean"] for p in proc_off]),
            "late_search_cost_per_rule_stream_with_prior": _mean(
                [p["late_worlds_search_cost_per_rule_stream"] for p in proc_on]),
            "late_search_cost_per_rule_stream_without_prior": _mean(
                [p["late_worlds_search_cost_per_rule_stream"] for p in proc_off]),
            "late_cpu_seconds_per_world_with_prior": _mean([p["late_worlds_cpu"] for p in proc_on]),
            "late_cpu_seconds_per_world_without_prior": _mean([p["late_worlds_cpu"] for p in proc_off]),
            "late_accuracy_with_prior": _mean([p["late_worlds_accuracy"] for p in proc_on]),
            "late_accuracy_without_prior": _mean([p["late_worlds_accuracy"] for p in proc_off]),
            "library_search_share_with_prior": [p["library_search_share_per_world"] for p in proc_on],
        },
        "stage6_autonomy": {
            f"budget_{b}/{strat}": {
                "knowledge_score": _mean([r["knowledge_score"] for r in runs]),
                "noise_share": _mean([r["noise_share"] for r in runs]),
                "interactions_used": _mean([r["interactions"] for r in runs]),
                "objectives_achieved": _mean([r["objectives"]["achieved"] for r in runs]),
                "per_seed_knowledge": [r["knowledge_score"] for r in runs]}
            for (b, strat), runs in auto.items()},
        "stage7_self_improvement": {
            "reproducible": improve["reproducible"], "decision": improve["decision"],
            "baseline_trial": improve["baseline"]["score"],
            "variants": [(v["change"], v["trial"]["score"], v["gain"]) for v in improve.get("variants", [])],
            "candidate": improve.get("candidate"),
        },
    }
    return summary
