# GENESIS — Measured results (deliverable J)

Reproduce:

```bash
python -m genesis benchmark --seeds 100-109 --out results/benchmark-seeds100-109.json
```

- **Seeds and data:** 10 held-out seeds (100–109), not used during development. Every world's rules are
  sampled from its seed and never shown to the agent. The raw output is in
  [`../results/benchmark-seeds100-109.json`](../results/benchmark-seeds100-109.json).
- **Runtime:** about 10 minutes wall time on one CPU core for the whole suite (three strategies, budget sweep,
  ablations).
- **Accuracy definition:** every accuracy below is measured against the *hidden* rules, on inputs the agent has
  not experienced.

## Why this demonstrates acquired, not given, knowledge

1. **Nothing is known before experience.** Accuracy before any experience is **0.0**. The manifest lists no
   domain knowledge, and the stage-0 audit verifies that memory is empty.
2. **Rules cannot be hard-coded.** Rules are sampled per seed, and these seeds were unseen during development.
3. **The agent cannot see the rules.** It reaches the world only through a whitelisting sandbox, and a static
   test (`test_cognitive_modules_never_touch_world_internals`) verifies that agent code never references hidden
   members.
4. **It generalises rather than memorises.** It predicts configurations it never experienced, including
   **out-of-range values** it never saw, where a memorising (tabular) baseline fails.

## Stage 1 — pattern discovery (10 streams per seed: 8 rule streams + 2 noise streams)

| Metric | GENESIS | Repeat-last baseline | Linear-extrapolation baseline |
|---|---|---|---|
| Accuracy extrapolating 10 unseen terms (rule streams) | **98.75%** | 7.6% | 16.0% |
| Rule streams validated | 79 / 80 | – | – |
| Accuracy of *validated* rules | **100%** | – | – |
| Noise streams judged "unpredictable" | **20 / 20** | – | – |
| False validations on noise | **0** | – | – |

Cost per seed:
- **Observations per stream to validation:** 8.4 on average.
- **Interactions:** 214 in total, including spaced reviews.
- **Compute:** about 6 s CPU and about 2.1M candidate programs evaluated.
- **Memory:** about 190 KB.
- **Calibration:** ECE 0.066.

The single failure is seed 104, stream S6 (a modular counter). The needed modulus constant was never
proposed, so the agent labelled the stream *unpredictable* rather than inventing a rule. This is the honest
failure mode, but it is still a failure of the constant-proposal heuristic.

### Attention / curiosity allocation (fraction of rule streams validated within a fixed budget)

| Budget (observations) | Curiosity | Round-robin | Random |
|---|---|---|---|
| 60 | 0.0% | 0.0% | **23.8%** |
| 100 | **96.3%** | 91.3% | 73.8% |
| 150 | **98.8%** | 96.3% | 90.0% |

Curiosity is best once the budget allows each stream about 11 observations.

**Negative result:** at 60 observations, curiosity spreads its samples breadth-first (like round-robin), so no
stream reaches the validation threshold. Random sampling validates some streams by chance concentration.
Curiosity also does not reduce observations spent on noise (about 22 vs 20 for round-robin at budget 100),
because noise must be observed at least 8 times before it can be judged unpredictable.

Allocation by learning progress (focusing on streams likely to resolve) is on the roadmap (stage 6).

## Stages 2–3 — relationships and causal discovery (CausalWorld: 4 properties, 4 actions, ≥ 1 confounded rule)

| Condition | Held-out accuracy (unseen configurations) |
|---|---|
| Before any experience | 0.0% |
| After 60 passive observations, on the natural distribution | 98.9% |
| After 60 passive observations, under intervention | **27.8%** (the confound misleads) |
| After experiments (curiosity) | **100%** |
| Transfer to out-of-range values never experienced | **100%** |
| Tabular memorisation baseline | 46.4% (transfer: 46.0%) |
| "Nothing changes" baseline | 44.6% |

Learning curve (mean held-out accuracy on unseen configurations, 10 seeds):

| Experiments | 0 | 5 | 10 | 20 | 40 | 80 | 120 |
|---|---|---|---|---|---|---|---|
| Curiosity (EIG) | 27.2% | **99.0%** | 99.3% | 99.3% | 99.3% | 99.7% | 100% |
| Random experiments | 27.2% | 74.3% | 83.6% | 96.6% | 97.3% | 99.2% | 100% |

Experiments needed to reach 100%: **24.5** with curiosity vs 58.5 with random experiments (mean over seeds).

### Confounded rules

| Measure | Result |
|---|---|
| Confounded rules with the true cause identified after passive watching | 0% |
| Confounded rules with the true cause identified after experiments | **100%** |
| Associations labelled causal / spurious | 7 / 6 |
| Spurious labels that coincide with a real rule condition | 0 |

## Stage 4 — memory

| Metric | Value |
|---|---|
| Accuracy on world A after a restart (new session) | 100% (retention 1.0) |
| Accuracy on world B learned afterwards | 100% |
| Forgetting of world A after learning world B | **0.0** |
| Replay consistency of stored beliefs vs episodes | 1.0 |

Caveat: knowledge is stored as separate symbolic rules per context, so interference is structurally
impossible here. This is not evidence about catastrophic forgetting in neural learners.

## Stage 5 — planning to unseen goals

| Policy | Success | Mean steps when successful |
|---|---|---|
| BFS planning over the learned model | **100%** | 1.29 |
| Random actions (8 steps max) | 78.5% | 3.35 |

Caveat: the current tasks are easy (random actions often succeed). Harder multi-step goals are needed to
measure reasoning more seriously.

## Meta-learning (procedural memory of successful rule shapes)

Mean observations to validation in later worlds (worlds 4–6 of 6, 5 seeds): **8.23 with** the procedural prior
vs **8.25 without** it.

**No measurable effect.** Validation is limited by the forward-evidence requirement (4 correct predictions),
not by the search. A useful test of this mechanism needs a larger hypothesis space, where the prior changes
what is found first (roadmap phase 3, library learning).

## Developmental gating

On the demo run (seed 0), the agent passes the gates for stages 0–5 one after another, each based on measured
metrics. It stops at stage 6 because that gate is explicitly *not implemented*.
