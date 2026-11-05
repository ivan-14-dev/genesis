# GENESIS — Measured results (deliverable J)

Reproduce:

```bash
python -m genesis benchmark --seeds 100-109 --out results/benchmark-seeds100-109.json
```

- **Seeds and data:** 10 held-out seeds (100–109), not used during development. Every world's rules are
  sampled from its seed and never shown to the agent. The raw output is in
  [`../results/benchmark-seeds100-109.json`](../results/benchmark-seeds100-109.json).
- **Runtime:** about 15 minutes wall time on one CPU core for the whole suite (three strategies, budget sweep,
  ablations, stage 6 allocation, one stage 7 self-improvement cycle).
- **Determinism:** results do not depend on `PYTHONHASHSEED` (a set ordering that did was fixed; regression
  test `test_behaviour_independent_of_hash_seed`).
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
- **Interactions:** 194 in total, including spaced reviews.
- **Compute:** about 2.5 s CPU and about 1.2M candidate programs evaluated (procedural memory now shortcuts
  the search, see below).
- **Memory:** about 190 KB.
- **Calibration:** ECE 0.065.

The single failure is seed 104, stream S6 (a modular counter). The needed modulus constant was never
proposed, so the agent labelled the stream *unpredictable* rather than inventing a rule. This is the honest
failure mode, but it is still a failure of the constant-proposal heuristic.

### Attention / curiosity allocation (fraction of rule streams validated within a fixed budget)

| Budget (observations) | Curiosity | Round-robin | Random |
|---|---|---|---|
| 60 | **60.0%** | 0.0% | 23.8% |
| 100 | **92.5%** | 91.3% | 73.8% |
| 150 | **98.8%** | 96.3% | 91.3% |

**Fix of the earlier negative result (60 observations: curiosity 0%).** Curiosity used to spread samples
breadth-first, so no stream reached the validation threshold. The agent now applies *budget-aware focus*
(`Genesis._focus`): when the remaining budget cannot cover the expected cost of every open problem, it finishes
the cheapest open problem first. The expected cost is the median number of observations its own past problems
needed (learned, default 8 before any experience). Problems that already took more than twice that are set
aside.

Remaining negative: curiosity still spends slightly more observations on noise (15.2 vs 12 for round-robin at
budget 60; 22.1 vs 20 at 100; 32.7 vs 30 at 150), because noise must be observed at least 8 times before it can
be judged unpredictable.

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
| Random experiments | 27.2% | 74.3% | 83.6% | 96.6% | 97.3% | 99.2% | 99.6% |

Experiments needed to reach 100%: **24.5** with curiosity (10/10 runs) vs 51.7 with random experiments
(only 9/10 runs reach 100% within 120 experiments).

### Confounded rules

| Measure | Result |
|---|---|
| Confounded rules with the true cause identified after passive watching | 0% |
| Confounded rules with the true cause identified after experiments | **100%** |
| Associations labelled causal / spurious | 7 / 5 |
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

Six worlds in sequence, later worlds 4–6, 5 seeds. With the prior, shapes that worked before are instantiated
first (*library-first search*); full enumeration is skipped when one of them fits the data exactly.

| Metric (later worlds) | With procedural memory | Without |
|---|---|---|
| Candidate programs evaluated per rule stream | **2,343** | 39,099 (16.7×) |
| CPU seconds per world | 2.38 | 2.52 (−6%) |
| Observations to validation | 8.11 | 8.25 |
| Extrapolation accuracy | 100% | 100% |

**Fixed:** remembering rule shapes now gives a large, measured reduction in search. **Still small:** wall-clock
gain is only 6% (noise streams, which have no remembered shape, dominate the CPU), and observations to
validation barely change, because validation requires 4 correct *forward* predictions whatever the search does.

## Stage 6 — autonomous exploration (`autonomy.py`)

Four worlds, one shared budget, no instruction on where to look: a sequence world, a pure-noise world and two
causal labs. The explorer allocates chunks of 10 interactions. Each world is probed twice, then chosen by
measured learning progress (|Δ accuracy| per step + newly validated rules; "unpredictable" verdicts don't count)
plus a small optimism bonus 0.1/√chunks. It stops when no world shows progress ≥ 0.05. Knowledge gaps become
stored objectives (achieved / abandoned as unpredictable). Knowledge score = mean held-out accuracy over the
three learnable worlds.

| Budget | Learning progress | Round-robin | Random |
|---|---|---|---|
| 100 | 0.702 (noise share 20%) | **0.723** (30%) | 0.663 (31%) |
| 150 | **0.849** (13%) | 0.792 (28%) | 0.772 (32%) |
| 300 | 0.980 (18%, stopped itself after 229) | 0.975 (40%, 295) | 0.981 (41%, 295) |

- At 150 it beats round-robin by +5.7 points and spends half as much on noise.
- At 300 it reaches the same knowledge using 22% fewer interactions, by stopping itself.
- **Negative:** at 100 it loses to round-robin. The two probe chunks per world cost 80 of the 100 interactions,
  so there is almost nothing left to exploit.
- On demo seed 0 at 150, random allocation won (86.2% vs 84.7%); per-seed variance is large.

## Stage 7 — controlled self-improvement (`self_improvement.py`)

Proposes one-parameter variants from an allowlist (safety parameters cannot be changed:
`PermissionError`). Evaluates each in isolated temporary agents with an *observable* score: predict 5 terms,
then observe them. It refuses to run if the safety digest has changed, and checks reproducibility by evaluating
the baseline twice. A variant must gain ≥ 0.01 on trial seeds and be **strictly better** on holdout seeds with no
more false validations. It is then stored as `pending-approval`; only a named human can `approve` it, and
`rollback` restores the previous configuration.

| Item | Result |
|---|---|
| Evaluation reproducible | yes |
| Baseline on trial seeds | 0.713 |
| Best variant | `sequence.validation_k` 4 → 3: trial +0.033 |
| Holdout (seeds 9101–9103) | 0.747 vs 0.740, same false validations (1) → `pending-approval` |
| Other variants (`shape_bias`, `default_resolution_cost`, `novelty_weight`) | exactly 0 effect at this budget |

**Honest reading:** the confirmed gain is small (+0.007 on holdout, one prediction in 150). It trades evidence for
speed (fewer forward checks), which a human should weigh before approving. Limitations found in its own records
(stuck problems, streams judged unpredictable, blocked gates) are reported by `limitations()`.

## Word learning in the chat (`lexicon.py`, `experiments.run_word_learning`)

A simulated user sends 15 training sentences (5 commands × 3 paraphrases, none containing an innate keyword) in
random order, 4 passes. It answers questions truthfully and names the intended command when GENESIS doesn't
understand. Evaluation uses 15 **unseen** paraphrases, read without learning, plus 6 unrelated control sentences.
The figures are the means over seeds 100–109.

| Pass | not understood | proposed, wrong | proposed, correct | acted alone, correct | acted alone, wrong |
|---|---|---|---|---|---|
| 1 | 9.0 | 4.4 | 1.6 | 0 | 0 |
| 2 | 0 | 0 | 14.2 | 0.8 | 0 |
| 3 | 0 | 0 | 12.9 | 2.1 | 0 |
| 4 | 0 | 0 | 1.2 | 13.8 | 0 |

| Metric | Value |
|---|---|
| Vocabulary learned / validated mappings | 38 words / 28.1 |
| Unseen paraphrases: top interpretation correct | **100%** (chance 9.1%), min over seeds 100% |
| Unseen paraphrases: would act alone / precision | 73% / **100%** |
| Control sentences acted on / asked about | 0 / 1 ("quelle heure est-il", via « est » learned from « qu'est-ce qui… ») |
| User exchanges for the whole run | 107.7 |

Limits: the unseen paraphrases share content words with the training ones by construction. A sentence with no
taught word stays "not understood". Function words are only diluted statistically, not understood. This is
association, not language comprehension.

## Developmental gating

On the demo run (seed 0), the agent passes the gates for stages 0–7 one after another, each based on measured
metrics. The stage 6 gate requires a gain ≥ 0.02 over round-robin and ≤ 25% of the budget spent on noise. The
stage 7 gate requires a reproducible evaluation, an unchanged safety policy, no holdout regression and at least
one self-identified limitation. It does **not** require any change to be deployed.
